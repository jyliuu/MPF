use ndarray::{Array1, ArrayView1, ArrayView2};
use rand::Rng;

use crate::FitResult;

use crate::grid::params::{SplitStrategyParams, TreeGridParams};
use crate::grid::splitting::{IntervalRandomSplit, RandomSplit, SplitStrategy};
use crate::grid::FittedTreeGrid;

use super::grid_index::GridIndex;
use super::identification::{IdentificationStrategy, L1Identification, L2Identification};
use super::params::IdentificationStrategyParams;
use super::reproject_values::reproject_grid_values;

pub fn fit<R: Rng + ?Sized>(
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
    hyperparameters: &TreeGridParams,
    rng: &mut R,
) -> (FitResult, FittedTreeGrid) {
    let fitter = TreeGridFitter::new(x.view(), y.view());

    let split_strategy = match hyperparameters.split_strategy_params {
        SplitStrategyParams::RandomSplit {
            split_try,
            colsample_bytree,
        } => SplitStrategy::Random(RandomSplit {
            split_try,
            colsample_bytree,
        }),
        SplitStrategyParams::IntervalRandomSplit {
            split_try,
            colsample_bytree,
        } => SplitStrategy::Interval(IntervalRandomSplit {
            split_try,
            colsample_bytree,
        }),
    };

    let (fit_res, mut tree_grid) = fitter.fit(
        hyperparameters.n_iter,
        hyperparameters.reproject_grid_values,
        &split_strategy,
        rng,
    );

    if hyperparameters.identification_strategy_params != IdentificationStrategyParams::None {
        let weights: Vec<Vec<f64>> = tree_grid
            .grid_index
            .observation_counts
            .iter()
            .map(|v| v.iter().map(|&x| x as f64).collect())
            .collect();
        match hyperparameters.identification_strategy_params {
            IdentificationStrategyParams::L1 => {
                L1Identification::identify(
                    &mut tree_grid.grid_values,
                    &weights,
                    &mut tree_grid.scaling,
                );
            }
            IdentificationStrategyParams::L2 => {
                L2Identification::identify(
                    &mut tree_grid.grid_values,
                    &weights,
                    &mut tree_grid.scaling,
                );
            }
            IdentificationStrategyParams::None => unreachable!(),
        }
    }

    (fit_res, tree_grid)
}

#[derive(Debug)]
pub struct TreeGridFitter<'a> {
    pub grid_index: GridIndex,
    pub grid_values: Vec<Vec<f64>>,
    pub labels: ArrayView1<'a, f64>,
    pub x: ArrayView2<'a, f64>,
    pub y_hat: Array1<f64>,
    pub residuals: Array1<f64>,
    pub scaling: f64,
}

impl TreeGridFitter<'_> {
    pub fn update_tree(&mut self, refine_candidate: RefineCandidate) {
        let RefineCandidate {
            col,
            split,
            index,
            update_a,
            update_b,
            cells,
        } = refine_candidate;

        let old_grid_value = self.grid_values[col][index];
        self.grid_values[col][index] *= update_a;
        self.grid_values[col].insert(index + 1, old_grid_value * update_b);

        let (pos_a, pos_b) = self
            .grid_index
            .split_axis(&cells, col, split, self.x.view());

        let left_grid_cells = self.grid_index.collect_fixed_axis_cells(col, pos_a);
        let right_grid_cells = self.grid_index.collect_fixed_axis_cells(col, pos_b);

        for &cell_idx in &left_grid_cells {
            let points = self.grid_index.cells.get(&cell_idx).unwrap();
            for &i in points {
                self.y_hat[i] *= update_a;
                self.residuals[i] = self.labels[i] - self.y_hat[i];
            }
        }
        for &cell_idx in &right_grid_cells {
            let points = self.grid_index.cells.get(&cell_idx).unwrap();
            for &i in points {
                self.y_hat[i] *= update_b;
                self.residuals[i] = self.labels[i] - self.y_hat[i];
            }
        }
    }
}

impl<'a> TreeGridFitter<'a> {
    pub fn new(x: ArrayView2<'a, f64>, y: ArrayView1<'a, f64>) -> Self {
        let mean = y.mean().unwrap();
        let init_value: f64 = mean.abs().powf(1.0 / x.ncols() as f64);
        let sign = mean.signum();
        let mut grid_values = vec![vec![init_value]; x.ncols() - 1];
        grid_values.insert(0, vec![sign * init_value]);
        let y_hat = Array1::from_vec(vec![mean; x.nrows()]);
        let residuals = y.to_owned() - &y_hat;

        let grid_index = GridIndex::new(x.view());

        TreeGridFitter {
            grid_index,
            grid_values,
            labels: y,
            x,
            y_hat,
            residuals,
            scaling: 1.0,
        }
    }

    fn fit<R>(
        mut self,
        n_iter: usize,
        reproject: bool,
        split_strategy: &SplitStrategy,
        rng: &mut R,
    ) -> (FitResult, FittedTreeGrid)
    where
        R: Rng + ?Sized,
    {
        // Main fitting loop
        for _ in 0..n_iter {
            let intervals = &self.grid_index.intervals;

            let splits = split_strategy.sample_splits(rng, self.x.view(), intervals);

            // Select best candidate based on strategy
            let best_candidate = {
                let mut best_candidate = None;
                let mut best_err_diff = f64::NEG_INFINITY;

                for (col, split) in splits {
                    let refine_candidate_res = find_refine_candidate(
                        split,
                        col,
                        &self.grid_index,
                        self.residuals.view(),
                        self.y_hat.view(),
                        self.x.view(),
                    );
                    if let Ok((err_new, err_old, refine_candidate)) = refine_candidate_res {
                        let err_diff = err_old - err_new;
                        if err_diff > best_err_diff {
                            best_candidate = Some(refine_candidate);
                            best_err_diff = err_diff;
                        }
                    }
                }

                best_candidate
            };

            // Update tree with best candidate
            if let Some(candidate) = best_candidate {
                self.update_tree(candidate);
            }
        }

        let err = self.residuals.pow2().mean().unwrap();
        if reproject {
            reproject_grid_values(
                self.labels.view(),
                self.y_hat.view_mut(),
                self.residuals.view_mut(),
                &self.grid_index,
                &mut self.grid_values,
            );
        }

        let residuals = self.residuals;
        let y_hat = self.y_hat;

        let tree_grid = FittedTreeGrid::new(self.grid_values, self.scaling, self.grid_index);

        let fit_res = FitResult {
            err,
            residuals,
            y_hat,
        };

        (fit_res, tree_grid)
    }
}

#[derive(Debug)]
pub struct RefineCandidate {
    pub col: usize,
    pub split: f64,
    pub index: usize,
    pub update_a: f64,
    pub update_b: f64,
    pub cells: Vec<usize>,
}

pub fn find_refine_candidate(
    split: f64,
    col: usize,
    grid_index: &GridIndex,
    residuals: ArrayView1<'_, f64>,
    y_hat: ArrayView1<'_, f64>,
    x: ArrayView2<'_, f64>,
) -> Result<(f64, f64, RefineCandidate), String> {
    let index = grid_index.compute_col_index_for_point(col, split);
    let cells = grid_index.collect_fixed_axis_cells(col, index);

    // Initialize accumulators
    let (mut n_a, mut n_b) = (0.0, 0.0);
    let (mut m_a, mut m_b) = (0.0, 0.0);
    let mut err_old = 0.0;
    // Single pass through data
    for cell_idx in &cells {
        let points = grid_index.cells.get(cell_idx).unwrap();

        // The value of one observation is the same for all points in the cell
        if let Some(v) = y_hat.get(points[0]) {
            let v_pow2 = v.powi(2);

            for &i in points {
                let x_val = x[[i, col]];
                let res = residuals[i];
                err_old += res.powi(2);

                // Use if-else instead of match for slight perf gain
                if x_val < split {
                    n_a += v_pow2;
                    m_a += res * v;
                } else {
                    n_b += v_pow2;
                    m_b += res * v;
                }
            }
        }
    }

    // Mathematical optimization: Calculate error difference directly
    let err_new = if n_a > 0.0 && n_b > 0.0 {
        err_old - (m_a.powi(2) / n_a + m_b.powi(2) / n_b)
    } else {
        return Err("No points to update".to_string());
    };

    // Compute update values
    let update_a = m_a / n_a + 1.0;
    let update_b = m_b / n_b + 1.0;

    Ok((
        err_new,
        err_old,
        RefineCandidate {
            col,
            split,
            index,
            update_a,
            update_b,
            cells,
        },
    ))
}

#[cfg(test)]
mod tests {

    use ndarray::Array2;

    use super::*;

    pub fn setup_data_hardcoded() -> (Array2<f64>, Array1<f64>) {
        // Returns hardcoded test data
        let dat = Array2::from_shape_vec(
            (20, 3),
            vec![
                1.99591859675606,
                -1.00591398212174,
                -1.47348548786449,
                -0.290104994456985,
                -0.507165037206491,
                -0.0992586787398358,
                -0.392657438987142,
                1.41894909677495,
                -0.674415207533763,
                0.541774508623095,
                0.134065164928921,
                0.634093564547107,
                0.981026718908818,
                0.29864258176132,
                1.29321982986182,
                2.14226826821187,
                -1.57541477899575,
                -1.20864031274097,
                0.614259969810645,
                -1.11273947093321,
                -0.747582520955759,
                0.742939152591961,
                0.367035148375779,
                0.629260294753607,
                -2.90764791321527,
                1.81674051159666,
                -1.27652692983198,
                -1.94290907058012,
                2.5208012003232,
                -0.871450106365531,
                0.272189306719476,
                1.01227462627796,
                -0.356579330585395,
                0.481004283284028,
                0.165976176377298,
                0.822063375479486,
                -0.245353149162764,
                -1.40974327898294,
                -0.334709204672301,
                -0.00460477602051997,
                0.0117210317817887,
                2.69171682068671,
                0.359824874802531,
                0.821234081234943,
                -0.318909828758849,
                -1.88722434288848,
                -1.01377986818573,
                0.400700584291665,
                -0.141615483262696,
                0.128123583066683,
                -1.59321040126916,
                0.136218360404787,
                0.112778041636902,
                0.0942204942429378,
                2.20921149756541,
                0.882698443188986,
                0.852817759799762,
                -2.73007802370526,
                -1.21615404372871,
                0.633442434384728,
            ],
        )
        .unwrap();
        let y = dat.slice(ndarray::s![.., 0]).to_owned();
        let x = dat.slice(ndarray::s![.., 1..]).to_owned();
        (x, y)
    }

    macro_rules! assert_float_eq {
        ($x:expr, $y:expr, $d:expr) => {
            assert!(($x - $y).abs() < $d);
        };
    }

    #[test]
    fn test_tree_grid_slice_refine_candidate() {
        let (x, y) = setup_data_hardcoded();
        let tree_grid = TreeGridFitter::new(x.view(), y.view());
        let (_, _, refine_candidate) = find_refine_candidate(
            1.0,
            0,
            &tree_grid.grid_index,
            tree_grid.residuals.view(),
            tree_grid.y_hat.view(),
            x.view(),
        )
        .unwrap();
        assert_float_eq!(refine_candidate.update_a, -93.53056943616252, 1e-10);
        assert_float_eq!(refine_candidate.update_b, 379.12227774465015, 1e-10);
    }

    #[test]
    fn test_tree_grid_multiple_refines() {
        let (x, y) = setup_data_hardcoded();
        let find_refine_candidate_closure =
            |tree_grid: &TreeGridFitter<'_>, split, col| -> (f64, f64, RefineCandidate) {
                find_refine_candidate(
                    split,
                    col,
                    &tree_grid.grid_index,
                    tree_grid.residuals.view(),
                    tree_grid.y_hat.view(),
                    x.view(),
                )
                .unwrap()
            };
        let mut tree_grid = TreeGridFitter::new(x.view(), y.view());
        {
            let (err_new, err_old, refine_candidate) =
                find_refine_candidate_closure(&tree_grid, x[[8, 0]], 0);

            assert_float_eq!(err_new, 26.61921453887834, 1e-10);
            assert_float_eq!(err_old, 39.65496224200821, 1e-10);
            assert_float_eq!(refine_candidate.update_a, -81.09657885629389, 1e-10);
            assert_float_eq!(refine_candidate.update_b, 739.869209706645, 1e-10);
            tree_grid.update_tree(refine_candidate);
            assert_eq!(
                tree_grid.grid_index.observation_counts,
                vec![vec![18, 2], vec![20]]
            )
        }
        {
            let (err_new, err_old, refine_candidate) =
                find_refine_candidate_closure(&tree_grid, x[[12, 0]], 0);

            assert_float_eq!(err_new, 22.425727245741378, 1e-10);
            assert_float_eq!(err_old, 26.153854021633833, 1e-10);
            assert_float_eq!(refine_candidate.update_a, 8.058693908258093, 1e-10);
            assert_float_eq!(refine_candidate.update_b, 0.5847827112789354, 1e-10);
            tree_grid.update_tree(refine_candidate);
            assert_eq!(
                tree_grid.grid_index.observation_counts,
                vec![vec![1, 17, 2], vec![20]]
            );
        }
        {
            let (err_new, err_old, refine_candidate) =
                find_refine_candidate_closure(&tree_grid, x[[0, 0]], 0);

            assert_float_eq!(err_new, 14.671443409333211, 1e-10);
            assert_float_eq!(err_old, 22.425727245741378, 1e-10);
            assert_float_eq!(refine_candidate.update_a, -6.832210436114132, 1e-10);
            assert_float_eq!(refine_candidate.update_b, 3.409910903419733, 1e-10);
            tree_grid.update_tree(refine_candidate);
            assert_eq!(
                tree_grid.grid_index.observation_counts,
                vec![vec![1, 4, 13, 2], vec![20]]
            );
        }
        {
            let (err_new, err_old, refine_candidate) =
                find_refine_candidate_closure(&tree_grid, x[[15, 1]], 1);

            assert_float_eq!(err_new, 11.248401175931308, 1e-10);
            assert_float_eq!(err_old, 15.136803926577713, 1e-10);
            assert_float_eq!(refine_candidate.update_a, 0.8256919200067316, 1e-10);
            assert_float_eq!(refine_candidate.update_b, 1.9098338089112117, 1e-10);
            tree_grid.update_tree(refine_candidate);
            assert_eq!(
                tree_grid.grid_index.observation_counts,
                vec![vec![1, 4, 13, 2], vec![12, 8]]
            );
        }
        {
            let (err_new, err_old, refine_candidate) =
                find_refine_candidate_closure(&tree_grid, x[[6, 1]], 1);

            assert_float_eq!(err_new, 3.9199063400989016, 1e-10);
            assert_float_eq!(err_old, 7.692086523143316, 1e-10);
            assert_float_eq!(refine_candidate.update_a, 1.2412453820291502, 1e-10);
            assert_float_eq!(refine_candidate.update_b, -0.11462749920417181, 1e-10);
            tree_grid.update_tree(refine_candidate);
            assert_eq!(
                tree_grid.grid_index.observation_counts,
                vec![vec![1, 4, 13, 2], vec![5, 7, 8]]
            );
        }
        {
            let (err_new, err_old, refine_candidate) =
                find_refine_candidate_closure(&tree_grid, x[[13, 0]], 0);

            assert_float_eq!(err_new, 3.5819148122645306, 1e-10);
            assert_float_eq!(err_old, 6.104930626516536, 1e-10);
            assert_float_eq!(refine_candidate.update_a, 3.691670930826113, 1e-10);
            assert_float_eq!(refine_candidate.update_b, 0.7617522326677412, 1e-10);
            tree_grid.update_tree(refine_candidate);
            assert_eq!(
                tree_grid.grid_index.observation_counts,
                vec![vec![1, 4, 2, 11, 2], vec![5, 7, 8]]
            );
        }
        {
            let (err_new, err_old, refine_candidate) =
                find_refine_candidate_closure(&tree_grid, x[[14, 0]], 0);

            assert_float_eq!(err_new, 1.1863917023933575, 1e-10);
            assert_float_eq!(err_old, 3.5708191124576096, 1e-10);
            assert_float_eq!(refine_candidate.update_a, 0.6518311983217222, 1e-10);
            assert_float_eq!(refine_candidate.update_b, 2.828492111820738, 1e-10);
            tree_grid.update_tree(refine_candidate);
            assert_eq!(
                tree_grid.grid_index.observation_counts,
                vec![vec![1, 4, 2, 7, 4, 2], vec![5, 7, 8]]
            );
        }
    }
}
