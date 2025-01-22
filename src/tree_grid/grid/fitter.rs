use std::vec;

use itertools::Itertools;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use rand::{seq::index::sample, Rng};

use crate::{FitResult, ModelFitter};

use super::FittedTreeGrid;

#[derive(Debug)]
pub struct TreeGridFitter<'a> {
    pub splits: Vec<Vec<f64>>,
    pub intervals: Vec<Vec<(f64, f64)>>,
    pub grid_values: Vec<Vec<f64>>,
    pub leaf_points: Array2<usize>,
    pub labels: ArrayView1<'a, f64>,
    pub x: ArrayView2<'a, f64>,
    pub y_hat: Array1<f64>,
    pub residuals: Array1<f64>,
}

#[derive(Debug, Clone)]
pub struct TreeGridParams {
    pub n_iter: usize,
    pub split_try: usize,
    pub colsample_bytree: f64,
}

impl Default for TreeGridParams {
    fn default() -> Self {
        TreeGridParams {
            n_iter: 50,
            split_try: 10,
            colsample_bytree: 1.0,
        }
    }
}

#[derive(Debug)]
pub struct SliceCandidate {
    col: usize,
    split: f64,
    index: usize,
    left: (f64, f64),
    right: (f64, f64),
}

#[derive(Debug)]
pub struct RefineCandidate {
    pub col: usize,
    pub split: f64,
    index: usize,
    left: (f64, f64),
    right: (f64, f64),
    pub update_a: f64,
    pub update_b: f64,
    a_points_idx: Vec<usize>,
    b_points_idx: Vec<usize>,
    curr_leaf_points_idx: Vec<usize>,
}

fn compute_initial_values(
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
) -> (f64, Vec<Vec<f64>>, Array1<f64>) {
    let mean = y.mean().unwrap();
    let init_value: f64 = mean.abs().powf(1.0 / x.ncols() as f64);
    let sign = mean.signum();
    let mut grid_values = vec![vec![init_value]; x.ncols() - 1];
    grid_values.insert(0, vec![sign * init_value]);
    let y_hat = Array1::from_vec(vec![mean; x.nrows()]);
    (mean, grid_values, y_hat)
}

pub fn find_slice_candidate(
    splits: &[Vec<f64>],
    intervals: &[Vec<(f64, f64)>],
    col: usize,
    split: f64,
) -> SliceCandidate {
    let splits = &splits[col];
    let intervals = &intervals[col];

    let index = splits
        .iter()
        .position(|&x| split < x)
        .unwrap_or(splits.len());

    let (begin, end) = intervals[index];
    let left = (begin, split);
    let right = (split, end);

    SliceCandidate {
        col,
        split,
        index,
        left,
        right,
    }
}

fn compute_leaf_values(leaf: &Array1<usize>, grid_values: &[Vec<f64>]) -> f64 {
    leaf.indexed_iter()
        .map(|(i, &idx)| grid_values[i][idx])
        .product()
}

pub fn find_refine_candidate(
    slice_candidate: SliceCandidate,
    x: ArrayView2<f64>,
    leaf_points: &Array2<usize>,
    grid_values: &[Vec<f64>],
    intervals: &[Vec<(f64, f64)>],
    residuals: ArrayView1<'_, f64>,
    y_hat: ArrayView1<'_, f64>,
) -> (f64, f64, RefineCandidate) {
    let SliceCandidate {
        col,
        split,
        index,
        left,
        right,
    } = slice_candidate;

    let mut dims = vec![];
    for dim in 0..x.ncols() {
        dims.push(if dim == col {
            vec![index]
        } else {
            (0..intervals[dim].len()).collect()
        });
    }
    let leaves: Vec<Array1<usize>> = dims
        .into_iter()
        .multi_cartesian_product()
        .map(Array1::from)
        .collect();

    let [mut n_a, mut n_b, mut m_a, mut m_b] = [0.0; 4];

    let curr_leaf_points_idx: Vec<usize> = leaf_points
        .index_axis(Axis(1), col)
        .indexed_iter()
        .filter(|(_, &x)| x == index)
        .map(|(i, _)| i)
        .collect();
    let curr_leaf_leaf_points = leaf_points.select(Axis(0), &curr_leaf_points_idx);

    for leaf in &leaves {
        let v = compute_leaf_values(leaf, grid_values);
        let v_pow2 = v.powi(2);
        for (idx, row) in curr_leaf_leaf_points.axis_iter(Axis(0)).enumerate() {
            if row != leaf {
                continue;
            }

            let idx = curr_leaf_points_idx[idx];
            if x[(idx, col)] < split {
                n_a += v_pow2;
                m_a += v * residuals[idx];
            } else {
                n_b += v_pow2;
                m_b += v * residuals[idx];
            }
        }
    }

    let update_a = if n_a == 0.0 { 1.0 } else { m_a / n_a + 1.0 };
    let update_b = if n_b == 0.0 { 1.0 } else { m_b / n_b + 1.0 };

    let err_old = residuals
        .select(Axis(0), &curr_leaf_points_idx)
        .pow2()
        .sum();

    let curr_points = x.select(Axis(0), &curr_leaf_points_idx);

    let (a_points_idx, b_points_idx): (Vec<_>, Vec<_>) = curr_points
        .index_axis(Axis(1), col)
        .indexed_iter()
        .map(|(i, &x)| {
            if x < split {
                (curr_leaf_points_idx[i], true)
            } else {
                (curr_leaf_points_idx[i], false)
            }
        })
        .partition(|(_, is_a)| *is_a);

    let new_resid_a = a_points_idx
        .iter()
        .map(|(i, _)| y_hat[*i].mul_add(1.0 - update_a, residuals[*i]))
        .collect::<Array1<f64>>();
    let new_resid_b = b_points_idx
        .iter()
        .map(|(i, _)| y_hat[*i].mul_add(1.0 - update_b, residuals[*i]))
        .collect::<Array1<f64>>();

    let err_new = new_resid_a.powi(2).sum() + new_resid_b.powi(2).sum();

    let refine_candidate = RefineCandidate {
        col,
        split,
        index,
        left,
        right,
        update_a,
        update_b,
        a_points_idx: a_points_idx.into_iter().map(|(i, _)| i).collect(),
        b_points_idx: b_points_idx.into_iter().map(|(i, _)| i).collect(),
        curr_leaf_points_idx,
    };

    (err_new, err_old, refine_candidate)
}

fn update_leaf_points(
    leaf_points: &mut Array2<usize>,
    dim: usize,
    index: usize,
    leaf_points_b: &[usize],
) {
    leaf_points.axis_iter_mut(Axis(0)).for_each(|mut x| {
        if x[dim] > index {
            x[dim] += 1;
        }
    });

    for &i in leaf_points_b {
        leaf_points[(i, dim)] += 1;
    }
}

fn update_predictions(
    y_hat: &mut Array1<f64>,
    residuals: &mut Array1<f64>,
    labels: ArrayView1<f64>,
    a_points_idx: &[usize],
    b_points_idx: &[usize],
    update_a: f64,
    update_b: f64,
) {
    for &i in a_points_idx {
        y_hat[i] *= update_a;
        residuals[i] = labels[i] - y_hat[i];
    }
    for &i in b_points_idx {
        y_hat[i] *= update_b;
        residuals[i] = labels[i] - y_hat[i];
    }
}

impl TreeGridFitter<'_> {
    pub fn update_tree(&mut self, refine_candidate: RefineCandidate) {
        let RefineCandidate {
            col,
            split,
            index,
            left,
            right,
            update_a,
            update_b,
            a_points_idx,
            b_points_idx,
            curr_leaf_points_idx: _,
        } = refine_candidate;

        let old_grid_value = self.grid_values[col][index];
        self.grid_values[col][index] *= update_a;
        self.grid_values[col].insert(index + 1, old_grid_value * update_b);

        self.splits[col].insert(index, split);
        self.intervals[col][index] = left;
        self.intervals[col].insert(index + 1, right);

        update_leaf_points(&mut self.leaf_points, col, index, &b_points_idx);
        update_predictions(
            &mut self.y_hat,
            &mut self.residuals,
            self.labels,
            &a_points_idx,
            &b_points_idx,
            update_a,
            update_b,
        );
    }
}

impl<'a> ModelFitter for TreeGridFitter<'a> {
    type HyperParameters = TreeGridParams;
    type Model = FittedTreeGrid;
    type Features = ArrayView2<'a, f64>;
    type Labels = ArrayView1<'a, f64>;

    fn new(x: Self::Features, y: Self::Labels) -> Self {
        let leaf_points = Array2::zeros((x.nrows(), x.ncols()));
        let splits = vec![vec![]; x.ncols()];
        let intervals = vec![vec![(f64::NEG_INFINITY, f64::INFINITY)]; x.ncols()];

        let (_, grid_values, y_hat) = compute_initial_values(x, y);
        let residuals = y.to_owned() - &y_hat;

        TreeGridFitter {
            splits,
            intervals,
            grid_values,
            leaf_points,
            labels: y,
            x,
            y_hat,
            residuals,
        }
    }

    fn fit(mut self, hyperparameters: &Self::HyperParameters) -> (FitResult, Self::Model) {
        let mut rng = rand::thread_rng();
        let n_cols = self.x.ncols();
        let n_rows = self.x.nrows();
        let n_cols_to_sample = (hyperparameters.colsample_bytree * n_cols as f64) as usize;

        let split_idx: Vec<usize> = sample(
            &mut rng,
            n_rows,
            hyperparameters.split_try * hyperparameters.n_iter,
        )
        .into_iter()
        .collect();

        let col_idx: Vec<usize> = (0..n_cols_to_sample * hyperparameters.n_iter)
            .map(|_| rng.gen_range(0..n_cols))
            .collect();

        for iter in 0..hyperparameters.n_iter {
            let mut best_candidate: Option<RefineCandidate> = None;
            let mut best_err_diff = f64::NEG_INFINITY;

            let curr_it_split_idx = &split_idx
                [iter * hyperparameters.split_try..(iter + 1) * hyperparameters.split_try];

            let curr_it_col_idx = &col_idx[iter * n_cols_to_sample..(iter + 1) * n_cols_to_sample];

            for &col in curr_it_col_idx {
                for &idx in curr_it_split_idx {
                    let split = self.x[[idx, col]];
                    let slice_candidate =
                        find_slice_candidate(&self.splits, &self.intervals, col, split);
                    let (err_new, err_old, refine_candidate) = find_refine_candidate(
                        slice_candidate,
                        self.x,
                        &self.leaf_points,
                        &self.grid_values,
                        &self.intervals,
                        self.residuals.view(),
                        self.y_hat.view(),
                    );

                    let err_diff = err_old - err_new;
                    if err_diff > best_err_diff {
                        best_candidate = Some(refine_candidate);
                        best_err_diff = err_diff;
                    }
                }
            }

            if let Some(candidate) = best_candidate {
                self.update_tree(candidate);
            }
        }

        let err = self.residuals.pow2().mean().unwrap();

        let residuals = self.residuals;
        let y_hat = self.y_hat;
        let tree_grid = FittedTreeGrid {
            splits: self.splits,
            intervals: self.intervals,
            grid_values: self.grid_values,
        };

        let fit_res = FitResult {
            err,
            residuals,
            y_hat,
        };

        (fit_res, tree_grid)
    }
}

#[cfg(test)]
mod tests {
    use ndarray::s;

    use super::*;

    macro_rules! assert_float_eq {
        ($x:expr, $y:expr, $d:expr) => {
            assert!(($x - $y).abs() < $d);
        };
    }

    fn setup_data() -> (Array2<f64>, Array1<f64>) {
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

        let y = dat.slice(s![.., 0]).to_owned();
        let x = dat.slice(s![.., 1..]).to_owned();
        (x, y)
    }

    #[test]
    fn test_tree_grid_slice_refine_candidate() {
        let (x, y) = setup_data();
        let tree_grid = TreeGridFitter::new(x.view(), y.view());
        let slice_candidate = find_slice_candidate(&tree_grid.splits, &tree_grid.intervals, 0, 1.0);
        let (_, _, refine_candidate) = find_refine_candidate(
            slice_candidate,
            x.view(),
            &tree_grid.leaf_points,
            &tree_grid.grid_values,
            &tree_grid.intervals,
            tree_grid.residuals.view(),
            tree_grid.y_hat.view(),
        );
        assert_eq!(
            refine_candidate.a_points_idx,
            vec![0, 1, 3, 4, 5, 6, 7, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        );
        assert_eq!(refine_candidate.b_points_idx, vec![2, 8, 9, 10]);
        assert_float_eq!(refine_candidate.update_a, -93.53056943616252, 1e-10);
        assert_float_eq!(refine_candidate.update_b, 379.12227774465015, 1e-10);
    }

    #[test]
    fn test_tree_grid_multiple_refines() {
        let (x, y) = setup_data();

        let find_refine_candidate_closure =
            |tree_grid: &TreeGridFitter<'_>, slice_candidate| -> (f64, f64, RefineCandidate) {
                find_refine_candidate(
                    slice_candidate,
                    x.view(),
                    &tree_grid.leaf_points,
                    &tree_grid.grid_values,
                    &tree_grid.intervals,
                    tree_grid.residuals.view(),
                    tree_grid.y_hat.view(),
                )
            };
        let mut tree_grid = TreeGridFitter::new(x.view(), y.view());
        {
            let slice_candidate =
                find_slice_candidate(&tree_grid.splits, &tree_grid.intervals, 0, x[[8, 0]]);
            let (err_new, err_old, refine_candidate) =
                find_refine_candidate_closure(&tree_grid, slice_candidate);

            assert_float_eq!(err_new, 26.61921453887834, 1e-10);
            assert_float_eq!(err_old, 39.65496224200821, 1e-10);
            assert_float_eq!(refine_candidate.update_a, -81.09657885629389, 1e-10);
            assert_float_eq!(refine_candidate.update_b, 739.869209706645, 1e-10);
            tree_grid.update_tree(refine_candidate);
        }
        {
            let slice_candidate =
                find_slice_candidate(&tree_grid.splits, &tree_grid.intervals, 0, x[[12, 0]]);
            let (err_new, err_old, refine_candidate) =
                find_refine_candidate_closure(&tree_grid, slice_candidate);

            assert_float_eq!(err_new, 22.425727245741378, 1e-10);
            assert_float_eq!(err_old, 26.153854021633833, 1e-10);
            assert_float_eq!(refine_candidate.update_a, 8.058693908258093, 1e-10);
            assert_float_eq!(refine_candidate.update_b, 0.5847827112789354, 1e-10);
            tree_grid.update_tree(refine_candidate);
        }
        {
            let slice_candidate =
                find_slice_candidate(&tree_grid.splits, &tree_grid.intervals, 0, x[[0, 0]]);
            let (err_new, err_old, refine_candidate) =
                find_refine_candidate_closure(&tree_grid, slice_candidate);
            assert_float_eq!(err_new, 14.671443409333211, 1e-10);
            assert_float_eq!(err_old, 22.425727245741378, 1e-10);
            assert_float_eq!(refine_candidate.update_a, -6.832210436114132, 1e-10);
            assert_float_eq!(refine_candidate.update_b, 3.409910903419733, 1e-10);
            tree_grid.update_tree(refine_candidate);
        }
        {
            let slice_candidate =
                find_slice_candidate(&tree_grid.splits, &tree_grid.intervals, 1, x[[15, 1]]);
            let (err_new, err_old, refine_candidate) =
                find_refine_candidate_closure(&tree_grid, slice_candidate);
            assert_float_eq!(err_new, 11.248401175931308, 1e-10);
            assert_float_eq!(err_old, 15.136803926577713, 1e-10);
            assert_float_eq!(refine_candidate.update_a, 0.8256919200067316, 1e-10);
            assert_float_eq!(refine_candidate.update_b, 1.9098338089112117, 1e-10);
            tree_grid.update_tree(refine_candidate);
        }
        {
            let slice_candidate =
                find_slice_candidate(&tree_grid.splits, &tree_grid.intervals, 1, x[[6, 1]]);
            let (err_new, err_old, refine_candidate) =
                find_refine_candidate_closure(&tree_grid, slice_candidate);
            assert_float_eq!(err_new, 3.9199063400989016, 1e-10);
            assert_float_eq!(err_old, 7.692086523143316, 1e-10);
            assert_float_eq!(refine_candidate.update_a, 1.2412453820291502, 1e-10);
            assert_float_eq!(refine_candidate.update_b, -0.11462749920417181, 1e-10);
            tree_grid.update_tree(refine_candidate);
        }
        {
            let slice_candidate =
                find_slice_candidate(&tree_grid.splits, &tree_grid.intervals, 0, x[[13, 0]]);
            let (err_new, err_old, refine_candidate) =
                find_refine_candidate_closure(&tree_grid, slice_candidate);
            assert_float_eq!(err_new, 3.5819148122645306, 1e-10);
            assert_float_eq!(err_old, 6.104930626516536, 1e-10);
            assert_float_eq!(refine_candidate.update_a, 3.691670930826113, 1e-10);
            assert_float_eq!(refine_candidate.update_b, 0.7617522326677412, 1e-10);
            tree_grid.update_tree(refine_candidate);
        }
        {
            let slice_candidate =
                find_slice_candidate(&tree_grid.splits, &tree_grid.intervals, 0, x[[14, 0]]);
            let (err_new, err_old, refine_candidate) =
                find_refine_candidate_closure(&tree_grid, slice_candidate);
            assert_float_eq!(err_new, 1.1863917023933575, 1e-10);
            assert_float_eq!(err_old, 3.5708191124576096, 1e-10);
            assert_float_eq!(refine_candidate.update_a, 0.6518311983217222, 1e-10);
            assert_float_eq!(refine_candidate.update_b, 2.828492111820738, 1e-10);
            tree_grid.update_tree(refine_candidate);
        }
    }
}
