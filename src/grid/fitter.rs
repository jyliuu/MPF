use std::vec;

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use rand::Rng;

use crate::FitResult;

use crate::grid::candidates::{
    find_refine_candidate, find_slice_candidate, update_leaf_points, update_predictions,
    RefineCandidate,
};
use crate::grid::params::{SplitStrategyParams, TreeGridParams};
use crate::grid::strategies::{
    compute_initial_values, get_component_weights, identify_no_sign, reproject_grid_values,
    RandomSplit, SplitStrategy,
};
use crate::grid::FittedTreeGrid;

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
        } => RandomSplit {
            split_try,
            ncols_to_sample: (colsample_bytree * x.ncols() as f64) as usize,
            nrows: x.nrows(),
            ncols: x.ncols(),
        },
    };

    fitter.fit(hyperparameters, &split_strategy, rng)
}

#[derive(Debug)]
pub struct TreeGridFitter<'a> {
    pub splits: Vec<Vec<f64>>,
    pub intervals: Vec<Vec<(f64, f64)>>,
    pub grid_values: Vec<Vec<f64>>,
    pub leaf_point_counts: Vec<Vec<usize>>,
    pub leaf_points: Array2<usize>,
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

        self.leaf_point_counts[col][index] = a_points_idx.len();
        self.leaf_point_counts[col].insert(index + 1, b_points_idx.len());

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

    fn identify_no_sign(&mut self) {
        let weights = get_component_weights(&self.leaf_points, &self.grid_values);
        identify_no_sign(&mut self.grid_values, &weights, &mut self.scaling);
    }
}

impl<'a> TreeGridFitter<'a> {
    fn new(x: ArrayView2<'a, f64>, y: ArrayView1<'a, f64>) -> Self {
        let leaf_points = Array2::zeros((x.nrows(), x.ncols()));
        let leaf_point_counts = vec![vec![x.nrows()]; x.ncols()];
        let splits = vec![vec![]; x.ncols()];
        let intervals: Vec<Vec<(f64, f64)>> =
            vec![vec![(f64::NEG_INFINITY, f64::INFINITY)]; x.ncols()];

        let (_, grid_values, y_hat) = compute_initial_values(x, y);
        let residuals = y.to_owned() - &y_hat;
        TreeGridFitter {
            splits,
            intervals,
            grid_values,
            leaf_points,
            leaf_point_counts,
            labels: y,
            x,
            y_hat,
            residuals,
            scaling: 1.0,
        }
    }

    fn fit<R, S>(
        mut self,
        hyperparameters: &TreeGridParams,
        split_strategy: &S,
        rng: &mut R,
    ) -> (FitResult, FittedTreeGrid)
    where
        R: Rng + ?Sized,
        S: SplitStrategy,
    {
        let n_cols = self.x.ncols();
        let n_rows = self.x.nrows();

        // Main fitting loop
        for iter in 0..hyperparameters.n_iter {
            let (curr_it_split_idx, curr_it_col_idx) = split_strategy.sample_splits(rng);

            // Select best candidate based on strategy
            let best_candidate = {
                let mut best_candidate = None;
                let mut best_err_diff = f64::NEG_INFINITY;

                for &col in &curr_it_col_idx {
                    for &idx in &curr_it_split_idx {
                        let split = self.x[[idx, col]];
                        let slice_candidate =
                            find_slice_candidate(&self.splits, &self.intervals, col, split);
                        let refine_candidate_res = find_refine_candidate(
                            slice_candidate,
                            self.x,
                            &self.leaf_points,
                            &self.grid_values,
                            &self.intervals,
                            self.residuals.view(),
                            self.y_hat.view(),
                        );
                        match refine_candidate_res {
                            Ok((err_new, err_old, refine_candidate)) => {
                                let err_diff = err_old - err_new;
                                if err_diff > best_err_diff {
                                    best_candidate = Some(refine_candidate);
                                    best_err_diff = err_diff;
                                }
                            }
                            Err(msg) => {
                                println!("Error: {:?}", msg);
                            }
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
        if hyperparameters.reproject_grid_values {
            reproject_grid_values(
                self.x.view(),
                &self.leaf_points,
                &mut self.grid_values,
                self.labels.view(),
                self.y_hat.view(),
            );
        }
        if hyperparameters.identified {
            self.identify_no_sign();
        }

        let residuals = self.residuals;
        let y_hat = self.y_hat;

        let tree_grid =
            FittedTreeGrid::new(self.splits, self.intervals, self.grid_values, self.scaling);

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

    use super::*;
    use crate::test_data::setup_data_hardcoded;

    macro_rules! assert_float_eq {
        ($x:expr, $y:expr, $d:expr) => {
            assert!(($x - $y).abs() < $d);
        };
    }

    #[test]
    fn test_tree_grid_slice_refine_candidate() {
        let (x, y) = setup_data_hardcoded();
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
        )
        .unwrap();
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
        let (x, y) = setup_data_hardcoded();

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
                .unwrap()
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
