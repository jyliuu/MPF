use core::f64;

use super::fitter::{RefineCandidate, TreeGridFitter};
use ndarray::{Array1, Array2, Axis};
use rand::prelude::*;

#[cfg(test)]
mod tests;

#[derive(Debug)]
pub struct TreeGridParams {
    pub n_iter: usize,
    pub split_try: usize,
    pub colsample_bytree: f64,
}

#[derive(Debug)]
pub struct TreeGrid {
    is_fitted: bool,
    splits: Vec<Vec<f64>>,
    intervals: Vec<Vec<(f64, f64)>>,
    grid_values: Vec<Vec<f64>>,
    pub hyperparameters: TreeGridParams,
}

#[derive(Debug)]
pub struct FitResult {
    pub err: f64,
    pub residuals: Array1<f64>,
    pub y_hat: Array1<f64>,
}

impl TreeGrid {
    pub fn new(hyperparameters: TreeGridParams) -> Self {
        let splits = vec![];
        let intervals = vec![];
        let grid_values = vec![];

        TreeGrid {
            is_fitted: false,
            splits,
            intervals,
            grid_values,
            hyperparameters,
        }
    }

    pub fn predict(&self, x: &Array2<f64>) -> Array1<f64> {
        let mut y_hat = Array1::zeros(x.nrows());
        for (i, row) in x.axis_iter(Axis(0)).enumerate() {
            let mut prod = 1.0;
            for (j, &val) in row.iter().enumerate() {
                let index = self.splits[j]
                    .iter()
                    .position(|&x| val < x)
                    .unwrap_or(self.splits[j].len());
                prod *= self.grid_values[j][index];
            }
            y_hat[i] = prod;
        }
        y_hat
    }

    pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> FitResult {
        let mut fitter = TreeGridFitter::new(x.view(), y.view());
        let mean_err = fitter.residuals.pow2().mean().unwrap();
        let mut rng = thread_rng();

        for _ in 0..self.hyperparameters.n_iter {
            // sample random columns to split on
            let n_cols_to_sample =
                (self.hyperparameters.colsample_bytree * x.ncols() as f64) as usize;

            let split_idx: Vec<usize> = (0..self.hyperparameters.split_try)
                .map(|_| rng.gen_range(0..x.nrows()))
                .collect();

            let mut possible_indices: Vec<usize> = (0..x.ncols()).collect();
            possible_indices.shuffle(&mut rng);

            let col_idx = possible_indices[0..n_cols_to_sample].to_vec();

            let mut best_candidate: Option<RefineCandidate> = None;
            let mut best_err_diff = f64::NEG_INFINITY;
            for col in &col_idx {
                for idx in &split_idx {
                    let split = x[[*idx, *col]];
                    let (err_new, err_old, refine_candidate) =
                        fitter.slice_and_refine_candidate(*col, split);

                    let err_diff = err_old - err_new;
                    if err_diff > best_err_diff {
                        best_candidate = Some(refine_candidate);
                        best_err_diff = err_diff;
                    }
                }
            }

            if let Some(update_candidate) = best_candidate {
                fitter.update_tree(update_candidate);
            }
        }

        let err = fitter.residuals.pow2().mean().unwrap();
        self.splits = fitter.splits;
        self.intervals = fitter.intervals;
        self.grid_values = fitter.grid_values;
        self.is_fitted = true;
        FitResult {
            err,
            residuals: fitter.residuals,
            y_hat: fitter.y_hat,
        }
    }
}
