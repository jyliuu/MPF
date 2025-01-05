use core::f64;

use super::tree_grid_fitter::TreeGridFitter;
use ndarray::{Array1, ArrayView1, ArrayView2, Axis};

#[cfg(test)]
mod tests;

#[derive(Debug, Clone)]
pub struct TreeGridParams {
    pub n_iter: usize,
    pub split_try: usize,
    pub colsample_bytree: f64,
}

#[derive(Debug)]
pub struct TreeGrid {
    pub is_fitted: bool,
    pub splits: Vec<Vec<f64>>,
    pub intervals: Vec<Vec<(f64, f64)>>,
    pub grid_values: Vec<Vec<f64>>,
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

    pub fn predict(&self, x: ArrayView2<f64>) -> Array1<f64> {
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

    pub fn fit<'a>(&mut self, x: ArrayView2<'a, f64>, y: ArrayView1<'a, f64>) -> FitResult {
        let mut fitter = TreeGridFitter::new(x, y);
        let result = fitter.fit(self.hyperparameters.clone());

        self.splits = fitter.splits;
        self.intervals = fitter.intervals;
        self.grid_values = fitter.grid_values;
        self.is_fitted = true;
        FitResult {
            err: result,
            residuals: fitter.residuals,
            y_hat: fitter.y_hat,
        }
    }
}
