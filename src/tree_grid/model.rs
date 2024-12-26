use super::fitter::{RefineCandidate, TreeGridFitter};
use ndarray::{Array1, Array2, Axis};
use rand::prelude::*;

#[derive(Debug)]
pub struct TreeGridParams {
    pub n_iter: usize,
    pub split_try: usize,
    pub colsample_bytree: f32,
}

#[derive(Debug)]
pub struct TreeGrid {
    is_fitted: bool,
    splits: Vec<Vec<f32>>,
    intervals: Vec<Vec<(f32, f32)>>,
    grid_values: Vec<Vec<f32>>,
    pub hyperparameters: TreeGridParams,
}

pub struct FitResult {
    pub err: f32,
    pub residuals: Array1<f32>,
    pub y_hat: Array1<f32>,
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

    pub fn predict(&self, x: &Array2<f32>) -> Array1<f32> {
        let mut y_hat = Array1::zeros(x.nrows());
        for (i, row) in x.axis_iter(Axis(0)).enumerate() {
            let mut prod = 1.0;
            for (j, &val) in row.iter().enumerate() {
                let index = self.splits[j]
                    .iter()
                    .position(|&x| x >= val)
                    .unwrap_or(self.splits[j].len());
                prod *= self.grid_values[j][index];
            }
            y_hat[i] = prod;
        }
        y_hat
    }

    pub fn fit(&mut self, x: &Array2<f32>, y: &Array1<f32>) -> FitResult {
        let mut fitter = TreeGridFitter::new(x, y);
        let mut rng = thread_rng();

        for _ in 0..self.hyperparameters.n_iter {
            // sample random columns to split on
            let n_cols_to_sample =
                (self.hyperparameters.colsample_bytree * x.ncols() as f32) as usize;

            let split_idx: Vec<usize> = (0..self.hyperparameters.split_try)
                .map(|_| rng.gen_range(0..x.nrows()))
                .collect();

            let col_idx: Vec<usize> = (0..n_cols_to_sample)
                .map(|_| rng.gen_range(0..x.ncols()))
                .collect();

            let mut best_candidate: Option<RefineCandidate> = None;
            let best_err = f32::INFINITY;
            for col in &col_idx {
                for idx in &split_idx {
                    println!("col: {}, idx: {}", col, idx);
                    let (err_new, err_old, refine_candidate) =
                        fitter.slice_and_refine_candidate(*col, *idx as f32);

                    if err_old - err_new < best_err {
                        best_candidate = Some(refine_candidate);
                    }
                }
            }

            println!("Error of best candidate: {:?}", best_err);
            if let Some(refine_candidate) = best_candidate {
                fitter.update_tree(refine_candidate);
            }
        }

        let err = fitter.residuals.mapv(|r| r * r).sum();
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
