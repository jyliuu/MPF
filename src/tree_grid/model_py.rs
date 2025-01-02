use core::f64;

use super::{
    fitter::{RefineCandidate, TreeGridFitter},
    model::TreeGridParams,
};
use ndarray::{Array1, Axis};
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2, ToPyArray};
use pyo3::prelude::*;
use rand::prelude::*;

#[derive(Debug)]
#[pyclass(name = "RTGrid")]
pub struct TreeGridPy {
    is_fitted: bool,
    splits: Vec<Vec<f64>>,
    intervals: Vec<Vec<(f64, f64)>>,
    grid_values: Vec<Vec<f64>>,
    hyperparameters: TreeGridParams,
}

#[derive(Debug)]
#[pyclass(name = "FResult")]
pub struct FitResultPy {
    #[pyo3(get)]
    pub err: f64,
    #[pyo3(get)]
    pub residuals: Py<PyArray1<f64>>,
    #[pyo3(get)]
    pub y_hat: Py<PyArray1<f64>>,
}

#[pymethods]
impl TreeGridPy {
    #[new]
    fn new(n_iter: usize, split_try: usize, colsample_bytree: f64) -> Self {
        let splits = vec![];
        let intervals = vec![];
        let grid_values = vec![];

        TreeGridPy {
            is_fitted: false,
            splits,
            intervals,
            grid_values,
            hyperparameters: TreeGridParams {
                n_iter,
                split_try,
                colsample_bytree,
            },
        }
    }

    pub fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let x = x.as_array();
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
        Ok(y_hat.to_pyarray(py))
    }

    pub fn fit<'py>(
        &mut self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
        y: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<FitResultPy> {
        let x = x.as_array();
        let y = y.as_array();
        let mut fitter = TreeGridFitter::new(x, y);
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
        Ok(FitResultPy {
            err,
            residuals: fitter.residuals.to_pyarray(py).unbind(),
            y_hat: fitter.y_hat.to_pyarray(py).unbind(),
        })
    }
}
