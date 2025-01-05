use core::f64;
use std::ops::{Deref, DerefMut};

use super::model::{FitResult, TreeGrid, TreeGridParams};
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2, ToPyArray};
use pyo3::prelude::*;

#[derive(Debug)]
#[pyclass(name = "TreeGrid")]
pub struct TreeGridPy(TreeGrid);

impl Deref for TreeGridPy {
    type Target = TreeGrid;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for TreeGridPy {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

#[derive(Debug)]
#[pyclass(name = "FitResult")]
pub struct FitResultPy(FitResult);

impl Deref for FitResultPy {
    type Target = FitResult;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for FitResultPy {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

#[pymethods]
impl FitResultPy {
    #[getter]
    pub fn get_err(&self) -> f64 {
        self.err
    }

    #[getter]
    pub fn get_residuals<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(self.residuals.to_pyarray(py))
    }

    #[getter]
    pub fn get_y_hat<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(self.y_hat.to_pyarray(py))
    }

    fn __repr__(&self) -> String {
        format!(
            "FitResult(err={}, residuals={}, y_hat={})",
            self.err, self.residuals, self.y_hat
        )
    }
}

#[pymethods]
impl TreeGridPy {
    #[new]
    fn new(n_iter: usize, split_try: usize, colsample_bytree: f64) -> Self {
        TreeGridPy(TreeGrid::new(TreeGridParams {
            n_iter,
            split_try,
            colsample_bytree,
        }))
    }

    #[getter]
    pub fn get_fitted(&self) -> PyResult<bool> {
        Ok(self.is_fitted)
    }

    #[pyo3(name = "predict")]
    pub fn _predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let x = x.as_array();
        let y_hat = self.predict(x);
        Ok(y_hat.to_pyarray(py))
    }

    #[pyo3(name = "fit")]
    pub fn _fit<'py>(
        &mut self,
        x: PyReadonlyArray2<'py, f64>,
        y: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<FitResultPy> {
        let x = x.as_array();
        let y = y.as_array();
        let fit_result = self.fit(x, y);
        Ok(FitResultPy(fit_result))
    }
}
