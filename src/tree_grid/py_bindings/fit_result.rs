use core::f64;

use super::FitResultPy;
use numpy::{PyArray1, ToPyArray};
use pyo3::prelude::*;

#[pymethods]
impl FitResultPy {
    pub fn get_error(&self) -> f64 {
        self.err
    }

    pub fn get_residuals<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(self.residuals.to_pyarray(py))
    }

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
