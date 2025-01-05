use core::f64;

use super::{
    super::model::{TreeGrid, TreeGridParams},
    FitResultPy, TreeGridPy,
};
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2, ToPyArray};
use pyo3::prelude::*;

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
    pub fn is_fitted(&self) -> bool {
        self.is_fitted
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
