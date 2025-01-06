use core::f64;

use crate::tree_grid::tree_grid_fitter::{TreeGridFitter, TreeGridParams};

use super::{FitResultPy, TreeGridPy};
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2, ToPyArray};
use pyo3::{prelude::*, types::PyType};

#[pymethods]
impl TreeGridPy {
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

    #[classmethod]
    #[pyo3(name = "fit")]
    pub fn _fit<'py>(
        _cls: &Bound<'_, PyType>,
        x: PyReadonlyArray2<'py, f64>,
        y: PyReadonlyArray1<'py, f64>,
        n_iter: usize,
        split_try: usize,
        colsample_bytree: f64,
    ) -> PyResult<(TreeGridPy, FitResultPy)> {
        let x = x.as_array();
        let y = y.as_array();
        let params = TreeGridParams {
            n_iter,
            split_try,
            colsample_bytree,
        };
        let tg_fitter = TreeGridFitter::new(x.view(), y.view());
        let (fit_result, tg) = tg_fitter.fit(params);
        Ok((tg.into(), FitResultPy(fit_result)))
    }
}
