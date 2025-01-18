use pyo3::prelude::*;

use core::f64;
use std::ops::{Deref, DerefMut};

use numpy::{PyArray1, ToPyArray};

use mpf::{
    tree_grid::{
        model::FittedTreeGrid,
        tree_grid_fitter::{TreeGridFitter, TreeGridParams},
    },
    FitResult,
};

use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::types::PyType;

#[derive(Debug)]
#[pyclass(name = "TreeGrid")]
pub struct TreeGridPy(FittedTreeGrid);

impl From<FittedTreeGrid> for TreeGridPy {
    fn from(tg: FittedTreeGrid) -> Self {
        TreeGridPy(tg)
    }
}

impl Deref for TreeGridPy {
    type Target = FittedTreeGrid;

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

#[pymodule]
fn mpf_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<TreeGridPy>()?;
    m.add_class::<FitResultPy>()?;
    Ok(())
}
