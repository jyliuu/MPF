use pyo3::{prelude::*, types::PyList};

use core::f64;
use std::ops::{Deref, DerefMut};

use numpy::{PyArray1, ToPyArray};

use mpf::{
    forest::{
        forest_fitter::{fit_bagged, MPFBaggedParams},
        mpf::MPF,
    },
    tree_grid::{
        family::{
            bagged::{BaggedVariant, TreeGridFamilyBaggedParams},
            grown::GrownVariant,
            TreeGridFamily,
        },
        grid::{
            fitter::{TreeGridFitter, TreeGridParams},
            FittedTreeGrid,
        },
    },
    FitResult, FittedModel, ModelFitter,
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

#[derive(Debug)]
#[pyclass(name = "MPFBagged")]
pub struct MPFBaggedPy(MPF<TreeGridFamily<BaggedVariant>>);

impl From<MPF<TreeGridFamily<BaggedVariant>>> for MPFBaggedPy {
    fn from(mpf: MPF<TreeGridFamily<BaggedVariant>>) -> Self {
        MPFBaggedPy(mpf)
    }
}

#[derive(Debug)]
#[pyclass(name = "MPFGrown")]
pub struct MPFGrownPy(MPF<TreeGridFamily<GrownVariant>>);

impl From<MPF<TreeGridFamily<GrownVariant>>> for MPFGrownPy {
    fn from(mpf: MPF<TreeGridFamily<GrownVariant>>) -> Self {
        MPFGrownPy(mpf)
    }
}

#[pymethods]
impl MPFBaggedPy {
    #[getter]
    pub fn get_tree_grid_families<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let tree_grid_families_py: Vec<Vec<TreeGridPy>> = self
            .0
            .get_tree_grid_families()
            .iter()
            .map(|tgf: &TreeGridFamily<BaggedVariant>| {
                tgf.get_tree_grids()
                    .iter()
                    .map(|tg| TreeGridPy(tg.clone()))
                    .collect()
            })
            .collect();

        PyList::new(py, tree_grid_families_py)
    }

    #[pyo3(name = "predict")]
    pub fn _predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let x = x.as_array();
        let y_hat = self.0.predict(x);
        Ok(y_hat.to_pyarray(py))
    }

    #[classmethod]
    #[pyo3(name = "fit")]
    #[allow(clippy::too_many_arguments)]
    pub fn _fit<'py>(
        _cls: &Bound<'_, PyType>,
        x: PyReadonlyArray2<'py, f64>,
        y: PyReadonlyArray1<'py, f64>,
        epochs: usize,
        B: usize,
        n_iter: usize,
        split_try: usize,
        colsample_bytree: f64,
    ) -> PyResult<(MPFBaggedPy, FitResultPy)> {
        let x = x.as_array();
        let y = y.as_array();
        let params = MPFBaggedParams {
            epochs,
            tgf_params: TreeGridFamilyBaggedParams {
                B,
                tg_params: TreeGridParams {
                    n_iter,
                    split_try,
                    colsample_bytree,
                },
            },
        };
        let (fit_result, mpf) = fit_bagged(x, y, params);
        Ok((mpf.into(), FitResultPy(fit_result)))
    }
}

#[pymethods]
impl TreeGridPy {
    #[getter]
    pub fn get_splits<'py>(&'py self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        // Convert the reference to a PyList
        PyList::new(py, &self.splits)
    }

    #[getter]
    pub fn get_intervals<'py>(&'py self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        // Convert the reference to a PyList
        PyList::new(py, &self.intervals)
    }

    #[getter]
    pub fn get_grid_values<'py>(&'py self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        // Convert the reference to a PyList
        PyList::new(py, &self.grid_values)
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
        let (fit_result, tg) = tg_fitter.fit(&params);
        Ok((tg.into(), FitResultPy(fit_result)))
    }
}

#[pymethods]
impl FitResultPy {
    #[getter]
    pub fn get_error(&self) -> f64 {
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
            "FitResult(error={}, residuals={}, y_hat={})",
            self.err, self.residuals, self.y_hat
        )
    }
}

#[pymodule]
fn mpf_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<TreeGridPy>()?;
    m.add_class::<FitResultPy>()?;
    m.add_class::<MPFBaggedPy>()?;
    Ok(())
}
