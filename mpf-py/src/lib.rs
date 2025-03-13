use pyo3::{prelude::*, types::PyList};
use rand::{rngs::StdRng, SeedableRng};

use core::f64;
use std::ops::{Deref, DerefMut};

use numpy::{PyArray1, ToPyArray};

use mpf::{
    family::{params::CombinationStrategyParams, BoostedVariant, TreeGridFamily},
    forest::{fit_boosted, params::MPFBoostedParamsBuilder, MPF},
    grid::{
        self,
        params::{IdentificationStrategyParams, SplitStrategyParams},
        FittedTreeGrid, TreeGridParamsBuilder,
    },
    FitResult, FittedModel,
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
pub struct FitResultPy {
    #[pyo3(get)]
    err: f64,
    #[pyo3(get)]
    residuals: Py<PyArray1<f64>>,
    #[pyo3(get)]
    y_hat: Py<PyArray1<f64>>,
}

impl From<FitResult> for FitResultPy {
    fn from(fit_result: FitResult) -> Self {
        Python::with_gil(|py| FitResultPy {
            err: fit_result.err,
            residuals: fit_result.residuals.to_pyarray(py).unbind(),
            y_hat: fit_result.y_hat.to_pyarray(py).unbind(),
        })
    }
}

#[derive(Debug)]
#[pyclass(name = "TreeGridFamilyBoosted")]
pub struct TreeGridFamilyBoostedPy(TreeGridFamily<BoostedVariant>);

#[pymethods]
impl TreeGridFamilyBoostedPy {
    #[getter]
    pub fn get_tree_grids<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let tree_grids_py: Vec<TreeGridPy> = self
            .0
            .get_tree_grids()
            .iter()
            .map(|tg| TreeGridPy::from(tg.clone()))
            .collect();

        PyList::new(py, tree_grids_py)
    }

    #[getter]
    pub fn get_combined_tree_grid(&self) -> PyResult<TreeGridPy> {
        if let Some(combined_tree_grid) = self.0.get_combined_tree_grid() {
            Ok(TreeGridPy::from(combined_tree_grid.clone()))
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyAttributeError, _>(
                "No combined tree grid, rerun with `combination_strategy`!",
            ))
        }
    }

    #[getter]
    pub fn get_candidate_indices(&self) -> PyResult<Vec<usize>> {
        if let Some(candidate_indices) = self.0.get_candidate_indices() {
            Ok(candidate_indices.to_vec())
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyAttributeError, _>(
                "No candidate indices, rerun with `combination_strategy`!",
            ))
        }
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
}

#[derive(Debug)]
#[pyclass(name = "MPFBoosted")]
pub struct MPFBoostedPy(MPF<TreeGridFamily<BoostedVariant>>);

impl From<MPF<TreeGridFamily<BoostedVariant>>> for MPFBoostedPy {
    fn from(mpf: MPF<TreeGridFamily<BoostedVariant>>) -> Self {
        MPFBoostedPy(mpf)
    }
}

#[pymethods]
impl MPFBoostedPy {
    #[getter]
    pub fn get_tree_grid_families<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let tree_grid_families_py: Vec<TreeGridFamilyBoostedPy> = self
            .0
            .get_tree_grid_families()
            .iter()
            .map(|tgf: &TreeGridFamily<BoostedVariant>| TreeGridFamilyBoostedPy(tgf.clone()))
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
        n_trees: usize,
        n_iter: usize,
        split_try: usize,
        colsample_bytree: f64,
        split_strategy: u8,
        identification_strategy: u8,
        combination_strategy: u8,
        reproject_grid_values: bool,
        optimize_scaling: bool,
        similarity_threshold: f64,
        seed: u64,
    ) -> PyResult<(MPFBoostedPy, FitResultPy)> {
        let x = x.as_array();
        let y = y.as_array();

        // Use the builder pattern
        let params = MPFBoostedParamsBuilder::new()
            .epochs(epochs)
            .n_trees(n_trees)
            .n_iter(n_iter)
            .split_strategy(match split_strategy {
                1 => SplitStrategyParams::RandomSplit {
                    split_try,
                    colsample_bytree,
                },
                2 => SplitStrategyParams::IntervalRandomSplit {
                    split_try,
                    colsample_bytree,
                },
                _ => SplitStrategyParams::RandomSplit {
                    split_try,
                    colsample_bytree,
                },
            })
            .reproject_grid_values(reproject_grid_values)
            .optimize_scaling(optimize_scaling)
            .combination_strategy(match combination_strategy {
                1 => CombinationStrategyParams::ArithMean(similarity_threshold),
                2 => CombinationStrategyParams::Median(similarity_threshold),
                3 => CombinationStrategyParams::ArithmeticGeometricMean(similarity_threshold),
                4 => CombinationStrategyParams::GeometricMean(similarity_threshold),
                _ => CombinationStrategyParams::None,
            })
            .identification_strategy(match identification_strategy {
                1 => IdentificationStrategyParams::L1,
                2 => IdentificationStrategyParams::L2,
                _ => IdentificationStrategyParams::None,
            })
            .seed(seed)
            .build();

        let (fit_result, mpf) = fit_boosted(x, y, &params);
        Ok((mpf.into(), FitResultPy::from(fit_result)))
    }
}

#[pymethods]
impl TreeGridPy {
    #[getter]
    pub fn get_scaling(&self) -> f64 {
        self.scaling
    }

    #[getter]
    pub fn get_splits<'py>(&'py self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        // Convert the reference to a PyList
        PyList::new(py, &self.grid_index.boundaries)
    }

    #[getter]
    pub fn get_intervals<'py>(&'py self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        // Convert the reference to a PyList
        PyList::new(py, &self.grid_index.intervals)
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
        identification_strategy: usize,
        seed: u64,
    ) -> PyResult<(TreeGridPy, FitResultPy)> {
        let x = x.as_array();
        let y = y.as_array();
        let params = TreeGridParamsBuilder::new()
            .n_iter(n_iter)
            .split_strategy(SplitStrategyParams::RandomSplit {
                split_try,
                colsample_bytree,
            })
            .identification_strategy(match identification_strategy {
                1 => IdentificationStrategyParams::L1,
                2 => IdentificationStrategyParams::L2,
                _ => IdentificationStrategyParams::None,
            })
            .build();
        let mut rng = StdRng::seed_from_u64(seed);
        let (fit_result, tg) = grid::fit(x.view(), y.view(), &params, &mut rng);
        Ok((tg.into(), FitResultPy::from(fit_result)))
    }
}

#[pymethods]
impl FitResultPy {
    fn __repr__(&self) -> String {
        format!(
            "FitResult(error={}, residuals={}, y_hat={})",
            self.err, self.residuals, self.y_hat
        )
    }
}

#[pymodule]
fn _mpf_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<TreeGridPy>()?;
    m.add_class::<FitResultPy>()?;
    m.add_class::<MPFBoostedPy>()?;
    Ok(())
}
