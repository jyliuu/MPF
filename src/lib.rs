pub mod forest;
pub mod tree_grid;
use ndarray::Array1;
use pyo3::prelude::*;

#[pymodule]
fn mpf(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<tree_grid::py_bindings::TreeGridPy>()?;
    m.add_class::<tree_grid::py_bindings::FitResultPy>()?;
    Ok(())
}

#[derive(Debug)]
pub struct FitResult {
    pub err: f64,
    pub residuals: Array1<f64>,
    pub y_hat: Array1<f64>,
}
