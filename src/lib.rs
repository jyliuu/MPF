pub mod forest;
pub mod tree_grid;
use ndarray::Array1;

#[derive(Debug)]
pub struct FitResult {
    pub err: f64,
    pub residuals: Array1<f64>,
    pub y_hat: Array1<f64>,
}
