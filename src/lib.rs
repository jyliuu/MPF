#![allow(dead_code)]
#![allow(unused_variables)]
pub mod forest;
pub mod tree_grid;
use ndarray::{Array1, ArrayView2};

#[derive(Debug)]
pub struct FitResult {
    pub err: f64,
    pub residuals: Array1<f64>,
    pub y_hat: Array1<f64>,
}

pub trait FittedModel {
    fn predict(&self, x: ArrayView2<f64>) -> Array1<f64>;
}

pub trait ModelFitter {
    type Model: FittedModel;
    type HyperParameters;
    type Features;
    type Labels;

    fn new(x: Self::Features, y: Self::Labels) -> Self;
    fn fit(self, hyperparameters: &Self::HyperParameters) -> (FitResult, Self::Model);
}
