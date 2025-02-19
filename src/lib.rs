#![allow(dead_code)]
#![allow(unused_variables)]
pub mod forest;
pub mod test_data;
pub mod tree_grid;
use ndarray::{Array1, ArrayView2};
use rand::Rng;

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
    fn fit<R: Rng>(
        self,
        hyperparameters: &Self::HyperParameters,
        rng: &mut R,
    ) -> (FitResult, Self::Model);
}
