use super::grid::FittedTreeGrid;
mod fitter;
pub mod combine_grids;
pub mod params;
pub use fitter::fit;

#[derive(Debug, Clone)]
pub struct TreeGridFamily<T>(Vec<FittedTreeGrid>, T);

#[derive(PartialEq)]
pub enum Aggregation {
    Average,
    Sum,
}
pub trait AggregationMethod {
    const AGGREGATION_METHOD: Aggregation;
}

impl<T> TreeGridFamily<T> {
    pub fn get_tree_grids(&self) -> &Vec<FittedTreeGrid> {
        &self.0
    }
}

use ndarray::{Array1, ArrayView2};

use crate::FittedModel;

#[derive(Debug, Clone)]
pub struct BoostedVariant {
    pub combined_tree_grid: Option<FittedTreeGrid>,
}

impl AggregationMethod for BoostedVariant {
    const AGGREGATION_METHOD: Aggregation = Aggregation::Sum;
}

impl TreeGridFamily<BoostedVariant> {
    pub fn get_combined_tree_grid(&self) -> Option<&FittedTreeGrid> {
        self.1.combined_tree_grid.as_ref()
    }

    fn predict_majority_voted_sign(&self, x: ArrayView2<f64>) -> Array1<f64> {
        let mut result = Array1::ones(x.shape()[0]);
        let mut signs = Array1::from_elem(x.shape()[0], 0.0);
        for grid in &self.0 {
            let pred = grid.predict(x.view());
            result *= &pred;
            signs += &pred.signum();
        }

        signs = signs.signum();

        result.zip_mut_with(&signs, |v, sign| {
            *v = sign * (*v).abs().powf(1.0 / self.0.len() as f64);
        });

        result
    }

    fn predict_arithmetic_mean(&self, x: ArrayView2<f64>) -> Array1<f64> {
        let mut result = Array1::zeros(x.shape()[0]);
        for grid in &self.0 {
            result += &grid.predict(x.view());
        }
        result /= self.0.len() as f64;
        result
    }
}

impl FittedModel for TreeGridFamily<BoostedVariant> {
    fn predict(&self, x: ArrayView2<f64>) -> Array1<f64> {
        if let Some(combined_tree_grid) = &self.1.combined_tree_grid {
            combined_tree_grid.predict(x)
        } else {
            self.predict_majority_voted_sign(x)
        }
    }
}
