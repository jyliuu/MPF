use super::grid::FittedTreeGrid;
mod fitter;
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

#[cfg(test)]
mod tests {
    use super::params::TreeGridFamilyBoostedParams;
    use super::*;
    use crate::test_data::setup_data_csv;
    use rand::{rngs::StdRng, SeedableRng};

    #[test]
    fn test_tgf_boosted_fit() {
        let (x, y) = setup_data_csv();
        let mut rng = StdRng::seed_from_u64(42);
        let hyperparameters = TreeGridFamilyBoostedParams::default();
        let (fit_result, _) = fitter::fit(x.view(), y.view(), &hyperparameters, &mut rng);
        let mean = y.mean().unwrap();
        let base_err = (y - mean).powi(2).mean().unwrap();
        println!("Base error: {:?}, Error: {:?}", base_err, fit_result.err);
        assert!(
            fit_result.err < base_err,
            "Error is not less than mean error"
        );
    }
}
