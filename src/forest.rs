use ndarray::{Array1, ArrayView2};

use crate::{
    family::{Aggregation, AggregationMethod, TreeGridFamily}, FittedModel,
};

mod fitter;
pub mod params;
pub use fitter::fit_boosted;
#[derive(Debug)]
pub struct MPF<T: FittedModel> {
    tree_grid_families: Vec<T>,
}

impl<T: FittedModel> MPF<T> {
    pub fn get_tree_grid_families(&self) -> &Vec<T> {
        &self.tree_grid_families
    }
}

impl<T: FittedModel> MPF<T> {
    pub const fn new(tree_grid_families: Vec<T>) -> Self {
        Self { tree_grid_families }
    }
}

impl<T> FittedModel for MPF<TreeGridFamily<T>>
where
    T: AggregationMethod,
    TreeGridFamily<T>: FittedModel,
{
    fn predict(&self, x: ArrayView2<f64>) -> Array1<f64> {
        let mut result = Array1::zeros(x.shape()[0]);
        for tree_grid_family in &self.tree_grid_families {
            result += &tree_grid_family.predict(x);
        }

        match T::AGGREGATION_METHOD {
            Aggregation::Average => result / self.tree_grid_families.len() as f64,
            Aggregation::Sum => result,
        }
    }
}
