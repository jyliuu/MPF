use ndarray::{Array1, ArrayView2};

use crate::{
    tree_grid::family::{bagged::BaggedVariant, grown::GrownVariant, TreeGridFamily},
    FittedModel,
};

pub struct MPF<T: FittedModel> {
    tree_grid_families: Vec<T>,
}

impl<T: FittedModel> MPF<T> {
    pub const fn new(tree_grid_families: Vec<T>) -> Self {
        Self { tree_grid_families }
    }
}

impl FittedModel for MPF<TreeGridFamily<BaggedVariant>> {
    fn predict(&self, x: ArrayView2<f64>) -> Array1<f64> {
        let mut result = Array1::zeros(x.shape()[0]);
        for tree_grid_family in &self.tree_grid_families {
            result += &tree_grid_family.predict(x);
        }
        result
    }
}

impl FittedModel for MPF<TreeGridFamily<GrownVariant>> {
    fn predict(&self, x: ArrayView2<f64>) -> Array1<f64> {
        let mut result = Array1::zeros(x.shape()[0]);
        for tree_grid_family in &self.tree_grid_families {
            result += &tree_grid_family.predict(x);
        }
        result / self.tree_grid_families.len() as f64
    }
}
