use ndarray::{Array1, ArrayView2};

use crate::{tree_grid::{tree_grid_family::TreeGridFamily, tree_grid_family_2::TreeGridFamilyBagged}, FittedModel};

pub struct MPF<T : FittedModel> {
    tree_grid_families: Vec<T>,
}

impl<T: FittedModel> MPF<T> {
    pub fn new(tree_grid_families: Vec<T>) -> Self {
        Self { tree_grid_families }
    }
}

impl FittedModel for MPF<TreeGridFamily> {
    fn predict(&self, x: ArrayView2<f64>) -> Array1<f64> {
        let mut result = Array1::zeros(x.shape()[0]);
        for tree_grid_family in self.tree_grid_families.iter() {
            result += &tree_grid_family.predict(x);
        }
        result / self.tree_grid_families.len() as f64
    }
}


impl FittedModel for MPF<TreeGridFamilyBagged> {
    fn predict(&self, x: ArrayView2<f64>) -> Array1<f64> {
        let mut result = Array1::zeros(x.shape()[0]);
        for tree_grid_family in self.tree_grid_families.iter() {
            result += &tree_grid_family.predict(x);
        }
        result / self.tree_grid_families.len() as f64
    }
}
