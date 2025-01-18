use ndarray::{Array1, ArrayView2};
use std::collections::{BTreeSet, HashMap};

use crate::tree_grid::{model::FittedTreeGrid, tree_grid_family::TreeGridFamily};


pub struct MPF {
    tree_grid_families: Vec<TreeGridFamily>,
}

impl MPF {
    pub fn new(tree_grid_families: Vec<TreeGridFamily>) -> Self {
        Self { tree_grid_families }
    }

    pub fn predict(&self, x: ArrayView2<f64>) -> Array1<f64> {
        let mut result = Array1::zeros(x.shape()[0]);
        for tree_grid_family in self.tree_grid_families.iter() {
            result += &tree_grid_family.predict(x);
        }
        result / self.tree_grid_families.len() as f64
    }
}
