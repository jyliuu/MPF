pub mod forest_fitter;

use ndarray::{Array1, ArrayView2};
use std::collections::{BTreeSet, HashMap};

use crate::tree_grid::model::FittedTreeGrid;

#[derive(Debug)]
pub struct MPF {
    tree_grids: HashMap<BTreeSet<usize>, Vec<FittedTreeGrid>>,
}

impl MPF {
    pub fn new(tree_grids: HashMap<BTreeSet<usize>, Vec<FittedTreeGrid>>) -> Self {
        Self { tree_grids }
    }

    pub fn predict(&self, x: ArrayView2<f64>) -> Array1<f64> {
        let mut result = Array1::zeros(x.shape()[0]);
        for grids in self.tree_grids.values() {
            for grid in grids {
                result += &grid.predict(x);
            }
        }
        result
    }
}
