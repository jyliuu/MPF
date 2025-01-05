pub mod forest_fitter;

use ndarray::{Array1, Array2, ArrayView2};
use std::collections::{BTreeSet, HashMap};

use crate::forest::forest_fitter::ForestFitter;
use crate::tree_grid::model::{FitResult, TreeGrid};

#[derive(Debug)]
pub struct MPF {
    n_iter: usize,
    m_try: f64,
    split_try: usize,
    is_fitted: bool,
    tree_grids: HashMap<BTreeSet<usize>, Vec<TreeGrid>>,
}

impl MPF {
    pub fn new(n_iter: usize, m_try: f64, split_try: usize) -> Self {
        Self {
            n_iter,
            m_try,
            split_try,
            is_fitted: false,
            tree_grids: HashMap::new(),
        }
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

    pub fn fit(&mut self, points: Array2<f64>, labels: Array1<f64>) -> FitResult {
        let mut fitter = ForestFitter::new(points.view(), labels.view());
        let result = fitter.fit(self.n_iter, self.m_try, self.split_try);
        self.tree_grids = fitter.tree_grids;
        self.is_fitted = true;

        FitResult {
            err: result,
            residuals: fitter.residuals,
            y_hat: fitter.y_hat,
        }
    }
}
