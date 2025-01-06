use std::ops::{Deref, DerefMut};

use super::{model::FittedTreeGrid, tree_grid_fitter::FitResult};
use pyo3::prelude::*;

mod fit_result;
mod tree_grid;

#[derive(Debug)]
#[pyclass(name = "TreeGrid")]
pub struct TreeGridPy(FittedTreeGrid);

impl From<FittedTreeGrid> for TreeGridPy {
    fn from(tg: FittedTreeGrid) -> Self {
        TreeGridPy(tg)
    }
}

impl Deref for TreeGridPy {
    type Target = FittedTreeGrid;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for TreeGridPy {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

#[derive(Debug)]
#[pyclass(name = "FitResult")]
pub struct FitResultPy(FitResult);

impl Deref for FitResultPy {
    type Target = FitResult;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
