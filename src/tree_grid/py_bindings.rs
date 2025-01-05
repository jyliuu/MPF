use std::ops::{Deref, DerefMut};

use super::model::{FitResult, TreeGrid};
use pyo3::prelude::*;

mod tree_grid;
mod fit_result;

#[derive(Debug)]
#[pyclass(name = "TreeGrid")]
pub struct TreeGridPy(TreeGrid);

impl Deref for TreeGridPy {
    type Target = TreeGrid;

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
