use grid_index::GridIndex;
use ndarray::{Array1, ArrayView2, Axis};

use crate::FittedModel;

mod fitter;
pub mod grid_index;
mod reproject_values;
mod splitting;

use fitter::TreeGridFitter;
pub mod identification;
pub mod params;
pub use fitter::fit;

// Re-export TreeGridParams and related types
pub use params::{TreeGridParams, TreeGridParamsBuilder};

#[derive(Debug, Clone)]
pub struct FittedTreeGrid {
    pub grid_values: Vec<Vec<f64>>,
    pub scaling: f64,
    // Add dimension_grids for faster lookups
    pub grid_index: GridIndex,
}

impl FittedTreeGrid {
    pub fn new(grid_values: Vec<Vec<f64>>, scaling: f64, grid_index: GridIndex) -> Self {
        Self {
            grid_values,
            scaling,
            grid_index,
        }
    }

    /// Optimized prediction for a single sample
    #[inline]
    pub fn predict_single(&self, x: &[f64]) -> f64 {
        debug_assert_eq!(
            x.len(),
            self.grid_index.current_dims().len(),
            "Input dimension must match tree grid dimension"
        );

        let mut product = 1.0;

        // Use dimension grid for faster lookups
        for (i, val) in x.iter().enumerate() {
            let col_idx = self.grid_index.compute_col_index_for_point(i, *val);
            product *= self.grid_values[i][col_idx];
        }

        self.scaling * product
    }
}

impl FittedModel for FittedTreeGrid {
    fn predict(&self, x: ArrayView2<f64>) -> Array1<f64> {
        let n_rows = x.nrows();
        let mut y_hat = Array1::zeros(n_rows);

        for (i, row) in x.axis_iter(Axis(0)).enumerate() {
            let row_slice = row.as_slice().unwrap();
            y_hat[i] = self.predict_single(row_slice);
        }
        y_hat
    }
}

impl<'a> From<TreeGridFitter<'a>> for FittedTreeGrid {
    fn from(fitter: TreeGridFitter<'a>) -> Self {
        Self {
            grid_values: fitter.grid_values,
            scaling: fitter.scaling,
            grid_index: fitter.grid_index,
        }
    }
}
