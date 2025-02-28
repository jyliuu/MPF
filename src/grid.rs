use ndarray::{Array1, ArrayView2, Axis};

use crate::FittedModel;

pub mod candidates;
pub mod fitter;
pub mod params;
pub mod strategies;

// Re-export TreeGridFitter for backward compatibility
pub use fitter::TreeGridFitter;
// Re-export TreeGridParams and related types
pub use params::{TreeGridParams, TreeGridParamsBuilder};

/// A more efficient representation of grid data for a single dimension
#[derive(Debug, Clone)]
pub struct DimensionGrid {
    /// Ordered sequence of split points
    pub splits: Vec<f64>,
    /// Values for each interval (length = splits.len() + 1)
    pub values: Vec<f64>,
}

impl DimensionGrid {
    /// Create a new DimensionGrid from splits and values
    pub fn new(splits: Vec<f64>, values: Vec<f64>) -> Self {
        assert_eq!(
            values.len(),
            splits.len() + 1,
            "Values length must be one more than splits length"
        );
        Self { splits, values }
    }

    /// Get the intervals represented by this grid
    pub fn get_intervals(&self) -> Vec<(f64, f64)> {
        let mut intervals = Vec::with_capacity(self.values.len());

        if self.splits.is_empty() {
            intervals.push((f64::NEG_INFINITY, f64::INFINITY));
            return intervals;
        }

        // First interval: -inf to first split
        intervals.push((f64::NEG_INFINITY, self.splits[0]));

        // Middle intervals
        for i in 0..self.splits.len() - 1 {
            intervals.push((self.splits[i], self.splits[i + 1]));
        }

        // Last interval: last split to +inf
        intervals.push((self.splits[self.splits.len() - 1], f64::INFINITY));

        intervals
    }

    /// Find the interval index containing the given value
    #[inline]
    pub fn find_interval_index(&self, value: f64) -> usize {
        // Handle edge case with no splits
        if self.splits.is_empty() {
            return 0;
        }

        // Modified to ensure proper handling of edge cases by comparing if split <= value
        // instead of default comparison which would place values exactly equal to splits
        // in a possibly inconsistent way
        match self.splits.binary_search_by(|&split| {
            if split <= value {
                std::cmp::Ordering::Less
            } else {
                std::cmp::Ordering::Greater
            }
        }) {
            Ok(index) => index + 1, // If equal to a split point, use the next interval
            Err(index) => index,    // If not found, this is the insertion point
        }
    }

    /// Get the value for a given input
    #[inline]
    pub fn get_value_for(&self, value: f64) -> f64 {
        let index = self.find_interval_index(value);
        self.values[index]
    }
}


#[derive(Debug, Clone)]
pub struct FittedTreeGrid {
    pub splits: Vec<Vec<f64>>,
    pub intervals: Vec<Vec<(f64, f64)>>,
    pub grid_values: Vec<Vec<f64>>,
    pub scaling: f64,
    // Add dimension_grids for faster lookups
    pub dimension_grids: Vec<DimensionGrid>,
}

impl FittedTreeGrid {
    pub fn new(
        splits: Vec<Vec<f64>>,
        intervals: Vec<Vec<(f64, f64)>>,
        grid_values: Vec<Vec<f64>>,
        scaling: f64,
    ) -> Self {
        // Create dimension grids for efficient access
        let dimension_grids = splits
            .iter()
            .zip(grid_values.iter())
            .map(|(split, values)| DimensionGrid::new(split.clone(), values.clone()))
            .collect();

        Self {
            splits,
            intervals,
            grid_values,
            scaling,
            dimension_grids,
        }
    }

    /// Optimized prediction for a single sample
    #[inline]
    pub fn predict_single(&self, x: &[f64]) -> f64 {
        debug_assert_eq!(
            x.len(),
            self.dimension_grids.len(),
            "Input dimension must match tree grid dimension"
        );

        let mut product = 1.0;

        // Use dimension grid for faster lookups
        for (i, grid) in self.dimension_grids.iter().enumerate() {
            product *= grid.get_value_for(x[i]);
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
        let dimension_grids = fitter
            .splits
            .iter()
            .zip(fitter.grid_values.iter())
            .map(|(split, values)| DimensionGrid::new(split.clone(), values.clone()))
            .collect();

        Self {
            splits: fitter.splits,
            intervals: fitter.intervals,
            grid_values: fitter.grid_values,
            scaling: fitter.scaling,
            dimension_grids,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{grid::params::IdentificationStrategyParams, test_data::setup_data_csv};
    use rand::{rngs::StdRng, SeedableRng};

    #[test]
    fn test_model_fit() {
        let (x, y) = setup_data_csv();
        let mut rng = StdRng::seed_from_u64(42);
        let hyperparameters = TreeGridParams::default();
        let (fit_result, _) = fitter::fit(x.view(), y.view(), &hyperparameters, &mut rng);
        let mean = y.mean().unwrap();
        let base_err = (y - mean).powi(2).mean().unwrap();
        println!("Base error: {:?}, Error: {:?}", base_err, fit_result.err);
        assert!(
            fit_result.err < base_err,
            "Error is not less than mean error"
        );
    }

    #[test]
    fn test_model_predict() {
        let (x, y) = setup_data_csv();
        let mut rng = StdRng::seed_from_u64(42);
        let hyperparameters = TreeGridParams::default();
        let (fit_result, tg) = fitter::fit(x.view(), y.view(), &hyperparameters, &mut rng);
        let pred = tg.predict(x.view());
        let diff = fit_result.y_hat - pred;
        assert!(diff.iter().all(|&x| x < 1e-6));
    }

    #[test]
    fn test_model_predict_identified_equals_unidentified() {
        let (x, y) = setup_data_csv();
        let mut rng = StdRng::seed_from_u64(42);
        let mut hyperparameters = TreeGridParams {
            identification_strategy_params: IdentificationStrategyParams::None,
            ..Default::default()
        };
        let (_, tg_unidentified) = fitter::fit(x.view(), y.view(), &hyperparameters, &mut rng);
        let pred_unidentified = tg_unidentified.predict(x.view());

        let mut rng = StdRng::seed_from_u64(42);
        hyperparameters.identification_strategy_params =
            IdentificationStrategyParams::L2_ARITH_MEAN;
        let (_, tg_identified) = fitter::fit(x.view(), y.view(), &hyperparameters, &mut rng);
        let pred_identified = tg_identified.predict(x.view());

        let diff = pred_identified - pred_unidentified;
        assert!(diff.iter().all(|&x| x.abs() < 1e-6));
    }
}
