use gridindex::GridIndex;
use ndarray::{Array1, ArrayView2, Axis};

use crate::FittedModel;

pub mod candidates;
mod fitter;
mod gridindex;

use fitter::TreeGridFitter;
pub mod params;
pub mod strategies;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        grid::params::{IdentificationStrategyParams, SplitStrategyParams},
        test_data::setup_data_csv,
    };
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
    fn test_model_fit_interval_split() {
        let (x, y) = setup_data_csv();
        let mut rng = StdRng::seed_from_u64(42);
        let hyperparameters = TreeGridParamsBuilder::new()
            .n_iter(24)
            .split_strategy(SplitStrategyParams::IntervalRandomSplit {
                split_try: 3,
                colsample_bytree: 1.0,
            })
            .build();
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
        hyperparameters.identification_strategy_params = IdentificationStrategyParams::L2ArithMean;
        let (_, tg_identified) = fitter::fit(x.view(), y.view(), &hyperparameters, &mut rng);
        let pred_identified = tg_identified.predict(x.view());

        let diff = pred_identified - pred_unidentified;
        assert!(diff.iter().all(|&x| x.abs() < 1e-6));
    }
}
