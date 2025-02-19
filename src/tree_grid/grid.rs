use fitter::TreeGridFitter;
use ndarray::{Array1, ArrayView2, Axis};

use crate::FittedModel;

pub mod fitter;

#[derive(Debug, Clone)]
pub struct FittedTreeGrid {
    pub splits: Vec<Vec<f64>>,
    pub intervals: Vec<Vec<(f64, f64)>>,
    pub grid_values: Vec<Vec<f64>>,
    pub scaling: f64,
}

impl FittedTreeGrid {
    pub const fn new(
        splits: Vec<Vec<f64>>,
        intervals: Vec<Vec<(f64, f64)>>,
        grid_values: Vec<Vec<f64>>,
    ) -> Self {
        Self {
            splits,
            intervals,
            grid_values,
            scaling: 1.0,
        }
    }
}

impl FittedModel for FittedTreeGrid {
    fn predict(&self, x: ArrayView2<f64>) -> Array1<f64> {
        let mut y_hat = Array1::zeros(x.nrows());
        for (i, row) in x.axis_iter(Axis(0)).enumerate() {
            let mut prod = 1.0;
            for (j, &val) in row.iter().enumerate() {
                let index = self.splits[j]
                    .iter()
                    .position(|&x| val < x)
                    .unwrap_or(self.splits[j].len());
                prod *= self.grid_values[j][index];
            }
            y_hat[i] = prod;
        }
        self.scaling * y_hat
    }
}

impl<'a> From<TreeGridFitter<'a>> for FittedTreeGrid {
    fn from(fitter: TreeGridFitter<'a>) -> Self {
        Self {
            splits: fitter.splits,
            intervals: fitter.intervals,
            grid_values: fitter.grid_values,
            scaling: fitter.scaling,
        }
    }
}

#[cfg(test)]
mod tests {

    use fitter::TreeGridParams;

    use crate::test_data::setup_data_csv;

    use super::*;

    #[test]
    fn test_model_fit() {
        let (x, y) = setup_data_csv();

        let (fit_result, tree_grid) = fitter::fit(x.view(), y.view(), &TreeGridParams::default());

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
        let (fit_result, tree_grid) = fitter::fit(x.view(), y.view(), &TreeGridParams::default());

        let y_hat = tree_grid.predict(x.view());
        let diff = &fit_result.y_hat - &y_hat;
        println!("diff: {diff:?}");

        assert!(diff.iter().all(|&x| x < 1e-6));
    }
}
