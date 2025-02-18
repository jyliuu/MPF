use std::ops::Div;

use fitter::TreeGridFitter;
use ndarray::{Array1, ArrayView2, Axis};

use crate::FittedModel;

pub mod fitter;

#[derive(Debug, Clone)]
pub struct FittedTreeGrid {
    pub splits: Vec<Vec<f64>>,
    pub intervals: Vec<Vec<(f64, f64)>>,
    pub grid_values: Vec<Vec<f64>>,
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
        }
    }

    pub fn identify(&self) -> (f64, FittedTreeGrid) {
        let mut identified = self.clone();
        let gv = &mut identified.grid_values;
        let mut scaling = 1.0;
        for (i, grid) in gv.iter_mut().enumerate() {
            let n = grid.len();
            let norm = grid.iter().map(|x| x.powi(2).div(n as f64)).sum::<f64>();
            let sum = grid.iter().sum::<f64>();
            let factor = norm * sum.signum();
            scaling *= factor;

            grid.iter_mut().for_each(|x| *x /= factor);
        }
        (scaling, identified)
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
        y_hat
    }
}

impl<'a> From<TreeGridFitter<'a>> for FittedTreeGrid {
    fn from(fitter: TreeGridFitter<'a>) -> Self {
        Self {
            splits: fitter.splits,
            intervals: fitter.intervals,
            grid_values: fitter.grid_values,
        }
    }
}

#[cfg(test)]
mod tests {
    use csv::ReaderBuilder;
    use fitter::TreeGridParams;
    use ndarray::Array1;
    use ndarray::Array2;

    use crate::ModelFitter;

    use super::*;

    fn setup_data() -> (Array2<f64>, Array1<f64>) {
        let mut rdr = ReaderBuilder::new()
            .has_headers(true)
            .from_path("./dat.csv")
            .expect("Failed to open file");

        let mut x_data = Vec::new();
        let mut y_data = Vec::new();

        for result in rdr.records() {
            let record = result.expect("Failed to read record");
            let y: f64 = record[0].parse().expect("Failed to parse y");
            let x1: f64 = record[1].parse().expect("Failed to parse x1");
            let x2: f64 = record[2].parse().expect("Failed to parse x2");

            y_data.push(y);
            x_data.push(vec![x1, x2]);
        }

        let x = Array2::from_shape_vec((x_data.len(), 2), x_data.into_iter().flatten().collect())
            .expect("Failed to create Array2");
        let y = Array1::from(y_data);

        (x, y)
    }

    #[test]
    fn test_model_fit() {
        let (x, y) = setup_data();
        let tree_grid_fitter = TreeGridFitter::new(x.view(), y.view());

        let (fit_result, tree_grid) = tree_grid_fitter.fit(&TreeGridParams::default());

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
        let (x, y) = setup_data();
        let tree_grid_fitter = TreeGridFitter::new(x.view(), y.view());
        let (fit_result, tree_grid) = tree_grid_fitter.fit(&TreeGridParams::default());

        let y_hat = tree_grid.predict(x.view());
        let diff = &fit_result.y_hat - &y_hat;
        println!("diff: {diff:?}");

        assert!(diff.iter().all(|&x| x < 1e-6));
    }
}
