use ndarray::{Array1, ArrayView2, Axis};

#[cfg(test)]
mod tests;

#[derive(Debug)]
pub struct FittedTreeGrid {
    pub splits: Vec<Vec<f64>>,
    pub intervals: Vec<Vec<(f64, f64)>>,
    pub grid_values: Vec<Vec<f64>>,
}

impl FittedTreeGrid {
    pub fn new(
        splits: Vec<Vec<f64>>,
        intervals: Vec<Vec<(f64, f64)>>,
        grid_values: Vec<Vec<f64>>,
    ) -> Self {
        FittedTreeGrid {
            splits,
            intervals,
            grid_values,
        }
    }

    pub fn predict(&self, x: ArrayView2<f64>) -> Array1<f64> {
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

    // pub fn fit<'a>(&mut self, x: ArrayView2<'a, f64>, y: ArrayView1<'a, f64>) -> FitResult {
    //     let mut fitter = TreeGridFitter::new(x, y);
    //     let result = fitter.fit(self.hyperparameters.clone());

    //     self.splits = fitter.splits;
    //     self.intervals = fitter.intervals;
    //     self.grid_values = fitter.grid_values;
    //     self.is_fitted = true;
    //     FitResult {
    //         err: result,
    //         residuals: fitter.residuals,
    //         y_hat: fitter.y_hat,
    //     }
    // }
}
