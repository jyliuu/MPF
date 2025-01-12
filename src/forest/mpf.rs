use ndarray::{Array1, ArrayView2};
use std::collections::{BTreeSet, HashMap};

use crate::tree_grid::model::FittedTreeGrid;

#[derive(Debug)]
pub struct TreeGridFamily {
    tree_grids: HashMap<BTreeSet<usize>, Vec<FittedTreeGrid>>,
}

impl TreeGridFamily {
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

#[cfg(test)]
mod tests {
    use csv::ReaderBuilder;
    use ndarray::Array2;

    use crate::forest::forest_fitter::TreeGridFamilyFitter;

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
    fn test_fit() {
        let (x, y) = setup_data();
        let tgf_fitter = TreeGridFamilyFitter::new(x.view(), y.view());
        let (fit_result, _) = tgf_fitter.fit(100, 1.0, 10);
        let mean = y.mean().unwrap();
        let base_err = (y - mean).powi(2).mean().unwrap();
        println!("Base error: {:?}, Error: {:?}", base_err, fit_result.err);
        assert!(
            fit_result.err < base_err,
            "Error is not less than mean error"
        );
    }

    #[test]
    fn test_tgf_predict() {
        let (x, y) = setup_data();
        let tgf_fitter = TreeGridFamilyFitter::new(x.view(), y.view());
        let (fit_result, tgf) = tgf_fitter.fit(100, 1.0, 10);

        let pred = tgf.predict(x.view());
        let diff = fit_result.y_hat - pred;
        assert!(diff.iter().all(|&x| x < 1e-6));
    }
}
