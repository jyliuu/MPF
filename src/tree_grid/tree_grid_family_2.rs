use ndarray::{Array1, ArrayView1, ArrayView2};
use rand::Rng;

use crate::{tree_grid::tree_grid_fitter::TreeGridFitter, FitResult};

use super::model::FittedTreeGrid;
use super::tree_grid_fitter::TreeGridParams;

#[derive(Debug)]
pub struct TreeGridFamilyBagged {
    tree_grids: Vec<FittedTreeGrid>,
}

impl TreeGridFamilyBagged {
    pub fn new(tree_grids: Vec<FittedTreeGrid>) -> Self {
        Self { tree_grids }
    }

    pub fn predict(&self, x: ArrayView2<f64>) -> Array1<f64> {
        let mut result = Array1::ones(x.shape()[0]);
        let mut signs = Array1::from_elem(x.shape()[0], 0.0);
        for grids in &self.tree_grids {
            let pred = grids.predict(x.view());

            result *= &pred;
            signs += &pred.signum();
        }

        signs = signs.signum();

        result.zip_mut_with(&signs, |v, sign| {
            *v = sign * (*v).abs().powf(1.0 / self.tree_grids.len() as f64)
        });

        result
    }

    pub fn predict2(&self, x: ArrayView2<f64>) -> Array1<f64> {
        let mut result = Array1::ones(x.shape()[0]);
        let mut signs = Array1::from_elem(x.shape()[0], 0.0);
        for grids in &self.tree_grids {
            let pred = grids.predict(x.view());

            result += &pred;
            signs += &pred.signum();
        }

        signs = signs.signum();

        result.zip_mut_with(&signs, |v, sign| {
            *v = sign * (*v).abs() / self.tree_grids.len() as f64
        });

        result
    }
}

pub struct TreeGridFamilyBaggedFitter<'a> {
    pub x: ArrayView2<'a, f64>,
    pub y: ArrayView1<'a, f64>,
}

impl<'a> TreeGridFamilyBaggedFitter<'a> {
    pub fn new(x: ArrayView2<'a, f64>, y: ArrayView1<'a, f64>) -> Self {
        Self { x, y }
    }

    pub fn fit(
        self,
        B: usize,
        n_iter: usize,
        colsample_bytree: f64,
        split_try: usize,
    ) -> (FitResult, TreeGridFamilyBagged) {
        let mut rng = rand::thread_rng();
        let mut tree_grids = vec![];

        let n = self.x.nrows();

        for b in 0..B {
            let sample_indices: Vec<usize> = (0..n).map(|_| rng.gen_range(0..n)).collect();
            let x_sample = self.x.select(ndarray::Axis(0), &sample_indices);
            let y_sample = self.y.select(ndarray::Axis(0), &sample_indices);
            let tg_fitter = TreeGridFitter::new(x_sample.view(), y_sample.view());
            let (fit_res, tg): (FitResult, FittedTreeGrid) = tg_fitter.fit(TreeGridParams {
                n_iter,
                split_try,
                colsample_bytree,
            });
            println!("b: {:?}, err: {:?}", b, fit_res.err);
            tree_grids.push(tg);
        }

        let tgf = TreeGridFamilyBagged::new(tree_grids);

        let preds = tgf.predict(self.x);
        let residuals = &self.y - &preds;
        let err = residuals.pow2().mean().unwrap();

        (
            FitResult {
                err,
                residuals: residuals.to_owned(),
                y_hat: preds,
            },
            tgf,
        )
    }
}

#[cfg(test)]
mod tests {
    use csv::ReaderBuilder;
    use ndarray::Array2;

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
    fn test_tgf_fit() {
        let (x, y) = setup_data();
        let tgf_fitter: TreeGridFamilyBaggedFitter<'_> =
            TreeGridFamilyBaggedFitter::new(x.view(), y.view());
        let (fit_result, _) = tgf_fitter.fit(100, 10, 1.0, 10);
        let mean = y.mean().unwrap();
        let base_err = (y - mean).powi(2).mean().unwrap();
        println!("Base error: {:?}, Error: {:?}", base_err, fit_result.err);
        assert!(
            fit_result.err < base_err,
            "Error is not less than mean error"
        );
    }
}
