use ndarray::{ArrayView1, ArrayView2};

use crate::{
    tree_grid::tree_grid_family::{TreeGridFamily, TreeGridFamilyFitter, TreeGridFamilyParams},
    FitResult, ModelFitter,
};

use super::mpf::MPF;

pub struct MPFFitter<'a> {
    pub x: ArrayView2<'a, f64>,
    pub y: ArrayView1<'a, f64>,
}

pub struct MPFParams {
    pub n_families: usize,
    pub n_iter: usize,
    pub m_try: f64,
    pub split_try: usize,
}

impl<'a> ModelFitter<'a> for MPFFitter<'a> {
    type Model = MPF<TreeGridFamily>;
    type HyperParameters = MPFParams;
    type Features = ArrayView2<'a, f64>;
    type Labels = ArrayView1<'a, f64>;
    fn new(x: Self::Features, y: Self::Labels) -> Self {
        Self { x, y }
    }

    fn fit(self, hyperparameters: Self::HyperParameters) -> (FitResult, MPF<TreeGridFamily>) {
        let MPFParams {
            n_families,
            n_iter,
            m_try,
            split_try,
        } = hyperparameters;
        let mut fitted_tree_grid_families = Vec::new();
        let mut fit_results = Vec::new();
        for _ in 0..n_families {
            let tg_family_fitter = TreeGridFamilyFitter::new(self.x, self.y);
            let (tgf_fit_result, tree_grid_family) = tg_family_fitter.fit(TreeGridFamilyParams {
                n_iter,
                m_try,
                split_try,
            });
            fitted_tree_grid_families.push(tree_grid_family);
            fit_results.push(tgf_fit_result);
        }

        let mut fit_result = fit_results
            .into_iter()
            .reduce(|a, b| FitResult {
                err: 0.0,
                residuals: a.residuals + b.residuals,
                y_hat: a.y_hat + b.y_hat,
            })
            .unwrap();

        fit_result.residuals /= n_families as f64;
        fit_result.y_hat /= n_families as f64;
        fit_result.err = fit_result.residuals.pow2().mean().unwrap();

        (fit_result, MPF::new(fitted_tree_grid_families))
    }
}

#[cfg(test)]
mod tests {
    use csv::ReaderBuilder;
    use ndarray::{Array1, Array2};

    use crate::FittedModel;

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
    fn test_mpf_fit() {
        let (x, y) = setup_data();
        let mpf_fitter = MPFFitter::new(x.view(), y.view());
        let (fit_result, _) = mpf_fitter.fit(MPFParams {
            n_families: 100,
            n_iter: 100,
            m_try: 1.0,
            split_try: 10,
        });

        let mean = y.mean().unwrap();
        let base_err = (y - mean).powi(2).mean().unwrap();
        println!("Base error: {:?}, Error: {:?}", base_err, fit_result.err);
        assert!(
            fit_result.err < base_err,
            "Error is not less than mean error"
        );
    }

    #[test]
    fn test_mpf_predict() {
        let (x, y) = setup_data();
        let mpf_fitter = MPFFitter::new(x.view(), y.view());
        let (fit_result, mpf) = mpf_fitter.fit(MPFParams {
            n_families: 100,
            n_iter: 100,
            m_try: 1.0,
            split_try: 10,
        });
        let pred = mpf.predict(x.view());
        let diff = fit_result.y_hat - pred;
        assert!(diff.iter().all(|&x| x < 1e-6));
    }
}
