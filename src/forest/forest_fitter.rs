use ndarray::{ArrayView1, ArrayView2};

use crate::{
    tree_grid::family::{
        bagged::{self, BaggedVariant, TreeGridFamilyBaggedParams},
        grown::{self, GrownVariant, TreeGridFamilyGrownParams},
        TreeGridFamily,
    },
    FitResult,
};

use super::mpf::MPF;

pub struct MPFParams {
    pub n_families: usize,
    pub n_iter: usize,
    pub m_try: f64,
    pub split_try: usize,
}

pub struct MPFBaggedParams {
    pub epochs: usize,
    pub tgf_params: TreeGridFamilyBaggedParams,
}

pub fn fit_grown(
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
    hyperparameters: MPFParams,
) -> (FitResult, MPF<TreeGridFamily<GrownVariant>>) {
    let MPFParams {
        n_families,
        n_iter,
        m_try,
        split_try,
    } = hyperparameters;
    let mut fitted_tree_grid_families = Vec::new();
    let mut fit_results = Vec::new();
    for _ in 0..n_families {
        let (tgf_fit_result, tree_grid_family) = grown::fit(
            x.view(),
            y.view(),
            &TreeGridFamilyGrownParams {
                n_iter,
                m_try,
                split_try,
            },
        );
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

pub fn fit_bagged(
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
    hyperparameters: MPFBaggedParams,
) -> (FitResult, MPF<TreeGridFamily<BaggedVariant>>) {
    let MPFBaggedParams { epochs, tgf_params } = hyperparameters;

    // Ensure that x_input and y_input have the same lifetime ('c) during this loop.
    let i = 0;
    let mut y_new = y.to_owned();
    let mut tree_grid_families = Vec::new();

    for _ in 0..epochs {
        let (fit_result, tree_grid_family) = bagged::fit(x.view(), y.view(), &tgf_params);
        tree_grid_families.push(tree_grid_family);
        y_new = fit_result.residuals;
    }

    let fit_result = FitResult {
        err: y_new.pow2().mean().unwrap(),
        residuals: y_new.clone(),
        y_hat: -y_new + y,
    };

    (fit_result, MPF::new(tree_grid_families))
}

#[cfg(test)]
mod tests {
    use std::ops::Div;

    use csv::ReaderBuilder;
    use ndarray::{s, Array1, Array2};

    use crate::{tree_grid::grid::fitter::TreeGridParams, FittedModel};

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
        let n = y.len();
        println!("Fitting and testing on {} samples", n / 2);
        let x_train = x.slice(s![..n / 2, ..]);
        let y_train = y.slice(s![..n / 2]);

        let x_test = x.slice(s![n / 2.., ..]);
        let y_test = y.slice(s![n / 2..]);

        let (fit_result, mpf) = fit_grown(
            x_train,
            y_train,
            MPFParams {
                n_families: 100,
                n_iter: 100,
                m_try: 1.0,
                split_try: 10,
            },
        );

        let mean = y_test.mean().unwrap();
        let base_err = y_test.view().map(|v| v - mean).powi(2).mean().unwrap();
        let preds = mpf.predict(x_test.view());
        let test_err: f64 = y_test
            .indexed_iter()
            .map(|(i, v)| (v - preds[i]).powi(2).div(y_test.len() as f64))
            .sum();
        println! {"Base error: {:?}, Training Error: {:?}, Test Error: {:?}", base_err, fit_result.err, test_err};

        assert!(
            fit_result.err < base_err,
            "Error is not less than mean error"
        );
    }

    #[test]
    fn test_mpf_bagged_fit() {
        let (x, y) = setup_data();
        let n = y.len();
        println!("Fitting and testing on {} samples", n / 2);
        let x_train = x.slice(s![..n / 2, ..]);
        let y_train = y.slice(s![..n / 2]);

        let x_test = x.slice(s![n / 2.., ..]);
        let y_test = y.slice(s![n / 2..]);

        let (fit_result, mpf) = fit_bagged(
            x_train,
            y_train,
            MPFBaggedParams {
                epochs: 5,
                tgf_params: TreeGridFamilyBaggedParams {
                    B: 100,
                    tg_params: TreeGridParams {
                        n_iter: 100,
                        split_try: 10,
                        colsample_bytree: 1.0,
                    },
                },
            },
        );
        let mean = y_test.mean().unwrap();
        let base_err = y_test.view().map(|v| v - mean).powi(2).mean().unwrap();
        let preds = mpf.predict(x_test.view());
        let test_err: f64 = y_test
            .indexed_iter()
            .map(|(i, v)| (v - preds[i]).powi(2).div(y_test.len() as f64))
            .sum();
        println! {"Base error: {:?}, Training Error: {:?}, Test Error: {:?}", base_err, fit_result.err, test_err};

        assert!(
            fit_result.err < base_err,
            "Error is not less than mean error"
        );
    }

    #[test]
    fn test_mpf_predict() {
        let (x, y) = setup_data();
        let (fit_result, mpf) = fit_grown(
            x.view(),
            y.view(),
            MPFParams {
                n_families: 100,
                n_iter: 100,
                m_try: 1.0,
                split_try: 10,
            },
        );
        let pred = mpf.predict(x.view());
        let diff = fit_result.y_hat - pred;
        assert!(diff.iter().all(|&x| x < 1e-6));
    }
}
