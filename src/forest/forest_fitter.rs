use ndarray::{Array1, ArrayView1, ArrayView2};
use rand::{rngs::StdRng, SeedableRng};

use crate::{
    tree_grid::family::{
        averaged::{self, AveragedVariant, TreeGridFamilyAveragedParams},
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

impl Default for MPFParams {
    fn default() -> Self {
        MPFParams {
            n_families: 100,
            n_iter: 100,
            m_try: 1.0,
            split_try: 10,
        }
    }
}

pub struct MPFBaggedParams {
    pub epochs: usize,
    pub tgf_params: TreeGridFamilyBaggedParams,
}

impl Default for MPFBaggedParams {
    fn default() -> Self {
        MPFBaggedParams {
            epochs: 5,
            tgf_params: TreeGridFamilyBaggedParams::default(),
        }
    }
}

pub struct MPFAveragedParams {
    pub epochs: usize,
    pub tgf_params: TreeGridFamilyAveragedParams,
}

impl Default for MPFAveragedParams {
    fn default() -> Self {
        MPFAveragedParams {
            epochs: 5,
            tgf_params: TreeGridFamilyAveragedParams::default(),
        }
    }
}

pub fn fit_grown(
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
    hyperparameters: MPFParams,
) -> (FitResult, MPF<TreeGridFamily<GrownVariant>>) {
    let mut rng = StdRng::seed_from_u64(42);
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
            &mut rng,
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
    hyperparameters: &MPFBaggedParams,
) -> (FitResult, MPF<TreeGridFamily<BaggedVariant>>) {
    let mut rng = StdRng::seed_from_u64(42);
    let MPFBaggedParams { epochs, tgf_params } = hyperparameters;

    // Ensure that x_input and y_input have the same lifetime ('c) during this loop.
    let mut y_new = y.to_owned();
    let mut tree_grid_families = Vec::new();

    for _ in 0..*epochs {
        let (fit_result, tree_grid_family) =
            bagged::fit(x.view(), y_new.view(), tgf_params, &mut rng);
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

pub fn fit_averaged(
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
    hyperparameters: MPFAveragedParams,
) -> (FitResult, MPF<TreeGridFamily<AveragedVariant>>) {
    let mut rng = StdRng::seed_from_u64(42);
    let MPFAveragedParams { epochs, tgf_params } = hyperparameters;

    let mut tree_grid_families = Vec::new();
    let mut residuals = Array1::zeros(y.len());

    for _ in 0..epochs {
        let (fit_result, tree_grid_family) =
            averaged::fit(x.view(), y.view(), &tgf_params, &mut rng);
        tree_grid_families.push(tree_grid_family);
        residuals += &fit_result.residuals;
    }

    let residuals = residuals / epochs as f64;
    let y_hat = y.to_owned() - residuals.view();
    let err = residuals.pow2().mean().unwrap();

    let fit_result = FitResult {
        err,
        residuals,
        y_hat,
    };

    (fit_result, MPF::new(tree_grid_families))
}

#[cfg(test)]
mod tests {
    use std::ops::Div;

    use ndarray::s;

    use crate::{
        forest::forest_fitter::{
            fit_averaged, fit_bagged, fit_grown, MPFAveragedParams, MPFBaggedParams, MPFParams,
        },
        test_data::setup_data_csv,
        FittedModel,
    };

    #[test]
    fn test_mpf_fit() {
        let (x, y) = setup_data_csv();
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
        let (x, y) = setup_data_csv();
        let n = y.len();
        println!("Fitting and testing on {} samples", n / 2);
        let x_train = x.slice(s![..n / 2, ..]);
        let y_train = y.slice(s![..n / 2]);

        let x_test = x.slice(s![n / 2.., ..]);
        let y_test = y.slice(s![n / 2..]);

        let (fit_result, mpf) = fit_bagged(x_train, y_train, &MPFBaggedParams::default());
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
    fn test_mpf_averaged_fit() {
        let (x, y) = setup_data_csv();
        let n = y.len();
        println!("Fitting and testing on {} samples", n / 2);
        let x_train = x.slice(s![..n / 2, ..]);
        let y_train = y.slice(s![..n / 2]);

        let x_test = x.slice(s![n / 2.., ..]);
        let y_test = y.slice(s![n / 2..]);

        let (fit_result, mpf) = fit_averaged(x_train, y_train, MPFAveragedParams::default());
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
        let (x, y) = setup_data_csv();
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
