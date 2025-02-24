use ndarray::{ArrayView1, ArrayView2};
use rand::{rngs::StdRng, SeedableRng};

use crate::{
    tree_grid::family::{
        bagged::{self, BaggedVariant, TreeGridFamilyBaggedParams},
        TreeGridFamily,
    },
    FitResult,
};

use super::mpf::MPF;

pub struct MPFBaggedParams {
    pub epochs: usize,
    pub tgf_params: TreeGridFamilyBaggedParams,
    pub seed: u64,
}

impl Default for MPFBaggedParams {
    fn default() -> Self {
        MPFBaggedParams {
            epochs: 5,
            tgf_params: TreeGridFamilyBaggedParams::default(),
            seed: 42,
        }
    }
}

pub fn fit_bagged(
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
    hyperparameters: &MPFBaggedParams,
) -> (FitResult, MPF<TreeGridFamily<BaggedVariant>>) {
    let mut rng = StdRng::seed_from_u64(hyperparameters.seed);
    let MPFBaggedParams {
        epochs,
        tgf_params,
        seed: _,
    } = hyperparameters;

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

#[cfg(test)]
mod tests {
    use crate::{
        forest::forest_fitter::{fit_bagged, MPFBaggedParams},
        test_data::setup_data_csv,
        tree_grid::{family::bagged::TreeGridFamilyBaggedParams, grid::fitter::TreeGridParams},
        FittedModel,
    };
    use ndarray::s;
    use std::ops::Div;

    #[test]
    fn test_mpf_bagged_fit() {
        let (x, y) = setup_data_csv();
        let n = y.len();
        println!("Fitting and testing on {} samples", n / 2);
        let x_train = x.slice(s![..n / 2, ..]);
        let y_train = y.slice(s![..n / 2]);

        let x_test = x.slice(s![n / 2.., ..]);
        let y_test = y.slice(s![n / 2..]);

        let params = MPFBaggedParams {
            epochs: 5,
            tgf_params: TreeGridFamilyBaggedParams::default(),
            seed: 42,
        };
        let (fit_result, mpf) = fit_bagged(x_train, y_train, &params);
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
    fn test_mpf_bagged_reproducibility() {
        let (x, y) = setup_data_csv();
        let params = MPFBaggedParams {
            epochs: 2,
            tgf_params: TreeGridFamilyBaggedParams {
                B: 5,
                tg_params: TreeGridParams {
                    n_iter: 10,
                    split_try: 5,
                    colsample_bytree: 1.0,
                    identified: true,
                },
            },
            seed: 42,
        };

        // Train two models with the same seed
        let (_, model1) = fit_bagged(x.view(), y.view(), &params);
        let (_, model2) = fit_bagged(x.view(), y.view(), &params);

        // Generate predictions
        let pred1 = model1.predict(x.view());
        let pred2 = model2.predict(x.view());

        // Check predictions are identical
        let diff = &pred1 - &pred2;
        assert!(
            diff.iter().all(|&x| x.abs() < 1e-10),
            "Models with same seed produced different predictions"
        );
    }

    #[test]
    fn test_mpf_bagged_different_seeds() {
        let (x, y) = setup_data_csv();
        let base_params = MPFBaggedParams {
            epochs: 2,
            tgf_params: TreeGridFamilyBaggedParams {
                B: 5,
                tg_params: TreeGridParams {
                    n_iter: 10,
                    split_try: 5,
                    colsample_bytree: 1.0,
                    identified: true,
                },
            },
            seed: 42,
        };

        // Train models with different seeds
        let (_, model1) = fit_bagged(x.view(), y.view(), &base_params);

        let mut params2 = base_params;
        params2.seed = 43;
        let (_, model2) = fit_bagged(x.view(), y.view(), &params2);

        // Generate predictions
        let pred1 = model1.predict(x.view());
        let pred2 = model2.predict(x.view());

        // Check predictions are different
        let diff = &pred1 - &pred2;
        assert!(
            diff.iter().any(|&x| x.abs() > 1e-10),
            "Models with different seeds produced identical predictions"
        );
    }
}
