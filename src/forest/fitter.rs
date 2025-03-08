use ndarray::{ArrayView1, ArrayView2};
use rand::{rngs::StdRng, SeedableRng};

use super::{params::MPFBoostedParams, MPF};
use crate::family::{fit, BoostedVariant, TreeGridFamily};
use crate::FitResult;

pub fn fit_boosted(
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
    hyperparameters: &MPFBoostedParams,
) -> (FitResult, MPF<TreeGridFamily<BoostedVariant>>) {
    println!(
        "Fitting boosted model with hyperparameters: {:?}",
        hyperparameters
    );
    let mut rng = StdRng::seed_from_u64(hyperparameters.seed);
    let MPFBoostedParams {
        epochs,
        tgf_params,
        seed: _,
    } = hyperparameters;

    // Ensure that x_input and y_input have the same lifetime ('c) during this loop.
    let mut y_new = y.to_owned();
    let mut tree_grid_families = Vec::new();

    for i in 0..*epochs {
        let (fit_result, tree_grid_family) = fit(x.view(), y_new.view(), tgf_params, &mut rng);
        tree_grid_families.push(tree_grid_family);
        println!("Epoch {}, error: {:?}", i, fit_result.err);
        y_new = fit_result.residuals;
    }

    let fit_result = FitResult {
        err: y_new.pow2().mean().unwrap(),
        residuals: y_new.clone(),
        y_hat: -y_new + y,
    };

    (fit_result, MPF::new(tree_grid_families))
}
