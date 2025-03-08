use ndarray::{ArrayView1, ArrayView2};
use rand::{Rng, SeedableRng};

use crate::{
    grid::{
        self,
        params::IdentificationStrategyParams,
        strategies::{
            combine_into_single_tree_grid, L2ArithmeticGeometricMean, L2ArithmeticMean,
            L2GeometricMean, L2Median,
        },
    },
    FitResult, FittedModel,
};

use super::{params::TreeGridFamilyBoostedParams, BoostedVariant, FittedTreeGrid, TreeGridFamily};

#[cfg(feature = "use-rayon")]
use rayon::prelude::*;

pub fn fit<R: Rng + ?Sized>(
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
    hyperparameters: &TreeGridFamilyBoostedParams,
    rng: &mut R,
) -> (FitResult, TreeGridFamily<BoostedVariant>) {
    let TreeGridFamilyBoostedParams {
        n_trees,
        bootstrap,
        tg_params,
    } = hyperparameters;
    let n = x.nrows();

    // Pre-generate seeds for each thread
    let seeds: Vec<u64> = (0..*n_trees).map(|_| rng.gen()).collect();

    #[cfg(not(feature = "use-rayon"))]
    let (fit_results, tree_grids): (Vec<FitResult>, Vec<FittedTreeGrid>) = seeds
        .iter()
        .map(|&seed| {
            let mut thread_rng = rand::rngs::StdRng::seed_from_u64(seed);
            let (fit_res, tg) = if *bootstrap {
                let sample_indices: Vec<usize> =
                    (0..n).map(|_| thread_rng.gen_range(0..n)).collect();
                let x_sample = x.select(ndarray::Axis(0), &sample_indices);
                let y_sample = y.select(ndarray::Axis(0), &sample_indices);

                grid::fit(x_sample.view(), y_sample.view(), tg_params, &mut thread_rng)
            } else {
                grid::fit(x.view(), y.view(), tg_params, &mut thread_rng)
            };
            (fit_res, tg)
        })
        .collect();

    #[cfg(feature = "use-rayon")]
    let (fit_results, tree_grids): (Vec<FitResult>, Vec<FittedTreeGrid>) = seeds
        .into_par_iter()
        .map(|seed| {
            let mut thread_rng = rand::rngs::StdRng::seed_from_u64(seed);
            let (fit_res, tg) = if *bootstrap {
                let sample_indices: Vec<usize> =
                    (0..n).map(|_| thread_rng.gen_range(0..n)).collect();
                let x_sample = x.select(ndarray::Axis(0), &sample_indices);
                let y_sample = y.select(ndarray::Axis(0), &sample_indices);

                grid::fit(x_sample.view(), y_sample.view(), tg_params, &mut thread_rng)
            } else {
                grid::fit(x.view(), y.view(), tg_params, &mut thread_rng)
            };
            (fit_res, tg)
        })
        .collect();

    let (ref_idx, reference) = tree_grids
        .iter()
        .enumerate()
        .min_by(|(i, tg), (j, tg2)| {
            fit_results[*i]
                .err
                .partial_cmp(&fit_results[*j].err)
                .unwrap()
        })
        .unwrap();

    println!("reference: {:?}", fit_results[ref_idx].err);

    let combined_tree_grid = match tg_params.identification_strategy_params {
        IdentificationStrategyParams::L2ArithMean => Some(combine_into_single_tree_grid(
            &tree_grids,
            reference,
            &L2ArithmeticMean,
            x.view(),
        )),
        IdentificationStrategyParams::L2Median => Some(combine_into_single_tree_grid(
            &tree_grids,
            reference,
            &L2Median,
            x.view(),
        )),
        IdentificationStrategyParams::L2ArithmeticGeometricMean => {
            Some(combine_into_single_tree_grid(
                &tree_grids,
                reference,
                &L2ArithmeticGeometricMean,
                x.view(),
            ))
        }
        IdentificationStrategyParams::L2GeometricMean => Some(combine_into_single_tree_grid(
            &tree_grids,
            reference,
            &L2GeometricMean,
            x.view(),
        )),
        _ => None,
    };
    let mut tgf = TreeGridFamily(tree_grids, BoostedVariant { combined_tree_grid });
    let mut preds = tgf.predict(x);

    if let Some(combined_tree_grid) = &mut tgf.1.combined_tree_grid {
        let scaling = optimal_scaling(y.view(), preds.view());
        combined_tree_grid.scaling = scaling;
        preds *= scaling;
        println!("Optimal combined scaling: {:?}", scaling);
    }
    let residuals = &y - &preds;
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

pub fn optimal_scaling(y: ArrayView1<f64>, preds: ArrayView1<f64>) -> f64 {
    let preds_ssq = preds.pow2().sum();
    let y_preds_product = y.dot(&preds);
    y_preds_product / preds_ssq
}
