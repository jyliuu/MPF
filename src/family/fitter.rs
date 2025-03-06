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
    let tree_grids: Vec<FittedTreeGrid> = seeds
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
            tg
        })
        .collect();

    #[cfg(feature = "use-rayon")]
    let tree_grids: Vec<FittedTreeGrid> = seeds
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
            tg
        })
        .collect();

    let combined_tree_grid = match tg_params.identification_strategy_params {
        IdentificationStrategyParams::L2ArithMean => Some(combine_into_single_tree_grid(
            &tree_grids,
            &L2ArithmeticMean,
            x.view(),
        )),
        IdentificationStrategyParams::L2Median => Some(combine_into_single_tree_grid(
            &tree_grids,
            &L2Median,
            x.view(),
        )),
        IdentificationStrategyParams::L2ArithmeticGeometricMean => Some(
            combine_into_single_tree_grid(&tree_grids, &L2ArithmeticGeometricMean, x.view()),
        ),
        IdentificationStrategyParams::L2GeometricMean => Some(combine_into_single_tree_grid(
            &tree_grids,
            &L2GeometricMean,
            x.view(),
        )),
        _ => None,
    };
    let tgf = TreeGridFamily(tree_grids, BoostedVariant { combined_tree_grid });
    let preds = tgf.predict(x);
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

#[cfg(test)]
mod tests {

    use rand::{rngs::StdRng, SeedableRng};

    use crate::{
        family::{fitter::fit, params::TreeGridFamilyBoostedParams},
        grid::{
            params::TreeGridParams,
            strategies::{
                combine_into_single_tree_grid, L2ArithmeticGeometricMean, L2ArithmeticMean,
                L2GeometricMean, L2Median,
            },
        },
        test_data::setup_data_csv,
        FittedModel,
    };

    #[test]
    fn test_l2_median_combined_tree_grid_predicts_well() {
        let (x, y) = setup_data_csv();
        let mut rng = StdRng::seed_from_u64(42);
        let tgf = fit(
            x.view(),
            y.view(),
            &TreeGridFamilyBoostedParams {
                n_trees: 20,
                bootstrap: false,
                tg_params: TreeGridParams::default(),
            },
            &mut rng,
        );
        let combined_tree_grid = combine_into_single_tree_grid(&tgf.1 .0, &L2Median, x.view());
        let pred = combined_tree_grid.predict(x.view());
        let err = (y - pred).powi(2).mean().unwrap();
        println!("err: {:?}", err);
        assert!(err < 0.1);
    }

    #[test]
    fn test_l2_arith_geom_mean_combined_tree_grid_predicts_well() {
        let (x, y) = setup_data_csv();
        let mut rng = StdRng::seed_from_u64(42);
        let tgf = fit(
            x.view(),
            y.view(),
            &TreeGridFamilyBoostedParams {
                n_trees: 20,
                bootstrap: false,
                tg_params: TreeGridParams::default(),
            },
            &mut rng,
        );
        let combined_tree_grid =
            combine_into_single_tree_grid(&tgf.1 .0, &L2ArithmeticGeometricMean, x.view());
        let pred = combined_tree_grid.predict(x.view());
        let err = (y - pred).powi(2).mean().unwrap();
        println!("err: {:?}", err);
        assert!(err < 0.1);
    }

    #[test]
    fn test_l2_arith_mean_combined_tree_grid_predicts_well() {
        let (x, y) = setup_data_csv();
        let mut rng = StdRng::seed_from_u64(42);
        let tgf = fit(
            x.view(),
            y.view(),
            &TreeGridFamilyBoostedParams {
                n_trees: 20,
                bootstrap: false,
                tg_params: TreeGridParams::default(),
            },
            &mut rng,
        );
        let combined_tree_grid =
            combine_into_single_tree_grid(&tgf.1 .0, &L2ArithmeticMean, x.view());
        let pred = combined_tree_grid.predict(x.view());
        let err = (y - pred).powi(2).mean().unwrap();
        println!("err: {:?}", err);
        assert!(err < 0.1);
    }

    #[test]
    fn test_l2_geom_mean_combined_tree_grid_predicts_well() {
        let (x, y) = setup_data_csv();
        let mut rng = StdRng::seed_from_u64(42);
        let tgf = fit(
            x.view(),
            y.view(),
            &TreeGridFamilyBoostedParams {
                n_trees: 20,
                bootstrap: false,
                tg_params: TreeGridParams::default(),
            },
            &mut rng,
        );
        let combined_tree_grid =
            combine_into_single_tree_grid(&tgf.1 .0, &L2GeometricMean, x.view());
        let pred = combined_tree_grid.predict(x.view());
        let err = (y - pred).powi(2).mean().unwrap();
        println!("err: {:?}", err);
        assert!(err < 0.1);
    }
}
