use std::marker::PhantomData;

use ndarray::{Array1, ArrayView1, ArrayView2};
use rand::Rng;

use crate::{
    tree_grid::grid::{self, fitter::TreeGridParams},
    FitResult, FittedModel,
};

use super::{Aggregation, AggregationMethod, FittedTreeGrid, TreeGridFamily};

#[cfg(feature = "use-rayon")]
use rayon::prelude::*;

pub fn fit(
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
    hyperparameters: &TreeGridFamilyBaggedParams,
) -> (FitResult, TreeGridFamily<BaggedVariant>) {
    let TreeGridFamilyBaggedParams { B, tg_params } = hyperparameters;
    let mut tree_grids = vec![];

    let n = x.nrows();

    #[cfg(not(feature = "use-rayon"))]
    {
        let mut rng = rand::thread_rng();
        for b in 0..*B {
            let sample_indices: Vec<usize> = (0..n).map(|_| rng.gen_range(0..n)).collect();
            let x_sample = x.select(ndarray::Axis(0), &sample_indices);
            let y_sample = y.select(ndarray::Axis(0), &sample_indices);
            let (fit_res, tg): (FitResult, FittedTreeGrid) =
                grid::fitter::fit(x.view(), y.view(), tg_params);
            println!("b: {:?}, err: {:?}", b, fit_res.err);
            tree_grids.push(tg);
        }
    }
    #[cfg(feature = "use-rayon")]
    {
        println!("Using rayon");
        tree_grids = (0..*B)
            .into_par_iter()
            .map(|b| {
                let mut rng = rand::thread_rng();
                let sample_indices: Vec<usize> = (0..n).map(|_| rng.gen_range(0..n)).collect();
                let x_sample = x.select(ndarray::Axis(0), &sample_indices);
                let y_sample = y.select(ndarray::Axis(0), &sample_indices);
                let (fit_res, tg): (FitResult, FittedTreeGrid) =
                    grid::fitter::fit(x_sample.view(), y_sample.view(), tg_params);
                println!("b: {:?}, err: {:?}", b, fit_res.err);
                tg
            })
            .collect();
    }

    let tgf = TreeGridFamily(tree_grids, PhantomData);

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

#[derive(Debug)]
pub struct BaggedVariant;

impl AggregationMethod for BaggedVariant {
    const AGGREGATION_METHOD: Aggregation = Aggregation::Average;
}

impl FittedModel for TreeGridFamily<BaggedVariant> {
    fn predict(&self, x: ArrayView2<f64>) -> Array1<f64> {
        let mut result = Array1::ones(x.shape()[0]);
        let mut signs = Array1::from_elem(x.shape()[0], 0.0);
        for grids in &self.0 {
            let pred = grids.predict(x.view());

            result *= &pred;
            signs += &pred.signum();
        }

        signs = signs.signum();

        result.zip_mut_with(&signs, |v, sign| {
            *v = sign * (*v).abs().powf(1.0 / self.0.len() as f64);
        });

        result
    }
}

#[derive(Debug)]
pub struct TreeGridFamilyBaggedParams {
    pub B: usize,
    pub tg_params: TreeGridParams,
}

impl Default for TreeGridFamilyBaggedParams {
    fn default() -> Self {
        TreeGridFamilyBaggedParams {
            B: 100,
            tg_params: TreeGridParams::default(),
        }
    }
}
