use std::marker::PhantomData;

use ndarray::{Array1, ArrayView1, ArrayView2};

use crate::{tree_grid::grid::fitter::TreeGridParams, FitResult, FittedModel};

use super::{bagged, Aggregation, AggregationMethod, TreeGridFamily};

pub fn fit(
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
    hyperparameters: &TreeGridFamilyAveragedParams,
) -> (FitResult, TreeGridFamily<AveragedVariant>) {
    let (fr, tgf) = bagged::fit(
        x,
        y,
        &bagged::TreeGridFamilyBaggedParams {
            B: hyperparameters.B,
            tg_params: hyperparameters.tg_params.clone(),
        },
    );

    let tgf = TreeGridFamily(tgf.0, PhantomData);
    (fr, tgf)
}

#[derive(Debug)]
pub struct AveragedVariant;

impl AggregationMethod for AveragedVariant {
    const AGGREGATION_METHOD: Aggregation = Aggregation::Average;
}

impl FittedModel for TreeGridFamily<AveragedVariant> {
    fn predict(&self, x: ArrayView2<f64>) -> Array1<f64> {
        let mut result = Array1::zeros(x.shape()[0]);

        for grids in &self.0 {
            let pred = grids.predict(x.view());
            result += &pred;
        }

        result / self.0.len() as f64
    }
}

#[derive(Debug)]
pub struct TreeGridFamilyAveragedParams {
    pub B: usize,
    pub tg_params: TreeGridParams,
}
