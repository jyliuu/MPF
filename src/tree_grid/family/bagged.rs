use ndarray::{Array1, ArrayView1, ArrayView2};
use rand::Rng;

use crate::{
    tree_grid::grid::fitter::{TreeGridFitter, TreeGridParams},
    FitResult, FittedModel, ModelFitter,
};

use super::{FitAndPredictStrategy, FittedTreeGrid, TreeGridFamily};

pub struct BaggedVariant;

impl Default for BaggedVariant {
    fn default() -> Self {
        BaggedVariant
    }
}

impl FittedModel for TreeGridFamily<BaggedVariant> {
    fn predict(&self, x: ArrayView2<f64>) -> Array1<f64> {
        let mut result = Array1::ones(x.shape()[0]);
        let mut signs = Array1::from_elem(x.shape()[0], 0.0);
        for grids in &self.1 {
            let pred = grids.predict(x.view());

            result *= &pred;
            signs += &pred.signum();
        }

        signs = signs.signum();

        result.zip_mut_with(&signs, |v, sign| {
            *v = sign * (*v).abs().powf(1.0 / self.1.len() as f64);
        });

        result
    }
}

#[derive(Debug)]
pub struct TreeGridFamilyBaggedParams {
    pub B: usize,
    pub tg_params: TreeGridParams,
}

impl<'a> FitAndPredictStrategy<'a> for BaggedVariant {
    type HyperParameters = TreeGridFamilyBaggedParams;
    type Model = TreeGridFamily<BaggedVariant>;
    type Features = ArrayView2<'a, f64>;
    type Labels = ArrayView1<'a, f64>;

    fn fit(
        x: Self::Features,
        y: Self::Labels,
        hyperparameters: Self::HyperParameters,
    ) -> (FitResult, Self::Model) {
        TreeGridFamilyBaggedFitter::new(x, y).fit(hyperparameters)
    }
    fn predict(model: Self::Model, x: Self::Features) -> Array1<f64> {
        model.predict(x)
    }
}

pub struct TreeGridFamilyBaggedFitter<'a> {
    pub x: ArrayView2<'a, f64>,
    pub y: ArrayView1<'a, f64>,
}

impl<'a> ModelFitter<'a> for TreeGridFamilyBaggedFitter<'a> {
    type Features = ArrayView2<'a, f64>;
    type Labels = ArrayView1<'a, f64>;
    type Model = TreeGridFamily<BaggedVariant>;
    type HyperParameters = TreeGridFamilyBaggedParams;

    fn new(x: Self::Features, y: Self::Labels) -> Self {
        Self { x, y }
    }

    fn fit(
        self,
        hyperparameters: Self::HyperParameters,
    ) -> (FitResult, TreeGridFamily<BaggedVariant>) {
        let TreeGridFamilyBaggedParams { B, tg_params } = hyperparameters;
        let TreeGridParams {
            n_iter,
            split_try,
            colsample_bytree,
        } = tg_params;
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

        let tgf = TreeGridFamily(BaggedVariant, tree_grids);

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
