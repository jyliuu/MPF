use ndarray::{ArrayView1, ArrayView2};
use rand::{rngs::StdRng, SeedableRng};

use crate::{
    family::{
        boosted::{
            self, BoostedVariant, TreeGridFamilyBoostedParams, TreeGridFamilyBoostedParamsBuilder,
        },
        TreeGridFamily,
    },
    FitResult,
};

use super::mpf::MPF;

pub struct MPFBoostedParams {
    pub epochs: usize,
    pub tgf_params: TreeGridFamilyBoostedParams,
    pub seed: u64,
}

// Builder for MPFBoostedParams
pub struct MPFBoostedParamsBuilder {
    epochs: usize,
    tgf_params_builder: TreeGridFamilyBoostedParamsBuilder,
    seed: u64,
}

impl MPFBoostedParamsBuilder {
    pub fn new() -> Self {
        Self {
            epochs: 5,
            tgf_params_builder: TreeGridFamilyBoostedParamsBuilder::new(),
            seed: 42,
        }
    }

    pub fn epochs(mut self, epochs: usize) -> Self {
        self.epochs = epochs;
        self
    }

    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    // Convenience methods for nested parameters
    pub fn B(mut self, B: usize) -> Self {
        self.tgf_params_builder = self.tgf_params_builder.B(B);
        self
    }

    pub fn n_iter(mut self, n_iter: usize) -> Self {
        self.tgf_params_builder = self.tgf_params_builder.n_iter(n_iter);
        self
    }

    pub fn split_try(mut self, split_try: usize) -> Self {
        self.tgf_params_builder = self.tgf_params_builder.split_try(split_try);
        self
    }

    pub fn colsample_bytree(mut self, colsample_bytree: f64) -> Self {
        self.tgf_params_builder = self.tgf_params_builder.colsample_bytree(colsample_bytree);
        self
    }

    pub fn identified(mut self, identified: bool) -> Self {
        self.tgf_params_builder = self.tgf_params_builder.identified(identified);
        self
    }

    pub fn build(self) -> MPFBoostedParams {
        MPFBoostedParams {
            epochs: self.epochs,
            tgf_params: self.tgf_params_builder.build(),
            seed: self.seed,
        }
    }
}

impl Default for MPFBoostedParamsBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for MPFBoostedParams {
    fn default() -> Self {
        MPFBoostedParamsBuilder::new().build()
    }
}

pub fn fit_boosted(
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
    hyperparameters: &MPFBoostedParams,
) -> (FitResult, MPF<TreeGridFamily<BoostedVariant>>) {
    let mut rng = StdRng::seed_from_u64(hyperparameters.seed);
    let MPFBoostedParams {
        epochs,
        tgf_params,
        seed: _,
    } = hyperparameters;

    // Ensure that x_input and y_input have the same lifetime ('c) during this loop.
    let mut y_new = y.to_owned();
    let mut tree_grid_families = Vec::new();

    for _ in 0..*epochs {
        let (fit_result, tree_grid_family) =
            boosted::fit(x.view(), y_new.view(), tgf_params, &mut rng);
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
        family::boosted::TreeGridFamilyBoostedParams,
        forest::forest_fitter::{fit_boosted, MPFBoostedParams, MPFBoostedParamsBuilder},
        test_data::setup_data_csv,
        FittedModel,
    };
    use ndarray::s;
    use std::ops::Div;

    #[test]
    fn test_mpf_boosted_fit() {
        let (x, y) = setup_data_csv();
        let n = y.len();
        println!("Fitting and testing on {} samples", n / 2);
        let x_train = x.slice(s![..n / 2, ..]);
        let y_train = y.slice(s![..n / 2]);

        let x_test = x.slice(s![n / 2.., ..]);
        let y_test = y.slice(s![n / 2..]);

        let params = MPFBoostedParams {
            epochs: 5,
            tgf_params: TreeGridFamilyBoostedParams::default(),
            seed: 42,
        };
        let (fit_result, mpf) = fit_boosted(x_train, y_train, &params);
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
    fn test_mpf_boosted_reproducibility() {
        let (x, y) = setup_data_csv();

        // Use builder pattern for cleaner parameter construction
        let params = MPFBoostedParamsBuilder::new()
            .epochs(2)
            .B(5)
            .n_iter(25) // Using default, but explicitly stated for clarity
            .seed(42)
            .build();

        // Train two models with the same seed
        let (_, model1) = fit_boosted(x.view(), y.view(), &params);
        let (_, model2) = fit_boosted(x.view(), y.view(), &params);

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
    fn test_mpf_boosted_different_seeds() {
        let (x, y) = setup_data_csv();

        // Use builder pattern for cleaner parameter construction
        let params1 = MPFBoostedParamsBuilder::new()
            .epochs(2)
            .B(5)
            .n_iter(25) // Using default, but explicitly stated for clarity
            .seed(42)
            .build();

        // Train models with different seeds
        let (_, model1) = fit_boosted(x.view(), y.view(), &params1);

        let params2 = MPFBoostedParamsBuilder::new()
            .epochs(2)
            .B(5)
            .n_iter(25)
            .seed(43) // Different seed
            .build();

        let (_, model2) = fit_boosted(x.view(), y.view(), &params2);

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

    #[test]
    fn test_mpf_housing() {
        use crate::test_data::setup_data_housing_csv;

        let (x, y) = setup_data_housing_csv();

        let y_mean = y.mean().unwrap();
        let error = y.view().map(|v| (v - y_mean).powi(2)).mean().unwrap();

        // Use builder pattern for cleaner parameter construction
        let params = MPFBoostedParamsBuilder::new()
            .epochs(6)
            .B(44)
            .n_iter(25) // Using default, but explicitly stated for clarity
            .build();

        let (fit_result, model) = fit_boosted(x.view(), y.view(), &params);
        let preds = model.predict(x.view());
        println!("Error: {:?}", fit_result.err);
        assert!(fit_result.err < 0.7);
    }
}
