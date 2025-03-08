mod test_data;

#[cfg(test)]
mod tests {
    use super::test_data::setup_data_csv;
    use mpf::{
        forest::{fit_boosted, params::MPFBoostedParamsBuilder},
        grid::params::{CombinationStrategyParams, SplitStrategyParams},
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

        let params = MPFBoostedParamsBuilder::new()
            .epochs(5)
            .n_trees(5)
            .n_iter(25)
            .seed(42)
            .build();
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
            .n_trees(5)
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
            .n_trees(5)
            .n_iter(25) // Using default, but explicitly stated for clarity
            .seed(42)
            .build();

        // Train models with different seeds
        let (_, model1) = fit_boosted(x.view(), y.view(), &params1);

        let params2 = MPFBoostedParamsBuilder::new()
            .epochs(2)
            .n_trees(5)
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

    fn test_fit_result_error_is_y_minus_sum_preds() {
        let (x, y) = setup_data_csv();
        let params = MPFBoostedParamsBuilder::new()
            .epochs(10)
            .n_trees(10)
            .n_iter(10)
            .seed(42)
            .build();
        let (fit_result, model) = fit_boosted(x.view(), y.view(), &params);
        let preds = model.predict(x.view());
        let err = y
            .view()
            .iter()
            .zip(preds.iter())
            .map(|(y, p)| (y - p).powi(2))
            .sum::<f64>()
            .div(y.len() as f64);

        assert!((fit_result.err - err).abs() < 1e-15);
    }

    #[test]
    fn test_mpf_housing() {
        use crate::test_data::setup_data_housing_csv;

        let (x, y) = setup_data_housing_csv();
        // Use builder pattern for cleaner parameter construction
        let params = MPFBoostedParamsBuilder::new()
            .epochs(40)
            .n_iter(120) // Using default, but explicitly stated for clarity
            .n_trees(4)
            .combination_strategy(CombinationStrategyParams::ArithmeticGeometricMean)
            .reproject_grid_values(true)
            .split_strategy(SplitStrategyParams::RandomSplit {
                split_try: 12,
                colsample_bytree: 1.0,
            })
            .build();

        let (fit_result, _) = fit_boosted(x.view(), y.view(), &params);
        println!("Error: {:?}", fit_result.err);
        assert!(fit_result.err < 0.7);
    }
}
