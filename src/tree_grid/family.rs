use ndarray::Array1;

use crate::{FitResult, FittedModel, ModelFitter};

use super::grid::FittedTreeGrid;
pub mod bagged;
pub mod grown;

pub trait FitAndPredictStrategy {
    type HyperParameters;
    type Model: FittedModel;
    type Features;
    type Labels;

    fn fit(
        x: Self::Features,
        y: Self::Labels,
        hyperparameters: &Self::HyperParameters,
    ) -> (FitResult, Self::Model);
    fn predict(model: Self::Model, x: Self::Features) -> Array1<f64>;
}
pub struct TreeGridFamilyFitter<T: FitAndPredictStrategy> {
    variant: T,
    x: T::Features,
    y: T::Labels,
}

#[derive(Debug)]
pub struct TreeGridFamily<T>(T, Vec<FittedTreeGrid>);

impl<T> TreeGridFamily<T> {
    pub fn get_tree_grids(&self) -> &Vec<FittedTreeGrid> {
        &self.1
    }
}

impl<'a, T> ModelFitter for TreeGridFamilyFitter<T>
where
    TreeGridFamily<T>: FittedModel,
    T: FitAndPredictStrategy + Default,
{
    type Features = T::Features;
    type Labels = T::Labels;
    type Model = T::Model;
    type HyperParameters = T::HyperParameters;

    fn new(x: Self::Features, y: Self::Labels) -> Self {
        Self {
            variant: T::default(),
            x,
            y,
        }
    }

    fn fit(self, hyperparameters: &Self::HyperParameters) -> (FitResult, Self::Model) {
        T::fit(self.x, self.y, hyperparameters)
    }
}

#[cfg(test)]
mod tests {
    use bagged::{TreeGridFamilyBaggedFitter, TreeGridFamilyBaggedParams};
    use csv::ReaderBuilder;
    use grown::TreeGridFamilyGrownParams;
    use ndarray::Array2;

    use crate::tree_grid::grid::fitter::TreeGridParams;

    use super::*;
    fn setup_data() -> (Array2<f64>, Array1<f64>) {
        let mut rdr = ReaderBuilder::new()
            .has_headers(true)
            .from_path("./dat.csv")
            .expect("Failed to open file");

        let mut x_data = Vec::new();
        let mut y_data = Vec::new();

        for result in rdr.records() {
            let record = result.expect("Failed to read record");
            let y: f64 = record[0].parse().expect("Failed to parse y");
            let x1: f64 = record[1].parse().expect("Failed to parse x1");
            let x2: f64 = record[2].parse().expect("Failed to parse x2");

            y_data.push(y);
            x_data.push(vec![x1, x2]);
        }

        let x = Array2::from_shape_vec((x_data.len(), 2), x_data.into_iter().flatten().collect())
            .expect("Failed to create Array2");
        let y = Array1::from(y_data);

        (x, y)
    }

    #[test]
    fn test_tgf_bagged_fit() {
        let (x, y) = setup_data();
        let tgf_fitter = TreeGridFamilyBaggedFitter::new(x.view(), y.view());
        let hyperparameters = TreeGridFamilyBaggedParams {
            B: 100,
            tg_params: TreeGridParams {
                n_iter: 100,
                split_try: 10,
                colsample_bytree: 1.0,
            },
        };
        let (fit_result, _) = tgf_fitter.fit(&hyperparameters);
        let mean = y.mean().unwrap();
        let base_err = (y - mean).powi(2).mean().unwrap();
        println!("Base error: {:?}, Error: {:?}", base_err, fit_result.err);
        assert!(
            fit_result.err < base_err,
            "Error is not less than mean error"
        );
    }

    #[test]
    fn test_tgf_grown_fit() {
        let (x, y) = setup_data();
        let hyperparameters = TreeGridFamilyGrownParams {
            n_iter: 100,
            m_try: 1.0,
            split_try: 10,
        };
        let (fit_result, _) = grown::fit(x.view(), y.view(), &hyperparameters);
        let mean = y.mean().unwrap();
        let base_err = (y - mean).powi(2).mean().unwrap();
        println!("Base error: {:?}, Error: {:?}", base_err, fit_result.err);
        assert!(
            fit_result.err < base_err,
            "Error is not less than mean error"
        );
    }

    #[test]
    fn test_tgf_grown_predict() {
        let (x, y) = setup_data();
        let hyperparameters = TreeGridFamilyGrownParams {
            n_iter: 100,
            m_try: 1.0,
            split_try: 10,
        };
        let (fit_result, tgf) = grown::fit(x.view(), y.view(), &hyperparameters);

        let pred = tgf.predict(x.view());
        let diff = fit_result.y_hat - pred;
        assert!(diff.iter().all(|&x| x < 1e-6));
    }
}
