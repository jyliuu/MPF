use super::grid::FittedTreeGrid;
pub mod averaged;
pub mod bagged;
pub mod grown;

#[derive(Debug, Clone)]
pub struct TreeGridFamily<T>(Vec<FittedTreeGrid>, T);

#[derive(PartialEq)]
pub enum Aggregation {
    Average,
    Sum,
}
pub trait AggregationMethod {
    const AGGREGATION_METHOD: Aggregation;
}

impl<T> TreeGridFamily<T> {
    pub fn get_tree_grids(&self) -> &Vec<FittedTreeGrid> {
        &self.0
    }
}

#[cfg(test)]
mod tests {
    use crate::test_data::setup_data_csv;
    use averaged::TreeGridFamilyAveragedParams;
    use bagged::TreeGridFamilyBaggedParams;
    use grown::TreeGridFamilyGrownParams;

    use crate::{tree_grid::grid::fitter::TreeGridParams, FittedModel};

    use super::*;

    #[test]
    fn test_tgf_bagged_fit() {
        let (x, y) = setup_data_csv();
        let hyperparameters = TreeGridFamilyBaggedParams {
            B: 100,
            tg_params: TreeGridParams {
                n_iter: 100,
                split_try: 10,
                colsample_bytree: 1.0,
            },
        };
        let (fit_result, _) = bagged::fit(x.view(), y.view(), &hyperparameters);
        let mean = y.mean().unwrap();
        let base_err = (y - mean).powi(2).mean().unwrap();
        println!("Base error: {:?}, Error: {:?}", base_err, fit_result.err);
        assert!(
            fit_result.err < base_err,
            "Error is not less than mean error"
        );
    }

    #[test]
    fn test_tgf_averaged_fit() {
        let (x, y) = setup_data_csv();
        let hyperparameters = TreeGridFamilyAveragedParams {
            B: 100,
            tg_params: TreeGridParams {
                n_iter: 100,
                split_try: 10,
                colsample_bytree: 1.0,
            },
        };
        let (fit_result, _) = averaged::fit(x.view(), y.view(), &hyperparameters);
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
        let (x, y) = setup_data_csv();
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
        let (x, y) = setup_data_csv();
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
