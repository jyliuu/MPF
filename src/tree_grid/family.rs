use super::grid::FittedTreeGrid;
pub mod bagged;

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
    use bagged::TreeGridFamilyBaggedParams;
    use rand::{rngs::StdRng, SeedableRng};

    use super::*;

    #[test]
    fn test_tgf_bagged_fit() {
        let (x, y) = setup_data_csv();
        let mut rng = StdRng::seed_from_u64(42);
        let hyperparameters = TreeGridFamilyBaggedParams::default();
        let (fit_result, _) = bagged::fit(x.view(), y.view(), &hyperparameters, &mut rng);
        let mean = y.mean().unwrap();
        let base_err = (y - mean).powi(2).mean().unwrap();
        println!("Base error: {:?}, Error: {:?}", base_err, fit_result.err);
        assert!(
            fit_result.err < base_err,
            "Error is not less than mean error"
        );
    }
}
