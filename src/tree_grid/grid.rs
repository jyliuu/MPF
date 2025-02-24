use fitter::TreeGridFitter;
use ndarray::{Array1, ArrayView2, Axis};

use crate::FittedModel;

pub mod fitter;

pub fn compute_inner_product(first: &FittedTreeGrid, second: &FittedTreeGrid, dim: usize) -> f64 {
    let first_splits = &first.splits[dim];
    let first_intervals = &first.intervals[dim];
    let first_values = &first.grid_values[dim];

    let second_splits = &second.splits[dim];
    let second_intervals = &second.intervals[dim];
    let second_values = &second.grid_values[dim];

    // Combine all split points
    let mut all_splits: Vec<f64> = first_splits
        .iter()
        .chain(second_splits.iter())
        .copied()
        .collect();
    all_splits.sort_by(|a, b| a.partial_cmp(b).unwrap());
    all_splits.dedup_by(|a, b| (*a - *b).abs() < 1e-12);

    // Create sub-intervals from combined splits
    let mut sub_intervals = Vec::new();
    if all_splits.is_empty() {
        sub_intervals.push((f64::NEG_INFINITY, f64::INFINITY));
    } else {
        sub_intervals.push((f64::NEG_INFINITY, all_splits[0]));
        for i in 0..all_splits.len() - 1 {
            sub_intervals.push((all_splits[i], all_splits[i + 1]));
        }
        sub_intervals.push((all_splits[all_splits.len() - 1], f64::INFINITY));
    }

    // Compute inner product over overlapping sub-intervals
    let mut inner_product = 0.0;
    for &(start, end) in &sub_intervals {
        // Find the self interval containing this sub-interval
        let self_idx = first_intervals
            .iter()
            .position(|&(a, b)| start >= a && end <= b)
            .unwrap();
        let self_val = first_values[self_idx];

        // Find the other interval containing this sub-interval
        let other_idx = second_intervals
            .iter()
            .position(|&(a, b)| start >= a && end <= b)
            .unwrap();
        let other_val = second_values[other_idx];

        // Multiply values for this overlap (no weighting by interval length here)
        inner_product += self_val * other_val;
    }

    inner_product
}

pub fn align_treegrid_to_reference_signs(
    first: &FittedTreeGrid,
    reference: &FittedTreeGrid,
) -> Vec<f64> {
    (0..first.grid_values.len())
        .map(|dim| compute_inner_product(first, reference, dim).signum())
        .collect()
}

pub fn get_aligned_signs_for_all_tree_grids(tree_grids: &[FittedTreeGrid]) -> Vec<Vec<f64>> {
    let reference = &tree_grids[0];
    let aligned_signs: Vec<Vec<f64>> = std::iter::once(vec![1.0; reference.grid_values.len()])
        .chain(
            tree_grids
                .iter()
                .skip(1)
                .map(|grid| align_treegrid_to_reference_signs(grid, reference)),
        )
        .collect();

    aligned_signs
}

#[derive(Debug, Clone)]
pub struct FittedTreeGrid {
    pub splits: Vec<Vec<f64>>,
    pub intervals: Vec<Vec<(f64, f64)>>,
    pub grid_values: Vec<Vec<f64>>,
    pub scaling: f64,
}

impl FittedTreeGrid {
    pub const fn new(
        splits: Vec<Vec<f64>>,
        intervals: Vec<Vec<(f64, f64)>>,
        grid_values: Vec<Vec<f64>>,
    ) -> Self {
        Self {
            splits,
            intervals,
            grid_values,
            scaling: 1.0,
        }
    }
}

impl FittedModel for FittedTreeGrid {
    fn predict(&self, x: ArrayView2<f64>) -> Array1<f64> {
        let mut y_hat = Array1::zeros(x.nrows());
        for (i, row) in x.axis_iter(Axis(0)).enumerate() {
            let mut prod = 1.0;
            for (j, &val) in row.iter().enumerate() {
                let index = self.splits[j]
                    .iter()
                    .position(|&x| val < x)
                    .unwrap_or(self.splits[j].len());
                prod *= self.grid_values[j][index];
            }
            y_hat[i] = prod;
        }
        self.scaling * y_hat
    }
}

impl<'a> From<TreeGridFitter<'a>> for FittedTreeGrid {
    fn from(fitter: TreeGridFitter<'a>) -> Self {
        Self {
            splits: fitter.splits,
            intervals: fitter.intervals,
            grid_values: fitter.grid_values,
            scaling: fitter.scaling,
        }
    }
}

#[cfg(test)]
mod tests {

    use fitter::TreeGridParams;

    use crate::test_data::setup_data_csv;

    use super::*;

    #[test]
    fn test_model_fit() {
        let (x, y) = setup_data_csv();

        let (fit_result, tree_grid) = fitter::fit(x.view(), y.view(), &TreeGridParams::default());

        let mean = y.mean().unwrap();
        let base_err = (y - mean).powi(2).mean().unwrap();
        println!("Base error: {:?}, Error: {:?}", base_err, fit_result.err);
        assert!(
            fit_result.err < base_err,
            "Error is not less than mean error"
        );
    }

    #[test]
    fn test_model_seeding_works() {
        let (x, y) = setup_data_csv();

        let hyperparameters = TreeGridParams::default();
        let seed = 42;

        let (fit_result_1, _) = fitter::fit_seeded(x.view(), y.view(), &hyperparameters, seed);
        let (fit_result_2, _) = fitter::fit_seeded(x.view(), y.view(), &hyperparameters, seed);

        let diff = &fit_result_1.y_hat - &fit_result_2.y_hat;
        assert!(diff.iter().all(|&x| x < 1e-6));
    }

    #[test]
    fn test_model_predict_identified_equals_unidentified() {
        let (x, y) = setup_data_csv();

        let mut hyperparameters = TreeGridParams::default();
        let (_, fit_identified) = fitter::fit(x.view(), y.view(), &hyperparameters);

        hyperparameters.identified = false;

        let (_, fit_unidentified) = fitter::fit(x.view(), y.view(), &hyperparameters);

        println!("Identified scaling: {:?}", fit_identified.scaling);
        let y_hat_identified = fit_identified.predict(x.view());
        let y_hat_unidentified = fit_unidentified.predict(x.view());

        let diff = &y_hat_identified - &y_hat_unidentified;
        println!("diff: {diff:?}");

        assert!(diff.iter().all(|&x| x < 1e-6));
    }

    #[test]
    fn test_model_predict() {
        let (x, y) = setup_data_csv();
        let (fit_result, tree_grid) = fitter::fit(x.view(), y.view(), &TreeGridParams::default());

        let y_hat = tree_grid.predict(x.view());
        let diff = &fit_result.y_hat - &y_hat;
        println!("diff: {diff:?}");

        assert!(diff.iter().all(|&x| x < 1e-6));
    }
}
