use ndarray::{Array1, ArrayView1, ArrayView2};
use rand::{Rng, SeedableRng};

use crate::{
    tree_grid::grid::{
        self, compute_inner_product, get_aligned_signs_for_all_tree_grids,
        params::{TreeGridParams, TreeGridParamsBuilder},
    },
    FitResult, FittedModel,
};

use super::{Aggregation, AggregationMethod, FittedTreeGrid, TreeGridFamily};

#[cfg(feature = "use-rayon")]
use rayon::prelude::*;

pub fn fit<R: Rng + ?Sized>(
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
    hyperparameters: &TreeGridFamilyBoostedParams,
    rng: &mut R,
) -> (FitResult, TreeGridFamily<BoostedVariant>) {
    let TreeGridFamilyBoostedParams { B, tg_params } = hyperparameters;
    let n = x.nrows();

    // Pre-generate seeds for each thread
    let seeds: Vec<u64> = (0..*B).map(|_| rng.gen()).collect();

    #[cfg(not(feature = "use-rayon"))]
    let tree_grids: Vec<FittedTreeGrid> = seeds
        .iter()
        .map(|&seed| {
            let mut thread_rng = rand::rngs::StdRng::seed_from_u64(seed);
            let sample_indices: Vec<usize> = (0..n).map(|_| thread_rng.gen_range(0..n)).collect();
            let x_sample = x.select(ndarray::Axis(0), &sample_indices);
            let y_sample = y.select(ndarray::Axis(0), &sample_indices);
            let (fit_res, tg) =
                grid::fitter::fit(x_sample.view(), y_sample.view(), tg_params, &mut thread_rng);
            println!("err: {:?}", fit_res.err);
            tg
        })
        .collect();

    #[cfg(feature = "use-rayon")]
    let tree_grids: Vec<FittedTreeGrid> = seeds
        .into_par_iter()
        .map(|seed| {
            let mut thread_rng = rand::rngs::StdRng::seed_from_u64(seed);
            let sample_indices: Vec<usize> = (0..n).map(|_| thread_rng.gen_range(0..n)).collect();
            let x_sample = x.select(ndarray::Axis(0), &sample_indices);
            let y_sample = y.select(ndarray::Axis(0), &sample_indices);
            let (fit_res, tg) =
                grid::fitter::fit(x_sample.view(), y_sample.view(), tg_params, &mut thread_rng);
            println!("err: {:?}", fit_res.err);
            tg
        })
        .collect();

    let combined_tree_grid = if hyperparameters.tg_params.identified {
        Some(combine_into_single_tree_grid(&tree_grids))
    } else {
        None
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

#[derive(Debug, Clone)]
pub struct BoostedVariant {
    combined_tree_grid: Option<FittedTreeGrid>,
}

impl AggregationMethod for BoostedVariant {
    const AGGREGATION_METHOD: Aggregation = Aggregation::Sum;
}

pub fn combine_into_single_tree_grid(grids: &[FittedTreeGrid]) -> FittedTreeGrid {
    println!("Combining tree grids into a single tree grid.");
    let reference = &grids[0];

    let aligned_signs = get_aligned_signs_for_all_tree_grids(grids);
    let num_axes = reference.splits.len();

    let mut combined_splits: Vec<Vec<f64>> = Vec::with_capacity(num_axes);
    let mut combined_intervals: Vec<Vec<(f64, f64)>> = Vec::with_capacity(num_axes);
    let mut combined_grid_values: Vec<Vec<f64>> = Vec::with_capacity(num_axes);

    let scalings: Vec<f64> = grids.iter().map(|grid| grid.scaling).collect();
    let combined_scaling = scalings.iter().sum::<f64>() / scalings.len() as f64;

    // Process each axis separately.
    for axis in 0..num_axes {
        // Collect and deduplicate splits
        let mut splits: Vec<f64> = grids
            .iter()
            .flat_map(|grid| grid.splits[axis].iter().copied()) // Avoid cloning, just copy f64
            .collect();

        splits.sort_by(|a, b| a.partial_cmp(b).unwrap());
        splits.dedup_by(|a, b| (*a - *b).abs() < 1e-12);

        // Create new intervals
        let mut new_intervals: Vec<(f64, f64)> = Vec::new();
        if splits.is_empty() {
            new_intervals.push((f64::NEG_INFINITY, f64::INFINITY));
        } else {
            new_intervals.push((f64::NEG_INFINITY, splits[0]));
            for i in 0..splits.len() - 1 {
                new_intervals.push((splits[i], splits[i + 1]));
            }
            new_intervals.push((splits[splits.len() - 1], f64::INFINITY));
        }
        combined_intervals.push(new_intervals.clone()); // Store intervals for later use

        // Prepare to collect combined grid values for this axis
        let mut new_grid_values: Vec<f64> = Vec::with_capacity(new_intervals.len());

        // For each new interval, combine grid values from all treegrids
        for &(a, b) in &new_intervals {
            let mut values: Vec<f64> = Vec::with_capacity(grids.len());
            for (grid_index, grid) in grids.iter().enumerate() {
                // Efficiently find the grid value for the interval [a, b)
                for (interval_index, &(ia, ib)) in grid.intervals[axis].iter().enumerate() {
                    if a >= ia && b <= ib {
                        values.push(
                            aligned_signs[grid_index][axis]
                                * grid.grid_values[axis][interval_index],
                        );
                        break; // Move to the next grid after finding a value
                    }
                }
            }
            // Combine values by taking simple average (handle empty values case)
            let combined_val = if values.is_empty() {
                0.0 // Or handle as NaN, or based on domain knowledge
            } else {
                values.iter().sum::<f64>() / values.len() as f64
            };
            new_grid_values.push(combined_val);
        }
        combined_grid_values.push(new_grid_values);
        combined_splits.push(splits);
    }

    FittedTreeGrid::new(
        combined_splits,
        combined_intervals,
        combined_grid_values,
        combined_scaling,
    )
}

impl TreeGridFamily<BoostedVariant> {
    fn predict_majority_voted_sign(&self, x: ArrayView2<f64>) -> Array1<f64> {
        let mut result = Array1::ones(x.shape()[0]);
        let mut signs = Array1::from_elem(x.shape()[0], 0.0);
        for grid in &self.0 {
            let pred = grid.predict(x.view());
            result *= &pred;
            signs += &pred.signum();
        }

        signs = signs.signum();

        result.zip_mut_with(&signs, |v, sign| {
            *v = sign * (*v).abs().powf(1.0 / self.0.len() as f64);
        });

        result
    }

    fn predict_arithmetic_mean(&self, x: ArrayView2<f64>) -> Array1<f64> {
        let mut result = Array1::zeros(x.shape()[0]);
        for grid in &self.0 {
            result += &grid.predict(x.view());
        }
        result /= self.0.len() as f64;
        result
    }

    pub fn get_aligned_tree_grids(&self) -> Vec<FittedTreeGrid> {
        let aligned_signs = get_aligned_signs_for_all_tree_grids(&self.0);
        self.0
            .iter()
            .zip(aligned_signs.iter())
            .enumerate()
            .map(|(i, (grid, signs))| {
                let mut new_grid = grid.clone();
                let ipr: Vec<f64> = (0..new_grid.grid_values.len())
                    .map(|dim| compute_inner_product(&new_grid, &self.0[0], dim))
                    .collect();

                for (axis, sign) in signs.iter().enumerate() {
                    new_grid.scaling *= sign;
                    new_grid.grid_values[axis] = new_grid.grid_values[axis]
                        .iter()
                        .map(|v| v * sign)
                        .collect();
                }
                new_grid
            })
            .collect()
    }

    pub fn combine_into_single_tree_grid(&self) -> FittedTreeGrid {
        combine_into_single_tree_grid(&self.0)
    }
}

impl FittedModel for TreeGridFamily<BoostedVariant> {
    fn predict(&self, x: ArrayView2<f64>) -> Array1<f64> {
        if let Some(combined_tree_grid) = &self.1.combined_tree_grid {
            combined_tree_grid.predict(x)
        } else {
            self.predict_majority_voted_sign(x)
        }
    }
}

#[derive(Debug)]
pub struct TreeGridFamilyBoostedParams {
    pub B: usize,
    pub tg_params: TreeGridParams,
}

// Builder for TreeGridFamilyBoostedParams
#[derive(Debug)]
pub struct TreeGridFamilyBoostedParamsBuilder {
    B: usize,
    tg_params_builder: TreeGridParamsBuilder,
}

impl TreeGridFamilyBoostedParamsBuilder {
    pub fn new() -> Self {
        Self {
            B: 100,
            tg_params_builder: TreeGridParamsBuilder::new(),
        }
    }

    pub fn B(mut self, B: usize) -> Self {
        self.B = B;
        self
    }

    // Convenience methods for TreeGridParams configuration
    pub fn n_iter(mut self, n_iter: usize) -> Self {
        self.tg_params_builder = self.tg_params_builder.n_iter(n_iter);
        self
    }

    pub fn split_try(mut self, split_try: usize) -> Self {
        self.tg_params_builder = self.tg_params_builder.split_try(split_try);
        self
    }

    pub fn colsample_bytree(mut self, colsample_bytree: f64) -> Self {
        self.tg_params_builder = self.tg_params_builder.colsample_bytree(colsample_bytree);
        self
    }

    pub fn identified(mut self, identified: bool) -> Self {
        self.tg_params_builder = self.tg_params_builder.identified(identified);
        self
    }

    pub fn build(self) -> TreeGridFamilyBoostedParams {
        TreeGridFamilyBoostedParams {
            B: self.B,
            tg_params: self.tg_params_builder.build(),
        }
    }
}

impl Default for TreeGridFamilyBoostedParamsBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for TreeGridFamilyBoostedParams {
    fn default() -> Self {
        TreeGridFamilyBoostedParamsBuilder::new().build()
    }
}

#[cfg(test)]
mod tests {

    use ndarray::Array1;

    use crate::{
        forest::forest_fitter::{fit_boosted, MPFBoostedParams},
        test_data::setup_data_csv,
        tree_grid::grid::{params::TreeGridParams, FittedTreeGrid},
        FittedModel,
    };

    use super::TreeGridFamilyBoostedParams;

    #[test]
    fn test_merged_tree_grids_predicts_the_same() {
        let (x, y) = setup_data_csv();

        let (fit_result, mpf) = fit_boosted(
            x.view(),
            y.view(),
            &MPFBoostedParams {
                epochs: 2,
                seed: 42,
                tgf_params: TreeGridFamilyBoostedParams {
                    B: 20,
                    tg_params: TreeGridParams::default(),
                },
            },
        );

        let merged_tree_grids: Vec<FittedTreeGrid> = mpf
            .get_tree_grid_families()
            .iter()
            .map(|tgf| tgf.combine_into_single_tree_grid())
            .collect();

        let merged_predictions: Vec<Array1<f64>> = merged_tree_grids
            .iter()
            .map(|tg| tg.predict(x.view()))
            .collect();

        let mpf_predictions: Vec<Array1<f64>> = mpf
            .get_tree_grid_families()
            .iter()
            .map(|tgf| tgf.predict(x.view()))
            .collect();

        for (merged_pred, mpf_pred) in merged_predictions.iter().zip(mpf_predictions.iter()) {
            let diff = merged_pred - mpf_pred;
            println!(
                "diff max: {:?}",
                diff.iter().max_by(|a, b| a.partial_cmp(b).unwrap())
            );
            println!(
                "diff min: {:?}",
                diff.iter().min_by(|a, b| a.partial_cmp(b).unwrap())
            );
        }

        let mpf_pred = mpf_predictions
            .iter()
            .fold(Array1::zeros(x.nrows()), |acc: Array1<f64>, pred| {
                acc + pred
            });

        let merged_pred = merged_predictions
            .iter()
            .fold(Array1::zeros(x.nrows()), |acc: Array1<f64>, pred| {
                acc + pred
            });
    }
}
