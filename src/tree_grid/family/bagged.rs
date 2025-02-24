use ndarray::{Array1, ArrayView1, ArrayView2};
use rand::Rng;

use crate::{
    tree_grid::grid::{
        self, compute_inner_product, fitter::TreeGridParams, get_aligned_signs_for_all_tree_grids,
    },
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
    let tree_grids_iter;
    let n = x.nrows();

    #[cfg(not(feature = "use-rayon"))]
    {
        tree_grids_iter = 0..*B
    }
    #[cfg(feature = "use-rayon")]
    {
        tree_grids_iter = (0..*B).into_par_iter()
    }

    let tree_grids: Vec<FittedTreeGrid> = tree_grids_iter
        .map(|b| {
            // Common sampling code without shifting.
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

    let tgf = TreeGridFamily(tree_grids, BaggedVariant);

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
pub struct BaggedVariant;

impl AggregationMethod for BaggedVariant {
    const AGGREGATION_METHOD: Aggregation = Aggregation::Sum;
}

fn geometric_mean(values: &[f64]) -> f64 {
    let n = values.len() as f64;
    let eps = 1e-12;
    let mut sum_log = 0.0;
    let mut neg_count = 0;
    for &v in values {
        if v < 0.0 {
            neg_count += 1;
        }
        // Add a small epsilon to avoid log(0).
        sum_log += (v.abs() + eps).ln();
    }
    let mean_log = sum_log / n;
    let gmean = mean_log.exp();
    if (neg_count as f64) > n / 2.0 {
        -gmean
    } else {
        gmean
    }
}

impl TreeGridFamily<BaggedVariant> {
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
        println!("Combining tree grids into a single tree grid.");
        let grids = &self.0;
        let reference = &grids[0];

        let aligned_signs = get_aligned_signs_for_all_tree_grids(grids);
        let num_axes = reference.splits.len();

        let mut combined_splits: Vec<Vec<f64>> = Vec::with_capacity(num_axes);
        let mut combined_intervals: Vec<Vec<(f64, f64)>> = Vec::with_capacity(num_axes);
        let mut combined_grid_values: Vec<Vec<f64>> = Vec::with_capacity(num_axes);

        let scalings = grids.iter().map(|grid| grid.scaling).collect::<Vec<f64>>();
        // Process each axis separately.
        for axis in 0..num_axes {
            let mut splits: Vec<f64> = grids
                .iter()
                .flat_map(|grid| grid.splits[axis].clone())
                .collect();

            // Sort the endpoints and remove duplicates (allowing for floating-point tolerance).
            splits.sort_by(|a, b| a.partial_cmp(b).unwrap());
            splits.dedup_by(|a, b| (*a - *b).abs() < 1e-12);

            // Create new (refined) intervals from consecutive endpoints.
            let mut new_intervals: Vec<(f64, f64)> = Vec::new();
            new_intervals.push((f64::NEG_INFINITY, splits[0]));
            for i in 0..splits.len() - 1 {
                new_intervals.push((splits[i], splits[i + 1]));
            }
            new_intervals.push((splits[splits.len() - 1], f64::INFINITY));

            // For each new interval, combine the grid values from all treegrids.
            let mut new_grid_values: Vec<f64> = Vec::new();
            for &(a, b) in &new_intervals {
                let mut values: Vec<f64> = Vec::new();
                // For each treegrid, find the grid value for which [a,b) is contained in its interval.
                for (idx, grid) in grids.iter().enumerate() {
                    let mut found_value = None;
                    for (i, &(ia, ib)) in grid.intervals[axis].iter().enumerate() {
                        if a >= ia && b <= ib {
                            found_value = Some(grid.grid_values[axis][i]);
                            break;
                        }
                    }
                    if let Some(val) = found_value {
                        values.push(aligned_signs[idx][axis] * val);
                    }
                }
                // Combine these values by taking simple average

                let combined_val = values.iter().sum::<f64>() / values.len() as f64;
                new_grid_values.push(combined_val);
            }
            combined_intervals.push(new_intervals);
            combined_splits.push(splits);
            combined_grid_values.push(new_grid_values);
        }

        let combined_scaling = scalings.iter().sum::<f64>() / scalings.len() as f64;
        FittedTreeGrid {
            splits: combined_splits,
            intervals: combined_intervals,
            grid_values: combined_grid_values,
            scaling: combined_scaling,
        }
    }
}

impl FittedModel for TreeGridFamily<BaggedVariant> {
    fn predict(&self, x: ArrayView2<f64>) -> Array1<f64> {
        self.predict_majority_voted_sign(x)
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

#[cfg(test)]
mod tests {

    use ndarray::Array1;

    use crate::{
        forest::forest_fitter::{fit_bagged, MPFBaggedParams},
        test_data::setup_data_csv,
        tree_grid::grid::{fitter::TreeGridParams, FittedTreeGrid},
        FittedModel,
    };

    use super::TreeGridFamilyBaggedParams;

    #[test]
    fn test_merged_tree_grids_predicts_the_same() {
        let (x, y) = setup_data_csv();

        let (fit_result, mpf) = fit_bagged(
            x.view(),
            y.view(),
            &MPFBaggedParams {
                epochs: 2,
                tgf_params: TreeGridFamilyBaggedParams {
                    B: 20,
                    tg_params: TreeGridParams {
                        n_iter: 10,
                        split_try: 10,
                        colsample_bytree: 1.0,
                        identified: true,
                    },
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
