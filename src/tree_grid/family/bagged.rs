use ndarray::{Array1, ArrayView1, ArrayView2};
use rand::Rng;

use crate::{
    tree_grid::grid::{self, fitter::TreeGridParams},
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

    pub fn combine_into_single_tree_grid(&self) -> FittedTreeGrid {
        let grids = &self.0;
        let num_axes = grids[0].splits.len();
        let mut combined_splits: Vec<Vec<f64>> = Vec::with_capacity(num_axes);
        let mut combined_intervals: Vec<Vec<(f64, f64)>> = Vec::with_capacity(num_axes);
        let mut combined_grid_values: Vec<Vec<f64>> = Vec::with_capacity(num_axes);

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
                for grid in grids {
                    let mut found_value = None;
                    for (i, &(ia, ib)) in grid.intervals[axis].iter().enumerate() {
                        if a >= ia && b <= ib {
                            found_value = Some(grid.grid_values[axis][i]);
                            break;
                        }
                    }
                    if let Some(val) = found_value {
                        values.push(val);
                    }
                }
                // Combine these values using a geometric mean that handles negatives.
                let combined_val = geometric_mean(&values);
                new_grid_values.push(combined_val);
            }
            combined_intervals.push(new_intervals);
            combined_splits.push(splits);
            combined_grid_values.push(new_grid_values);
        }

        FittedTreeGrid {
            splits: combined_splits,
            intervals: combined_intervals,
            grid_values: combined_grid_values,
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
