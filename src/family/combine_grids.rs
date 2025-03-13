use std::ptr;

use ndarray::ArrayView2;

use crate::grid::{grid_index::GridIndex, FittedTreeGrid};

pub trait CombinationStrategy: Send + Sync + 'static {
    fn combine_values(values: &[f64]) -> f64;
}

pub struct ArithmeticMean;
pub struct Median;
pub struct ArithmeticGeometricMean;
pub struct GeometricMean;

impl CombinationStrategy for ArithmeticMean {
    fn combine_values(values: &[f64]) -> f64 {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

impl CombinationStrategy for Median {
    fn combine_values(values: &[f64]) -> f64 {
        let len = values.len();
        if len == 0 {
            return 0.0; // Handle empty array case
        }

        if len == 1 {
            return values[0]; // Single value case
        }

        // For small arrays, use partial sort which is more efficient
        if len <= 10 {
            let mut partial = values.to_vec();
            let mid = len / 2;

            // Use partial_sort which is O(n) instead of O(n log n)
            partial.select_nth_unstable_by(mid, |a, b| a.partial_cmp(b).unwrap());

            if len % 2 == 0 {
                // Even length - need to find the other middle element
                let max_of_lower = partial[..mid]
                    .iter()
                    .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                (max_of_lower + partial[mid]) / 2.0
            } else {
                partial[mid]
            }
        } else {
            // For larger arrays, use a more efficient selection algorithm
            // that doesn't need to allocate a new vector
            let mid = len / 2;

            // Clone only if necessary
            let mut v: Vec<f64> = values.to_vec();

            // Find the median using quickselect algorithm (O(n) average case)
            let (_, median_val, _) =
                v.select_nth_unstable_by(mid, |a, b| a.partial_cmp(b).unwrap());

            // large arrays precision not so important
            *median_val
        }
    }
}

impl CombinationStrategy for ArithmeticGeometricMean {
    fn combine_values(values: &[f64]) -> f64 {
        // Collect positive and negative values into vectors.
        let positive: Vec<f64> = values.iter().copied().filter(|v| *v >= 0.0).collect();
        let negative: Vec<f64> = values
            .iter()
            .copied()
            .filter(|v| *v < 0.0)
            .map(|v| -v)
            .collect();

        let positive_count = positive.len() as f64;
        let negative_count = negative.len() as f64;

        // Guard against division by zero.
        let positive_geom_mean = if positive_count > 0.0 {
            let positive_product: f64 = positive.iter().product();
            positive_product.powf(1.0 / positive_count)
        } else {
            0.0
        };

        let negative_geom_mean = if negative_count > 0.0 {
            let negative_product: f64 = negative.iter().product();
            negative_product.powf(1.0 / negative_count)
        } else {
            0.0
        };

        (positive_count * positive_geom_mean - negative_count * negative_geom_mean)
            / (positive_count + negative_count)
    }
}

impl CombinationStrategy for GeometricMean {
    fn combine_values(values: &[f64]) -> f64 {
        let sign = values.iter().map(|v| v.signum()).sum::<f64>().signum();
        let abs_values: Vec<f64> = values.iter().map(|v| v.abs()).collect();
        let geom_mean = abs_values
            .iter()
            .product::<f64>()
            .powf(1.0 / abs_values.len() as f64);
        sign * geom_mean
    }
}

pub fn compute_inner_product_2_pointers(
    first_obs_count: &[usize],
    second_obs_count: &[usize],
    first_values: &[f64],
    second_values: &[f64],
) -> f64 {
    let mut inner_product = 0.0;
    let mut i = 0;
    let mut j = 0;

    // Track cumulative counts for both grids
    let mut cum_a = 0;
    let mut cum_b = 0;

    // Current interval boundaries
    let mut current_a = if !first_obs_count.is_empty() {
        first_obs_count[0]
    } else {
        0
    };
    let mut current_b = if !second_obs_count.is_empty() {
        second_obs_count[0]
    } else {
        0
    };

    while i < first_obs_count.len() && j < second_obs_count.len() {
        let overlap_start = cum_a.max(cum_b);
        let overlap_end = current_a.min(current_b);

        if overlap_end > overlap_start {
            let n_eff = overlap_end - overlap_start;
            inner_product += n_eff as f64 * first_values[i] * second_values[j];
        }

        // Move the pointer with the smaller current endpoint
        if current_a < current_b {
            i += 1;
            cum_a = current_a;
            current_a += first_obs_count.get(i).copied().unwrap_or(0);
        } else {
            j += 1;
            cum_b = current_b;
            current_b += second_obs_count.get(j).copied().unwrap_or(0);
        }
    }

    inner_product / first_obs_count.iter().sum::<usize>() as f64
}

pub fn compute_inner_product(first: &FittedTreeGrid, second: &FittedTreeGrid, dim: usize) -> f64 {
    let first_splits = &first.grid_index.boundaries[dim];
    let first_intervals = &first.grid_index.intervals[dim];
    let first_values = &first.grid_values[dim];

    let second_splits = &second.grid_index.boundaries[dim];
    let second_intervals = &second.grid_index.intervals[dim];
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

pub fn get_aligned_signs_for_all_tree_grids(
    tree_grids: &[FittedTreeGrid],
    reference: &FittedTreeGrid,
) -> Vec<Vec<f64>> {
    let aligned_signs: Vec<Vec<f64>> = tree_grids
        .iter()
        .map(|grid| {
            if ptr::eq(grid, reference) {
                vec![1.0; grid.grid_values.len()]
            } else {
                align_treegrid_to_reference_signs(grid, reference)
            }
        })
        .collect();

    aligned_signs
}

pub fn compute_treegrid_to_reference_inner_products(
    first: &FittedTreeGrid,
    reference: &FittedTreeGrid,
) -> Vec<f64> {
    (0..first.grid_index.intervals.len())
        .map(|dim| {
            compute_inner_product_2_pointers(
                &first.grid_index.observation_counts[dim],
                &reference.grid_index.observation_counts[dim],
                &first.grid_values[dim],
                &reference.grid_values[dim],
            )
        })
        .collect()
}

pub fn get_inner_products_for_all_tree_grids(
    tree_grids: &[FittedTreeGrid],
    reference: &FittedTreeGrid,
) -> Vec<Vec<f64>> {
    tree_grids
        .iter()
        .map(|grid| compute_treegrid_to_reference_inner_products(grid, reference))
        .collect()
}

pub fn get_all_pairwise_inner_products(tree_grids: &[FittedTreeGrid]) -> Vec<Vec<Vec<f64>>> {
    let n = tree_grids.len();
    if n == 0 {
        return Vec::new();
    }

    let num_axes = tree_grids[0].grid_index.intervals.len();
    let mut results = vec![vec![vec![0.0; num_axes]; n]; n];

    // Parallel iteration over upper triangle only
    tree_grids.iter().enumerate().for_each(|(i, grid_i)| {
        // Diagonal elements (self inner products)
        for axis in 0..num_axes {
            results[i][i][axis] = compute_inner_product_2_pointers(
                &grid_i.grid_index.observation_counts[axis],
                &grid_i.grid_index.observation_counts[axis],
                &grid_i.grid_values[axis],
                &grid_i.grid_values[axis],
            );
        }

        // Upper triangle elements
        tree_grids[i + 1..]
            .iter()
            .enumerate()
            .for_each(|(j_offset, grid_j)| {
                let j = i + 1 + j_offset;
                let ips: Vec<f64> = (0..num_axes)
                    .map(|axis| {
                        compute_inner_product_2_pointers(
                            &grid_i.grid_index.observation_counts[axis],
                            &grid_j.grid_index.observation_counts[axis],
                            &grid_i.grid_values[axis],
                            &grid_j.grid_values[axis],
                        )
                    })
                    .collect();

                // Mirror results
                results[i][j] = ips.clone();
                results[j][i] = ips;
            });
    });

    results
}

pub fn combine_into_single_tree_grid<I>(
    grids: &[FittedTreeGrid],
    points: ArrayView2<f64>,
    similarity_threshold: f64,
) -> FittedTreeGrid
where
    I: CombinationStrategy,
{
    println!(
        "Combining {:?} tree grids into a single tree grid.",
        grids.len()
    );

    // Compute all pairwise inner products
    let pairwise_ips = get_all_pairwise_inner_products(grids);

    let min_avg_abs_ip = grids.iter().enumerate().map(|(i, grid)| {
        (
            i,
            pairwise_ips[i]
                .iter()
                .map(|v| {
                    v.iter()
                        .map(|ip| ip.abs())
                        .min_by(|a, b| a.partial_cmp(b).unwrap())
                        .unwrap_or(0.0)
                })
                .sum::<f64>()
                / pairwise_ips[i].len() as f64,
        )
    });
    let (reference_idx, threshold) = min_avg_abs_ip
        .max_by(|&a, &b| a.1.partial_cmp(&b.1).unwrap())
        .unwrap();
    println!("Reference index: {}", reference_idx);
    println!("Threshold: {}", threshold);
    let reference = &grids[reference_idx];

    // Existing filtering logic
    let inner_products = get_inner_products_for_all_tree_grids(grids, reference);
    let (grids, filtered_inner_products): (Vec<&FittedTreeGrid>, Vec<Vec<f64>>) = {
        // Calculate scores for each grid
        let mut scored_grids: Vec<(usize, f64)> = inner_products
            .iter()
            .enumerate()
            .map(|(i, ips)| {
                let score = ips
                    .iter()
                    .map(|v| v.abs())
                    .min_by(|a, b| a.partial_cmp(b).unwrap())
                    .unwrap_or(0.0);
                (i, score)
            })
            .collect();

        // Sort by score descending
        scored_grids.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Calculate number to keep (top (1 - similarity_threshold)%)
        let keep_count = ((1.0 - similarity_threshold) * grids.len() as f64).round() as usize;
        let keep_count = keep_count.clamp(1, grids.len()); // Ensure at least 1

        // Take top candidates
        scored_grids
            .into_iter()
            .take(keep_count)
            .map(|(i, _)| (&grids[i], inner_products[i].clone()))
            .unzip()
    };
    println!("Candidate indices: {:?}", grids.len());

    let aligned_signs: Vec<Vec<f64>> = filtered_inner_products
        .iter()
        .map(|inner_products| inner_products.iter().map(|ip| ip.signum()).collect())
        .collect();

    let num_axes = reference.grid_index.intervals.len();

    let mut combined_splits: Vec<Vec<f64>> = Vec::with_capacity(num_axes);
    let mut combined_intervals: Vec<Vec<(f64, f64)>> = Vec::with_capacity(num_axes);
    let mut combined_grid_values: Vec<Vec<f64>> = Vec::with_capacity(num_axes);

    for axis in 0..num_axes {
        // Collect and deduplicate splits
        let mut splits: Vec<f64> = grids
            .iter()
            .flat_map(|grid| grid.grid_index.boundaries[axis].iter().copied()) // Avoid cloning, just copy f64
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
                for (interval_index, &(ia, ib)) in
                    grid.grid_index.intervals[axis].iter().enumerate()
                {
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
                I::combine_values(&values)
            };
            new_grid_values.push(combined_val);
        }
        combined_grid_values.push(new_grid_values);
        combined_splits.push(splits);
    }

    FittedTreeGrid::new(
        combined_grid_values,
        1.0,
        GridIndex::from_boundaries_and_points(combined_splits, points),
    )
}
