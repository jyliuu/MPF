use ndarray::{Array1, ArrayView1, ArrayView2, ArrayViewMut1};
use rand::{seq::index::sample, Rng};

use super::{gridindex::GridIndex, FittedTreeGrid};

pub trait IdentificationStrategy: Send + Sync + 'static {
    fn identify(&self, grid_values: &mut [Vec<f64>], weights: &[Vec<f64>], scaling: &mut f64);
    fn combine_values(&self, values: &[f64]) -> f64;
}

pub struct L2ArithmeticMean;
pub struct L2Median;
pub struct L2ArithmeticGeometricMean;
pub struct L2GeometricMean;

impl IdentificationStrategy for L2ArithmeticMean {
    fn identify(&self, grid_values: &mut [Vec<f64>], weights: &[Vec<f64>], scaling: &mut f64) {
        identify_no_sign(grid_values, weights, scaling);
    }

    fn combine_values(&self, values: &[f64]) -> f64 {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

impl IdentificationStrategy for L2Median {
    fn identify(&self, grid_values: &mut [Vec<f64>], weights: &[Vec<f64>], scaling: &mut f64) {
        identify_no_sign(grid_values, weights, scaling);
    }

    fn combine_values(&self, values: &[f64]) -> f64 {
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

impl IdentificationStrategy for L2ArithmeticGeometricMean {
    fn identify(&self, grid_values: &mut [Vec<f64>], weights: &[Vec<f64>], scaling: &mut f64) {
        identify_no_sign(grid_values, weights, scaling);
    }

    fn combine_values(&self, values: &[f64]) -> f64 {
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

impl IdentificationStrategy for L2GeometricMean {
    fn identify(&self, grid_values: &mut [Vec<f64>], weights: &[Vec<f64>], scaling: &mut f64) {
        identify_no_sign(grid_values, weights, scaling);
    }

    fn combine_values(&self, values: &[f64]) -> f64 {
        let sign = values.iter().map(|v| v.signum()).sum::<f64>().signum();
        let abs_values: Vec<f64> = values.iter().map(|v| v.abs()).collect();
        let geom_mean = abs_values
            .iter()
            .product::<f64>()
            .powf(1.0 / abs_values.len() as f64);
        sign * geom_mean
    }
}

pub enum SplitStrategy {
    Random(RandomSplit),
    Interval(IntervalRandomSplit),
}

impl SplitStrategy {
    pub fn sample_splits<R: Rng + ?Sized>(
        &self,
        rng: &mut R,
        x: ArrayView2<f64>,
        intervals: &[Vec<(f64, f64)>],
    ) -> Vec<(usize, f64)> {
        match self {
            SplitStrategy::Random(random_split) => random_split.sample_splits(rng, x),
            SplitStrategy::Interval(interval_split) => {
                interval_split.sample_splits(rng, x, intervals)
            }
        }
    }
}
#[derive(Debug, Clone)]
pub struct RandomSplit {
    pub split_try: usize,
    pub colsample_bytree: f64,
}

#[derive(Debug, Clone)]
pub struct IntervalRandomSplit {
    pub split_try: usize,
    pub colsample_bytree: f64,
}

impl RandomSplit {
    fn sample_splits<R: Rng + ?Sized>(&self, rng: &mut R, x: ArrayView2<f64>) -> Vec<(usize, f64)> {
        let nrows = x.nrows();
        let ncols = x.ncols();
        let ncols_to_sample = (self.colsample_bytree * ncols as f64) as usize;

        let mut splits = vec![];

        for col in sample(rng, ncols, ncols_to_sample) {
            for idx in sample(rng, nrows, self.split_try) {
                splits.push((col, x[[idx, col]]));
            }
        }

        splits
    }
}

impl IntervalRandomSplit {
    fn sample_splits<R: Rng + ?Sized>(
        &self,
        rng: &mut R,
        x: ArrayView2<f64>,
        intervals: &[Vec<(f64, f64)>],
    ) -> Vec<(usize, f64)> {
        let ncols = x.ncols();
        let ncols_to_sample = (self.colsample_bytree * ncols as f64) as usize;

        let cols = sample(rng, ncols, ncols_to_sample);

        let mut splits = vec![];
        for col in cols {
            let intervals = &intervals[col];
            for (a, b) in intervals {
                let a = if a.is_infinite() {
                    *x.column(col)
                        .iter()
                        .min_by(|a, b| a.partial_cmp(b).unwrap())
                        .unwrap()
                } else {
                    *a
                };
                let b = if b.is_infinite() {
                    *x.column(col)
                        .iter()
                        .max_by(|a, b| a.partial_cmp(b).unwrap())
                        .unwrap()
                } else {
                    *b
                };

                for _ in 0..self.split_try {
                    splits.push((col, rng.gen_range(a..b)));
                }
            }
        }
        splits
    }
}

pub fn compute_initial_values(
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
) -> (f64, Vec<Vec<f64>>, Array1<f64>) {
    let mean = y.mean().unwrap();
    let init_value: f64 = mean.abs().powf(1.0 / x.ncols() as f64);
    let sign = mean.signum();
    let mut grid_values = vec![vec![init_value]; x.ncols() - 1];
    grid_values.insert(0, vec![sign * init_value]);
    let y_hat = Array1::from_vec(vec![mean; x.nrows()]);
    (mean, grid_values, y_hat)
}

const MAX_PROJECTION_ITER: usize = 10000;
pub fn reproject_grid_values(
    labels: ArrayView1<f64>,
    mut y_hat: ArrayViewMut1<f64>,
    mut residuals: ArrayViewMut1<f64>,
    grid_index: &GridIndex,
    grid_values: &mut [Vec<f64>],
) -> f64 {
    let mut err = residuals.pow2().mean().unwrap();
    let grid_cells_along_dim = (0..grid_values.len())
        .map(|dim| {
            (0..grid_values[dim].len())
                .map(|idx| grid_index.collect_fixed_axis_cells(dim, idx))
                .collect()
        })
        .collect::<Vec<Vec<Vec<usize>>>>();
    for i in 0..MAX_PROJECTION_ITER {
        for (dim, curr_grid_values) in grid_values.iter_mut().enumerate() {
            for (idx, x) in curr_grid_values.iter_mut().enumerate() {
                let mut numerator = 0.0;
                let mut denominator = 0.0;

                for cell_idx in &grid_cells_along_dim[dim][idx] {
                    for &i in &grid_index.cells[*cell_idx] {
                        let numerator_sum: f64 = residuals[i] * y_hat[i];
                        let denominator_sum = denominator + y_hat[i].powi(2);

                        numerator += numerator_sum;
                        denominator += denominator_sum;
                    }
                }
                let v_hat = if denominator > 0.0 {
                    numerator / denominator + 1.0
                } else {
                    1.0
                };
                *x *= v_hat;

                for cell_idx in &grid_cells_along_dim[dim][idx] {
                    for &i in &grid_index.cells[*cell_idx] {
                        y_hat[i] *= v_hat;
                        residuals[i] = labels[i] - y_hat[i];
                    }
                }
            }
        }

        let new_err = residuals.pow2().mean().unwrap();
        let diff = (new_err - err).abs();
        if diff < 1e-6 {
            break;
        }
        err = new_err;
    }
    err
}

pub fn identify_no_sign(grid_values: &mut [Vec<f64>], weights: &[Vec<f64>], scaling: &mut f64) {
    for dim in 0..grid_values.len() {
        let curr_weights = &weights[dim];
        let curr_grid_values = &mut grid_values[dim];

        let weights_sum: f64 = curr_weights.iter().sum();

        let l2_weighted_norm = curr_grid_values
            .iter()
            .zip(curr_weights.iter())
            .map(|(&x, &w)| x.powi(2) * w)
            .sum::<f64>();

        let scale = (l2_weighted_norm / weights_sum).sqrt();

        curr_grid_values.iter_mut().for_each(|x| *x /= scale);
        *scaling *= scale;
    }
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
    println!("Inner product: {}", inner_product);
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

pub fn combine_into_single_tree_grid<I>(
    grids: &[FittedTreeGrid],
    reference: &FittedTreeGrid,
    combine_method: &I,
    points: ArrayView2<f64>,
) -> FittedTreeGrid
where
    I: IdentificationStrategy,
{
    println!(
        "Combining {:?} tree grids into a single tree grid.",
        grids.len()
    );

    let aligned_signs = get_aligned_signs_for_all_tree_grids(grids);
    let num_axes = reference.grid_index.intervals.len();

    let mut combined_splits: Vec<Vec<f64>> = Vec::with_capacity(num_axes);
    let mut combined_intervals: Vec<Vec<(f64, f64)>> = Vec::with_capacity(num_axes);
    let mut combined_grid_values: Vec<Vec<f64>> = Vec::with_capacity(num_axes);

    // let scalings: Vec<f64> = grids
    //     .iter()
    //     .zip(&aligned_signs)
    //     .map(|(grid, signs)| {
    //         let total_sign = signs.iter().product::<f64>();
    //         grid.scaling * total_sign
    //     })
    //     .collect();
    // let combined_scaling = combine_method.combine_values(&scalings);

    // Process each axis separately.
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
                combine_method.combine_values(&values)
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
