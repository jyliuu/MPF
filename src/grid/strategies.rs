use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use rand::{seq::index::sample, Rng};

pub trait SplitStrategy {
    fn sample_splits<R: Rng + ?Sized>(&self, rng: &mut R) -> (Vec<usize>, Vec<usize>);
}
#[derive(Debug, Clone)]
pub struct RandomSplit {
    pub split_try: usize,
    pub ncols_to_sample: usize,
    pub nrows: usize,
    pub ncols: usize,
}

impl SplitStrategy for RandomSplit {
    fn sample_splits<R: Rng + ?Sized>(&self, rng: &mut R) -> (Vec<usize>, Vec<usize>) {
        let split_idx: Vec<usize> = sample(rng, self.nrows, self.split_try)
            .into_iter()
            .collect();
        let col_idx: Vec<usize> = sample(rng, self.ncols, self.ncols_to_sample)
            .into_iter()
            .collect();
        (split_idx, col_idx)
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

pub fn get_component_weights(
    leaf_points: &Array2<usize>,
    grid_values: &[Vec<f64>],
) -> Vec<Vec<f64>> {
    let mut weights: Vec<Vec<f64>> = grid_values.iter().map(|col| vec![0.0; col.len()]).collect();

    for row in leaf_points.axis_iter(ndarray::Axis(0)) {
        for (j, &idx) in row.iter().enumerate() {
            weights[j][idx] += 1.0;
        }
    }

    weights
}

const MAX_PROJECTION_ITER: usize = 100;

pub fn reproject_grid_values(
    x: ArrayView2<f64>,
    leaf_points: &Array2<usize>,
    grid_values: &mut [Vec<f64>],
    labels: ArrayView1<'_, f64>,
    y_hat: ArrayView1<'_, f64>,
) {
    let mut y_hat_clone = y_hat.to_owned();
    let mut residuals = labels.to_owned() - &y_hat_clone;
    let mut err = residuals.pow2().sum();
    for i in 0..MAX_PROJECTION_ITER {
        for (dim, curr_grid_values) in grid_values.iter_mut().enumerate() {
            for (idx, x) in curr_grid_values.iter_mut().enumerate() {
                let curr_leaf_points_idx: Vec<usize> = leaf_points
                    .index_axis(Axis(1), dim)
                    .indexed_iter()
                    .filter(|(_, &x)| x == idx)
                    .map(|(i, _)| i)
                    .collect();
                let curr_y_hat = y_hat_clone.select(Axis(0), &curr_leaf_points_idx);
                let curr_residuals = residuals.select(Axis(0), &curr_leaf_points_idx);

                let numerator = curr_residuals
                    .iter()
                    .zip(curr_y_hat.iter())
                    .map(|(r, y)| r * y)
                    .sum::<f64>();
                let denominator = curr_y_hat.pow2().sum();

                let v_hat = numerator / denominator + 1.0;
                *x *= v_hat;

                for i in &curr_leaf_points_idx {
                    y_hat_clone[*i] *= v_hat;
                    residuals[*i] = labels[*i] - y_hat_clone[*i];
                }
            }
        }
        let new_err = residuals.pow2().sum();
        if (new_err - err).abs() < 1e-6 {
            break;
        }
        err = new_err;
    }
}

pub fn identify_no_sign(grid_values: &mut [Vec<f64>], weights: &[Vec<f64>], scaling: &mut f64) {
    for dim in 0..grid_values.len() {
        let curr_weights = &weights[dim];
        let curr_grid_values = &mut grid_values[dim];

        let weights_sum: f64 = curr_weights.iter().sum();

        let weighted_mean = curr_grid_values
            .iter()
            .zip(curr_weights.iter())
            .map(|(&x, &w)| x * w)
            .sum::<f64>()
            / weights_sum;

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
