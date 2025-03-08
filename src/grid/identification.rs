pub trait IdentificationStrategy: Send + Sync + 'static {
    fn identify(grid_values: &mut [Vec<f64>], weights: &[Vec<f64>], scaling: &mut f64);
}

pub struct L2Identification;
pub struct L1Identification;

impl IdentificationStrategy for L2Identification {
    fn identify(grid_values: &mut [Vec<f64>], weights: &[Vec<f64>], scaling: &mut f64) {
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
}

impl IdentificationStrategy for L1Identification {
    fn identify(grid_values: &mut [Vec<f64>], weights: &[Vec<f64>], scaling: &mut f64) {
        for dim in 0..grid_values.len() {
            let curr_weights = &weights[dim];
            let curr_grid_values = &mut grid_values[dim];

            let weights_sum: f64 = curr_weights.iter().sum();

            let l1_weighted_norm = curr_grid_values
                .iter()
                .zip(curr_weights.iter())
                .map(|(&x, &w)| x.abs() * w)
                .sum::<f64>();

            let scale = l1_weighted_norm / weights_sum;

            curr_grid_values.iter_mut().for_each(|x| *x /= scale);
            *scaling *= scale;
        }
    }
}
