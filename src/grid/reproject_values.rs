use ndarray::{ArrayView1, ArrayViewMut1};

use super::grid_index::GridIndex;

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
    for _ in 0..MAX_PROJECTION_ITER {
        for (dim, curr_grid_values) in grid_values.iter_mut().enumerate() {
            for (idx, x) in curr_grid_values.iter_mut().enumerate() {
                let mut numerator = 0.0;
                let mut denominator = 0.0;

                for cell_idx in &grid_cells_along_dim[dim][idx] {
                    let curr_cell_points_idx = grid_index.cells.get(cell_idx).unwrap();
                    for &i in curr_cell_points_idx {
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
                    let curr_cell_points_idx = grid_index.cells.get(cell_idx).unwrap();
                    for &i in curr_cell_points_idx {
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
