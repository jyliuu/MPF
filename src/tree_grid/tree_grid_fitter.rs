use std::vec;

use itertools::Itertools;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use rand::{seq::SliceRandom, Rng};

use crate::{FitResult, ModelFitter};

use super::tree_grid::FittedTreeGrid;

#[derive(Debug)]
pub struct TreeGridFitter<'a> {
    pub splits: Vec<Vec<f64>>,
    pub intervals: Vec<Vec<(f64, f64)>>,
    pub grid_values: Vec<Vec<f64>>,
    pub leaf_points: Array2<usize>,
    pub labels: ArrayView1<'a, f64>,
    pub x: ArrayView2<'a, f64>,
    pub y_hat: Array1<f64>,
    pub residuals: Array1<f64>,
}

#[derive(Debug, Clone)]
pub struct TreeGridParams {
    pub n_iter: usize,
    pub split_try: usize,
    pub colsample_bytree: f64,
}

#[derive(Debug)]
pub struct SliceCandidate {
    col: usize,
    split: f64,
    index: usize,
    left: (f64, f64),
    right: (f64, f64),
}

#[derive(Debug)]
pub struct RefineCandidate {
    pub col: usize,
    pub split: f64,
    index: usize,
    left: (f64, f64),
    right: (f64, f64),
    pub update_a: f64,
    pub update_b: f64,
    a_points_idx: Vec<usize>,
    b_points_idx: Vec<usize>,
    curr_leaf_points_idx: Vec<usize>,
}

fn compute_initial_values(
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

pub fn find_slice_candidate(
    splits: &[Vec<f64>],
    intervals: &[Vec<(f64, f64)>],
    col: usize,
    split: f64,
) -> SliceCandidate {
    let splits = &splits[col];
    let intervals = &intervals[col];

    let index = splits
        .iter()
        .position(|&x| split < x)
        .unwrap_or(splits.len());

    let (begin, end) = intervals[index];
    let left = (begin, split);
    let right = (split, end);

    SliceCandidate {
        col,
        split,
        index,
        left,
        right,
    }
}

fn compute_leaf_values(leaf: &Array1<usize>, grid_values: &[Vec<f64>]) -> f64 {
    leaf.indexed_iter()
        .map(|(i, &idx)| grid_values[i][idx])
        .product()
}

pub fn find_refine_candidate(
    slice_candidate: SliceCandidate,
    x: ArrayView2<f64>,
    leaf_points: &Array2<usize>,
    grid_values: &[Vec<f64>],
    intervals: &[Vec<(f64, f64)>],
    residuals: ArrayView1<'_, f64>,
    y_hat: ArrayView1<'_, f64>,
) -> (f64, f64, RefineCandidate) {
    let SliceCandidate {
        col,
        split,
        index,
        left,
        right,
    } = slice_candidate;

    let mut dims = vec![];
    for dim in 0..x.ncols() {
        dims.push(if dim == col {
            vec![index]
        } else {
            (0..intervals[dim].len()).collect()
        });
    }
    let leaves: Vec<Array1<usize>> = dims
        .into_iter()
        .multi_cartesian_product()
        .map(Array1::from)
        .collect();

    let [mut n_a, mut n_b, mut m_a, mut m_b] = [0.0; 4];

    let curr_leaf_points_idx: Vec<usize> = leaf_points
        .index_axis(Axis(1), col)
        .indexed_iter()
        .filter(|(_, &x)| x == index)
        .map(|(i, _)| i)
        .collect();

    let curr_points = x.select(Axis(0), &curr_leaf_points_idx);

    for leaf in &leaves {
        let v = compute_leaf_values(leaf, grid_values);

        let (resids_a, resids_b): (Vec<_>, Vec<_>) = leaf_points
            .select(Axis(0), &curr_leaf_points_idx)
            .axis_iter(Axis(0))
            .enumerate()
            .filter_map(|(i, row)| {
                if row == leaf {
                    Some((residuals[i], x[(i, col)] < split))
                } else {
                    None
                }
            })
            .partition(|(_, is_a)| *is_a);

        n_a += v.powi(2) * resids_a.len() as f64;
        n_b += v.powi(2) * resids_b.len() as f64;
        m_a += v * resids_a.into_iter().map(|(r, _)| r).sum::<f64>();
        m_b += v * resids_b.into_iter().map(|(r, _)| r).sum::<f64>();
    }

    let update_a = if n_a == 0.0 { 1.0 } else { m_a / n_a + 1.0 };
    let update_b = if n_b == 0.0 { 1.0 } else { m_b / n_b + 1.0 };

    let err_old = residuals
        .select(Axis(0), &curr_leaf_points_idx)
        .pow2()
        .sum();

    let (a_points_idx, b_points_idx): (Vec<_>, Vec<_>) = curr_points
        .index_axis(Axis(1), col)
        .indexed_iter()
        .map(|(i, &x)| {
            if x < split {
                (curr_leaf_points_idx[i], true)
            } else {
                (curr_leaf_points_idx[i], false)
            }
        })
        .partition(|(_, is_a)| *is_a);

    let new_resid_a = a_points_idx
        .iter()
        .map(|(i, _)| y_hat[*i].mul_add(1.0 - update_a, residuals[*i]))
        .collect::<Array1<f64>>();
    let new_resid_b = b_points_idx
        .iter()
        .map(|(i, _)| y_hat[*i].mul_add(1.0 - update_b, residuals[*i]))
        .collect::<Array1<f64>>();

    let err_new = new_resid_a.powi(2).sum() + new_resid_b.powi(2).sum();

    let refine_candidate = RefineCandidate {
        col,
        split,
        index,
        left,
        right,
        update_a,
        update_b,
        a_points_idx: a_points_idx.into_iter().map(|(i, _)| i).collect(),
        b_points_idx: b_points_idx.into_iter().map(|(i, _)| i).collect(),
        curr_leaf_points_idx,
    };

    (err_new, err_old, refine_candidate)
}

fn update_leaf_points(
    leaf_points: &mut Array2<usize>,
    dim: usize,
    index: usize,
    leaf_points_b: &[usize],
) {
    leaf_points.axis_iter_mut(Axis(0)).for_each(|mut x| {
        if x[dim] > index {
            x[dim] += 1;
        }
    });

    for &i in leaf_points_b {
        leaf_points[(i, dim)] += 1;
    }
}

fn update_predictions(
    y_hat: &mut Array1<f64>,
    residuals: &mut Array1<f64>,
    labels: ArrayView1<f64>,
    a_points_idx: &[usize],
    b_points_idx: &[usize],
    update_a: f64,
    update_b: f64,
) {
    for &i in a_points_idx {
        y_hat[i] *= update_a;
        residuals[i] = labels[i] - y_hat[i];
    }
    for &i in b_points_idx {
        y_hat[i] *= update_b;
        residuals[i] = labels[i] - y_hat[i];
    }
}

impl TreeGridFitter<'_> {
    pub fn update_tree(&mut self, refine_candidate: RefineCandidate) {
        let RefineCandidate {
            col,
            split,
            index,
            left,
            right,
            update_a,
            update_b,
            a_points_idx,
            b_points_idx,
            curr_leaf_points_idx: _,
        } = refine_candidate;

        let old_grid_value = self.grid_values[col][index];
        self.grid_values[col][index] *= update_a;
        self.grid_values[col].insert(index + 1, old_grid_value * update_b);

        self.splits[col].insert(index, split);
        self.intervals[col][index] = left;
        self.intervals[col].insert(index + 1, right);

        update_leaf_points(&mut self.leaf_points, col, index, &b_points_idx);
        update_predictions(
            &mut self.y_hat,
            &mut self.residuals,
            self.labels,
            &a_points_idx,
            &b_points_idx,
            update_a,
            update_b,
        );
    }
}

impl<'a> ModelFitter<'a> for TreeGridFitter<'a> {
    type HyperParameters = TreeGridParams;
    type Model = FittedTreeGrid;
    type Features = ArrayView2<'a, f64>;
    type Labels = ArrayView1<'a, f64>;

    fn new(x: Self::Features, y: Self::Labels) -> Self {
        let leaf_points = Array2::zeros((x.nrows(), x.ncols()));
        let splits = vec![vec![]; x.ncols()];
        let intervals = vec![vec![(f64::NEG_INFINITY, f64::INFINITY)]; x.ncols()];

        let (_, grid_values, y_hat) = compute_initial_values(x, y);
        let residuals = y.to_owned() - &y_hat;

        TreeGridFitter {
            splits,
            intervals,
            grid_values,
            leaf_points,
            labels: y,
            x,
            y_hat,
            residuals,
        }
    }

    fn fit(mut self, hyperparameters: Self::HyperParameters) -> (FitResult, Self::Model) {
        let mut rng = rand::thread_rng();
        let n_cols = self.x.ncols();
        let n_rows = self.x.nrows();

        for _ in 0..hyperparameters.n_iter {
            let n_cols_to_sample = (hyperparameters.colsample_bytree * n_cols as f64) as usize;
            let split_idx: Vec<usize> = (0..hyperparameters.split_try)
                .map(|_| rng.gen_range(0..n_rows))
                .collect();

            let mut possible_indices: Vec<usize> = (0..n_cols).collect();
            possible_indices.shuffle(&mut rng);

            let col_idx = possible_indices[0..n_cols_to_sample].to_vec();

            let mut best_candidate: Option<RefineCandidate> = None;
            let mut best_err_diff = f64::NEG_INFINITY;

            for &col in &col_idx {
                for &idx in &split_idx {
                    let split = self.x[[idx, col]];
                    let slice_candidate =
                        find_slice_candidate(&self.splits, &self.intervals, col, split);
                    let (err_new, err_old, refine_candidate) = find_refine_candidate(
                        slice_candidate,
                        self.x,
                        &self.leaf_points,
                        &self.grid_values,
                        &self.intervals,
                        self.residuals.view(),
                        self.y_hat.view(),
                    );

                    let err_diff = err_old - err_new;
                    if err_diff > best_err_diff {
                        best_candidate = Some(refine_candidate);
                        best_err_diff = err_diff;
                    }
                }
            }

            if let Some(candidate) = best_candidate {
                self.update_tree(candidate);
            }
        }

        let err = self.residuals.pow2().mean().unwrap();

        let residuals = self.residuals;
        let y_hat = self.y_hat;
        let tree_grid = FittedTreeGrid {
            splits: self.splits,
            intervals: self.intervals,
            grid_values: self.grid_values,
        };

        let fit_res = FitResult {
            err,
            residuals,
            y_hat,
        };

        (fit_res, tree_grid)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup_data() -> (Array2<f64>, Array1<f64>) {
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![0.1, 0.2, 0.3, 0.5, 0.7, 0.6, 1.1, 1.2, 1.3, 1.5, 1.7, 1.6],
        )
        .unwrap();

        let y = Array1::from_vec(vec![0.5, 0.5, 0.5, 1.5, 1.5, 1.5]);

        (x, y)
    }

    #[test]
    fn test_tree_grid_refine_candidate() {
        let (x, y) = setup_data();
        let tree_grid = TreeGridFitter::new(x.view(), y.view());
        let slice_candidate = find_slice_candidate(&tree_grid.splits, &tree_grid.intervals, 0, 1.0);
        let (_, _, refine_candidate) = find_refine_candidate(
            slice_candidate,
            x.view(),
            &tree_grid.leaf_points,
            &tree_grid.grid_values,
            &tree_grid.intervals,
            tree_grid.residuals.view(),
            tree_grid.y_hat.view(),
        );
        assert_eq!(refine_candidate.update_a, 0.5);
        assert_eq!(refine_candidate.update_b, 1.5);
    }

    #[test]
    fn test_tree_grid_slice_and_refine() {
        let (x, y) = setup_data();
        let mut tree_grid = TreeGridFitter::new(x.view(), y.view());

        let slice_candidate = find_slice_candidate(&tree_grid.splits, &tree_grid.intervals, 0, 1.0);
        let (_, _, refine_candidate) = find_refine_candidate(
            slice_candidate,
            x.view(),
            &tree_grid.leaf_points,
            &tree_grid.grid_values,
            &tree_grid.intervals,
            tree_grid.residuals.view(),
            tree_grid.y_hat.view(),
        );
        tree_grid.update_tree(refine_candidate);
        assert_eq!(tree_grid.grid_values[0], vec![0.5, 1.5]);
        assert_eq!(tree_grid.splits[0], vec![1.0]);
        assert_eq!(
            tree_grid.intervals[0],
            vec![(f64::NEG_INFINITY, 1.0), (1.0, f64::INFINITY)]
        );
        assert_eq!(tree_grid.residuals.sum(), 0.0);
    }
}
