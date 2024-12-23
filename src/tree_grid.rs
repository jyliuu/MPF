use itertools::Itertools;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};

mod tests;

#[derive(Debug)]
pub struct TreeGrid<'a> {
    splits: Vec<Vec<f32>>,
    intervals: Vec<Vec<(f32, f32)>>,
    grid_values: Vec<Vec<f32>>,
    leaf_points: Array2<usize>,
    labels: ArrayView1<'a, f32>,
    x: ArrayView2<'a, f32>,
    y_hat: Array1<f32>,
    residuals: Array1<f32>,
}

#[derive(Debug)]
pub struct SliceCandidate {
    col: usize,
    split: f32,
    index: usize,
    left: (f32, f32),
    right: (f32, f32),
}

#[derive(Debug)]
pub struct RefineCandidate {
    col: usize,
    split: f32,
    index: usize,
    left: (f32, f32),
    right: (f32, f32),
    update_a: f32,
    update_b: f32,
    a_points_idx: Vec<usize>,
    b_points_idx: Vec<usize>,
    curr_leaf_points_idx: Vec<usize>,
}

impl<'a> TreeGrid<'a> {
    pub fn new(x: &'a Array2<f32>, y: &'a Array1<f32>) -> Self {
        let labels = y.view();

        let x = x.view();
        let leaf_points = Array2::zeros((x.nrows(), x.ncols()));
        let splits = vec![vec![]; x.ncols()];
        let intervals = vec![vec![(f32::NEG_INFINITY, f32::INFINITY)]; x.ncols()];

        let mean = labels.mean().unwrap();
        let residuals = y.clone() - mean;
        let init_value: f32 = f32::max(mean, 1.).powf(1.0 / x.ncols() as f32);
        let grid_values = vec![vec![init_value]; x.ncols()];
        let y_hat = Array1::from_vec(vec![mean; x.nrows()]);

        TreeGrid {
            splits,
            intervals,
            grid_values,
            leaf_points,
            labels,
            x,
            y_hat,
            residuals,
        }
    }

    fn slice_candidate(&self, col: usize, split: f32) -> SliceCandidate {
        let splits = &self.splits[col];
        let intervals = &self.intervals[col];

        let index = splits
            .iter()
            .position(|&x| x >= split)
            .unwrap_or(splits.len());

        let (begin, end) = intervals[index];
        let left = (begin, split);
        let right = (split, end);

        return SliceCandidate {
            col,
            split,
            index,
            left,
            right,
        };
    }

    fn refine_candidate(&self, slice_candidate: SliceCandidate) -> (f32, f32, RefineCandidate) {
        let SliceCandidate {
            col,
            split,
            index,
            left,
            right,
        } = slice_candidate;

        let mut dims = vec![];
        for dim in 0..self.x.ncols() {
            dims.push(if dim == col {
                vec![index]
            } else {
                (0..self.intervals[dim].len()).collect()
            });
        }
        let leaves: Vec<Array1<usize>> = dims
            .into_iter()
            .multi_cartesian_product()
            .map(|vec| Array1::from(vec)) // Convert each Vec<usize> to Array1<usize>
            .collect();

        let [mut n_a, mut n_b, mut m_a, mut m_b] = [0.0; 4];

        let curr_leaf_points_idx: Vec<usize> = self
            .leaf_points
            .index_axis(Axis(1), col)
            .indexed_iter()
            .filter(|(_, &x)| x == index)
            .map(|(i, _)| i)
            .collect();

        let curr_points = self.x.select(Axis(0), &curr_leaf_points_idx);
        // TODO: effectivize this loop by not looping observations only once
        for leaf in leaves {
            let v: f32 = leaf
                .indexed_iter()
                .map(|(i, &idx)| self.grid_values[i][idx])
                .product();

            let (resids_a, resids_b): (Vec<_>, Vec<_>) = self
                .leaf_points
                .select(Axis(0), &curr_leaf_points_idx)
                .axis_iter(Axis(0))
                .enumerate()
                .filter_map(|(i, row)| {
                    if row == leaf {
                        Some((self.residuals[i], self.x[(i, col)] < split))
                    } else {
                        None
                    }
                })
                .partition(|(_, is_a)| *is_a);

            n_a += v.powf(2.0) * resids_a.len() as f32;
            n_b += v.powf(2.0) * resids_b.len() as f32;
            m_a += v * resids_a.into_iter().map(|(r, _)| r).sum::<f32>();
            m_b += v * resids_b.into_iter().map(|(r, _)| r).sum::<f32>();
        }

        let update_a = if n_a == 0.0 { 1.0 } else { m_a / n_a + 1.0 };
        let update_b = if n_b == 0.0 { 1.0 } else { m_b / n_b + 1.0 };

        let err_old =
            self.residuals.iter().map(|x| x.powi(2)).sum::<f32>() / self.residuals.len() as f32;

        let (a_points_idx, b_points_idx): (Vec<_>, Vec<_>) = curr_points
            .index_axis(Axis(1), col)
            .indexed_iter()
            .map(|(i, &x)| {
                if x <= split {
                    (curr_leaf_points_idx[i], true)
                } else {
                    (curr_leaf_points_idx[i], false)
                }
            })
            .partition(|(_, is_a)| *is_a);

        let new_resid_a = a_points_idx
            .iter()
            .map(|(i, _)| (self.labels[*i] - self.y_hat[*i] * update_a).powf(2.0))
            .sum::<f32>();

        let new_resid_b = b_points_idx
            .iter()
            .map(|(i, _)| (self.labels[*i] - self.y_hat[*i] * update_b).powf(2.0))
            .sum::<f32>();

        let err_new = new_resid_a + new_resid_b;

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

        return (err_new, err_old, refine_candidate);
    }

    pub fn slice_and_refine_candidate(
        &self,
        col: usize,
        split: f32,
    ) -> (f32, f32, RefineCandidate) {
        let slice_candidate = self.slice_candidate(col, split);
        self.refine_candidate(slice_candidate)
    }

    fn update_leaf_points(&mut self, dim: &usize, index: &usize, leaf_points_b: &[usize]) {
        for mut x in self.leaf_points.axis_iter_mut(Axis(0)) {
            if x[*dim] > *index {
                x[*dim] += 1;
            }
        }
        for &i in leaf_points_b {
            self.leaf_points[(i, *dim)] += 1;
        }
    }

    fn update_tree(&mut self, refine_candidate: RefineCandidate) {
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

        self.grid_values[col][index] = update_a;
        self.grid_values[col].insert(index + 1, update_b);

        self.splits[col].insert(index, split);
        self.intervals[col][index] = left;
        self.intervals[col].insert(index + 1, right);

        self.update_leaf_points(&col, &index, &b_points_idx);

        for &i in &a_points_idx {
            self.y_hat[i] *= update_a;
            self.residuals[i] = self.labels[i] - self.y_hat[i];
        }
        for &i in &b_points_idx {
            self.y_hat[i] *= update_b;
            self.residuals[i] = self.labels[i] - self.y_hat[i];
        }
    }

    pub fn slice_and_refine(&mut self, col: usize, split: f32) {
        let (_, _, refine_candidate) = self.slice_and_refine_candidate(col, split);
        self.update_tree(refine_candidate);
    }

    pub fn predict(&self, x: &Array2<f32>) -> Array1<f32> {
        let mut y_hat = Array1::zeros(x.nrows());
        for (i, row) in x.axis_iter(Axis(0)).enumerate() {
            let mut prod = 1.0;
            for (j, &val) in row.iter().enumerate() {
                let index = self.splits[j]
                    .iter()
                    .position(|&x| x >= val)
                    .unwrap_or(self.splits[j].len());
                prod *= self.grid_values[j][index];
            }
            y_hat[i] = prod;
        }
        y_hat
    }
}
