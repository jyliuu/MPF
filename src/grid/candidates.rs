use itertools::Itertools;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};

#[derive(Debug)]
pub struct SliceCandidate {
    pub col: usize,
    pub split: f64,
    pub index: usize,
    pub left: (f64, f64),
    pub right: (f64, f64),
}

#[derive(Debug)]
pub struct RefineCandidate {
    pub col: usize,
    pub split: f64,
    pub index: usize,
    pub left: (f64, f64),
    pub right: (f64, f64),
    pub update_a: f64,
    pub update_b: f64,
    pub a_points_idx: Vec<usize>,
    pub b_points_idx: Vec<usize>,
    pub curr_leaf_points_idx: Vec<usize>,
}

pub fn compute_leaf_values(leaf: &Array1<usize>, grid_values: &[Vec<f64>]) -> f64 {
    leaf.indexed_iter()
        .map(|(i, &idx)| grid_values[i][idx])
        .product()
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

pub fn find_refine_candidate(
    slice_candidate: SliceCandidate,
    x: ArrayView2<f64>,
    leaf_points: &Array2<usize>,
    grid_values: &[Vec<f64>],
    intervals: &[Vec<(f64, f64)>],
    residuals: ArrayView1<'_, f64>,
    y_hat: ArrayView1<'_, f64>,
) -> Result<(f64, f64, RefineCandidate), String> {
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
    let curr_leaf_leaf_points = leaf_points.select(Axis(0), &curr_leaf_points_idx);

    for leaf in &leaves {
        let v = compute_leaf_values(leaf, grid_values);
        let v_pow2 = v.powi(2);
        for (idx, row) in curr_leaf_leaf_points.axis_iter(Axis(0)).enumerate() {
            if row != leaf {
                continue;
            }

            let idx = curr_leaf_points_idx[idx];
            if x[(idx, col)] < split {
                n_a += v_pow2;
                m_a += v * residuals[idx];
            } else {
                n_b += v_pow2;
                m_b += v * residuals[idx];
            }
        }
    }

    let update_a = if n_a == 0.0 { 1.0 } else { m_a / n_a + 1.0 };
    let update_b = if n_b == 0.0 { 1.0 } else { m_b / n_b + 1.0 };

    let err_old = residuals
        .select(Axis(0), &curr_leaf_points_idx)
        .pow2()
        .sum();

    let curr_points = x.select(Axis(0), &curr_leaf_points_idx);

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

    if a_points_idx.is_empty() || b_points_idx.is_empty() {
        return Err("No points to update".to_string());
    }
    let new_resid_a = a_points_idx
        .iter()
        .map(|(i, _)| y_hat[*i].mul_add(1.0 - update_a, residuals[*i]))
        .collect::<Array1<f64>>();
    let new_resid_b = b_points_idx
        .iter()
        .map(|(i, _)| y_hat[*i].mul_add(1.0 - update_b, residuals[*i]))
        .collect::<Array1<f64>>();

    let err_a = new_resid_a.powi(2).sum();
    let err_b = new_resid_b.powi(2).sum();
    let err_new = err_a + err_b;

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

    Ok((err_new, err_old, refine_candidate))
}

pub fn update_leaf_points(
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


pub fn update_predictions(
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
