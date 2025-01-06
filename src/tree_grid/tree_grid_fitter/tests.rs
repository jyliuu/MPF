use ndarray::{Array1, Array2};

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
