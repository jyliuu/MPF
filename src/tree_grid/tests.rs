use super::*;

fn setup_data() -> (Array2<f32>, Array1<f32>) {
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
    let tree_grid = TreeGrid::new(&x, &y);
    let (_, _, refine_candidate) = tree_grid.slice_and_refine_candidate(0, 1.0);
    assert_eq!(refine_candidate.update_a, 0.5);
    assert_eq!(refine_candidate.update_b, 1.5);
}

#[test]
fn test_tree_grid_slice_and_refine() {
    let (x, y) = setup_data();
    let mut tree_grid = TreeGrid::new(&x, &y);
    tree_grid.slice_and_refine(0, 1.0);
    assert_eq!(tree_grid.grid_values[0], vec![0.5, 1.5]);
    assert_eq!(tree_grid.splits[0], vec![1.0]);
    assert_eq!(
        tree_grid.intervals[0],
        vec![(f32::NEG_INFINITY, 1.0), (1.0, f32::INFINITY)]
    );
    assert_eq!(tree_grid.residuals.sum(), 0.0);
}

#[test]
fn test_tree_grid_fit_and_predict() {
    let (x, y) = setup_data();
    let mut tree_grid = TreeGrid::new(&x, &y);
    tree_grid.slice_and_refine(0, 1.0);
    tree_grid.slice_and_refine(1, 1.0);

    let preds_on_x = tree_grid.predict(&x);
    let preds_extrapolated = unsafe {
        tree_grid.predict(&Array2::from_shape_vec_unchecked(
            (2, 2),
            vec![-1.0, -1.0, 2.0, 2.0],
        ))
    };

    assert_eq!(preds_on_x, tree_grid.y_hat);
    assert_eq!(preds_extrapolated, Array1::from_vec(vec![0.5, 1.5]));
}
