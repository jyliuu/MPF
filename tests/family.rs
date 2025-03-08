mod test_data;

#[cfg(test)]
mod tests {
    use super::test_data::setup_data_csv;
    use mpf::{
        family::{fit, params::TreeGridFamilyBoostedParams},
        grid::{
            params::TreeGridParams,
            strategies::{
                combine_into_single_tree_grid, L2ArithmeticGeometricMean, L2ArithmeticMean,
                L2GeometricMean, L2Median,
            },
        },
        FittedModel,
    };
    use rand::{rngs::StdRng, SeedableRng};
    #[test]
    fn test_tgf_boosted_fit() {
        let (x, y) = setup_data_csv();
        let mut rng = StdRng::seed_from_u64(42);
        let hyperparameters = TreeGridFamilyBoostedParams::default();
        let (fit_result, _) = fit(x.view(), y.view(), &hyperparameters, &mut rng);
        let mean = y.mean().unwrap();
        let base_err = (y - mean).powi(2).mean().unwrap();
        println!("Base error: {:?}, Error: {:?}", base_err, fit_result.err);
        assert!(
            fit_result.err < base_err,
            "Error is not less than mean error"
        );
    }

    #[test]
    fn test_l2_median_combined_tree_grid_predicts_well() {
        let (x, y) = setup_data_csv();
        let mut rng = StdRng::seed_from_u64(42);
        let (_, tgf) = fit(
            x.view(),
            y.view(),
            &TreeGridFamilyBoostedParams {
                n_trees: 20,
                bootstrap: false,
                tg_params: TreeGridParams::default(),
            },
            &mut rng,
        );
        let tree_grids = tgf.get_tree_grids();
        let reference = &tree_grids[0];
        let combined_tree_grid =
            combine_into_single_tree_grid(tree_grids, reference, &L2Median, x.view());
        let pred = combined_tree_grid.predict(x.view());
        let err = (y - pred).powi(2).mean().unwrap();
        println!("err: {:?}", err);
        assert!(err < 0.1);
    }

    #[test]
    fn test_l2_arith_geom_mean_combined_tree_grid_predicts_well() {
        let (x, y) = setup_data_csv();
        let mut rng = StdRng::seed_from_u64(42);
        let (_, tgf) = fit(
            x.view(),
            y.view(),
            &TreeGridFamilyBoostedParams {
                n_trees: 20,
                bootstrap: false,
                tg_params: TreeGridParams::default(),
            },
            &mut rng,
        );
        let tree_grids = tgf.get_tree_grids();
        let reference = &tree_grids[0];
        let combined_tree_grid = combine_into_single_tree_grid(
            tree_grids,
            reference,
            &L2ArithmeticGeometricMean,
            x.view(),
        );
        let pred = combined_tree_grid.predict(x.view());
        let err = (y - pred).powi(2).mean().unwrap();
        println!("err: {:?}", err);
        assert!(err < 0.1);
    }

    #[test]
    fn test_l2_arith_mean_combined_tree_grid_predicts_well() {
        let (x, y) = setup_data_csv();
        let mut rng = StdRng::seed_from_u64(42);
        let (_, tgf) = fit(
            x.view(),
            y.view(),
            &TreeGridFamilyBoostedParams {
                n_trees: 20,
                bootstrap: false,
                tg_params: TreeGridParams::default(),
            },
            &mut rng,
        );
        let tree_grids = tgf.get_tree_grids();
        let reference = &tree_grids[0];
        let combined_tree_grid =
            combine_into_single_tree_grid(tree_grids, reference, &L2ArithmeticMean, x.view());
        let pred = combined_tree_grid.predict(x.view());
        let err = (y - pred).powi(2).mean().unwrap();
        println!("err: {:?}", err);
        assert!(err < 0.1);
    }

    #[test]
    fn test_l2_geom_mean_combined_tree_grid_predicts_well() {
        let (x, y) = setup_data_csv();
        let mut rng = StdRng::seed_from_u64(42);
        let (_, tgf) = fit(
            x.view(),
            y.view(),
            &TreeGridFamilyBoostedParams {
                n_trees: 20,
                bootstrap: false,
                tg_params: TreeGridParams::default(),
            },
            &mut rng,
        );
        let tree_grids = tgf.get_tree_grids();
        let reference = &tree_grids[0];
        let combined_tree_grid =
            combine_into_single_tree_grid(tree_grids, reference, &L2GeometricMean, x.view());
        let pred = combined_tree_grid.predict(x.view());
        let err = (y - pred).powi(2).mean().unwrap();
        println!("err: {:?}", err);
        assert!(err < 0.1);
    }
}
