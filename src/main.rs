use std::{ops::Div, time::SystemTime};

use mpf::{
    forest::{
        forest_fitter::{fit_bagged, MPFBaggedParams},
        mpf::MPF,
    }, test_data::setup_data_csv, tree_grid::{
        family::{
            bagged::{BaggedVariant, TreeGridFamilyBaggedParams},
            TreeGridFamily,
        },
        grid::{
            fitter::{TreeGridFitter, TreeGridParams},
            FittedTreeGrid,
        },
    }, FitResult, FittedModel, ModelFitter
};
use ndarray::{s, ArrayView1, ArrayView2};

fn fit_bagged_mpf(
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
) -> (FitResult, MPF<TreeGridFamily<BaggedVariant>>) {
    let (fit_result, mpf) = fit_bagged(
        x.view(),
        y.view(),
        MPFBaggedParams {
            epochs: 5,
            tgf_params: TreeGridFamilyBaggedParams {
                B: 100,
                tg_params: TreeGridParams {
                    n_iter: 100,
                    split_try: 10,
                    colsample_bytree: 1.0,
                },
            },
        },
    );

    (fit_result, mpf)
}

fn fit_tree_grid(x: ArrayView2<f64>, y: ArrayView1<f64>) -> (FitResult, FittedTreeGrid) {
    let tgfitter = TreeGridFitter::new(x.view(), y.view());
    let (fit_result, tgf) = tgfitter.fit(&TreeGridParams {
        n_iter: 100,
        split_try: 10,
        colsample_bytree: 1.0,
    });

    (fit_result, tgf)
}

fn main() {
    println!("Running main");
    let (x, y) = setup_data_csv();
    let n = y.len();
    println!("Fitting and testing on {} samples", n / 2);
    let x_train = x.slice(s![..n / 2, ..]);
    let y_train = y.slice(s![..n / 2]);

    let x_test = x.slice(s![n / 2.., ..]);
    let y_test = y.slice(s![n / 2..]);

    let start = SystemTime::now();
    let (fr, model) = fit_tree_grid(x_train.view(), y_train.view());
    let elapsed = start.elapsed().unwrap();
    println!("Time elapsed: {:?}", elapsed);

    let mean = y_test.mean().unwrap();
    let base_err = y_test.view().map(|v| v - mean).powi(2).mean().unwrap();
    let preds = model.predict(x_test.view());
    let test_err: f64 = y_test
        .indexed_iter()
        .map(|(i, v)| (v - preds[i]).powi(2).div(y_test.len() as f64))
        .sum();
    println! {"Base error: {:?}, Training Error: {:?}, Test Error: {:?}", base_err, fr.err, test_err};
}
