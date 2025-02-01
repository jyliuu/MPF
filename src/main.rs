use std::{ops::Div, time::SystemTime};

use mpf::{
    forest::forest_fitter::{fit_bagged, MPFBaggedParams},
    FittedModel,
};
use ndarray::s;
mod test_data;

fn main() {
    println!("Running main");
    let (x, y) = test_data::setup_data_csv();
    let n = y.len();
    println!("Fitting and testing on {} samples", n / 2);
    let x_train = x.slice(s![..n / 2, ..]);
    let y_train = y.slice(s![..n / 2]);

    let x_test = x.slice(s![n / 2.., ..]);
    let y_test = y.slice(s![n / 2..]);

    let start = SystemTime::now();
    let (fr, model) = fit_bagged(x_train.view(), y_train.view(), MPFBaggedParams::default());
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
