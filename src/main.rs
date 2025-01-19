use csv::ReaderBuilder;
use mpf::{
    forest::forest_fitter::{MPFFitter, MPFParams},
    ModelFitter,
};
use ndarray::{Array1, Array2};

fn setup_data() -> (Array2<f64>, Array1<f64>) {
    let mut rdr = ReaderBuilder::new()
        .has_headers(true)
        .from_path("./dat.csv")
        .expect("Failed to open file");

    let mut x_data = Vec::new();
    let mut y_data = Vec::new();

    for result in rdr.records() {
        let record = result.expect("Failed to read record");
        let y: f64 = record[0].parse().expect("Failed to parse y");
        let x1: f64 = record[1].parse().expect("Failed to parse x1");
        let x2: f64 = record[2].parse().expect("Failed to parse x2");

        y_data.push(y);
        x_data.push(vec![x1, x2]);
    }

    let x = Array2::from_shape_vec((x_data.len(), 2), x_data.into_iter().flatten().collect())
        .expect("Failed to create Array2");
    let y = Array1::from(y_data);

    (x, y)
}

fn main() {
    println!("Running main");
    let (x, y) = setup_data();
    let mpf_fitter = MPFFitter::new(x.view(), y.view());
    let (fit_result, mpf) = mpf_fitter.fit_grown(MPFParams {
        n_families: 100,
        n_iter: 100,
        m_try: 1.0,
        split_try: 10,
    });

    let mean = y.mean().unwrap();
    let base_err = (y - mean).powi(2).mean().unwrap();
    println!("Base error: {:?}, Error: {:?}", base_err, fit_result.err);
    assert!(
        fit_result.err < base_err,
        "Error is not less than mean error"
    );
}
