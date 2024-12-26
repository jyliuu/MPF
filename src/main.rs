use csv::ReaderBuilder;
use mpf_rust::tree_grid::model;
use ndarray::{Array1, Array2};

fn setup_data() -> (Array2<f32>, Array1<f32>) {
    let mut rdr = ReaderBuilder::new()
        .has_headers(true)
        .from_path("src/tree_grid/model/dat.csv")
        .expect("Failed to open file");

    let mut x_data = Vec::new();
    let mut y_data = Vec::new();

    for result in rdr.records() {
        let record = result.expect("Failed to read record");
        let y: f32 = record[0].parse().expect("Failed to parse y");
        let x1: f32 = record[1].parse().expect("Failed to parse x1");
        let x2: f32 = record[2].parse().expect("Failed to parse x2");

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
    let mut tree_grid = model::TreeGrid::new(model::TreeGridParams {
        n_iter: 50,
        split_try: 10,
        colsample_bytree: 1.0,
    });
    let fit_result = tree_grid.fit(&x, &y);

    println!("Error: {:?}", fit_result.err);
}
