use csv::ReaderBuilder;
use ndarray::{Array1, Array2};

#[allow(dead_code)]
pub fn setup_data_housing_csv(full: bool) -> (Array2<f64>, Array1<f64>) {
    // Reads data from file "data/housing.csv" and makes median_house_value (last column) the response (y)
    let mut rdr = ReaderBuilder::new()
        .has_headers(false)
        .from_path(if full {
            "./data/housing_full.csv"
        } else {
            "./data/housing.csv"
        })
        .expect("Failed to open file");

    let mut x_data_: Vec<Vec<f64>> = Vec::new();
    let mut y_data = Vec::new();

    // Skip the header row since we're using has_headers(true)
    for result in rdr.records() {
        let record = result.expect("Failed to read record");
        let mut x_row = Vec::new();

        // Parse all columns except the last one as features
        for i in 0..record.len() - 1 {
            let val = record[i].parse::<f64>().unwrap();
            x_row.push(val);
        }
        // Parse the last column as the target
        let y_val = record[record.len() - 1].parse::<f64>().unwrap();
        x_data_.push(x_row);
        y_data.push(y_val);
    }

    let num_features = if let Some(first_row) = x_data_.first() {
        first_row.len()
    } else {
        0
    };

    let x_data: Vec<f64> = x_data_.into_iter().flatten().collect();
    let x = Array2::from_shape_vec((x_data.len() / num_features, num_features), x_data)
        .expect("Failed to create Array2 for x");
    let y = Array1::from(y_data);
    (x, y)
}

pub fn setup_data_csv() -> (Array2<f64>, Array1<f64>) {
    // Reads data from file "dat.csv"
    let mut rdr = ReaderBuilder::new()
        .has_headers(true)
        .from_path("./data/dat.csv")
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
