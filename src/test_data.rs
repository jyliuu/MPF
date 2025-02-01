use ndarray::{Array1, Array2};
use csv::ReaderBuilder;

pub fn setup_data_csv() -> (Array2<f64>, Array1<f64>) {
    // Reads data from file "dat.csv"
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
    let x = Array2::from_shape_vec(

        (x_data.len(), 2),
        x_data.into_iter().flatten().collect(),
    )
    .expect("Failed to create Array2");
    let y = Array1::from(y_data);
    (x, y)
}

pub fn setup_data_hardcoded() -> (Array2<f64>, Array1<f64>) {
    // Returns hardcoded test data
    let dat = Array2::from_shape_vec(
        (20, 3),
        vec![
            1.99591859675606,
            -1.00591398212174,
            -1.47348548786449,
            -0.290104994456985,
            -0.507165037206491,
            -0.0992586787398358,
            -0.392657438987142,
            1.41894909677495,
            -0.674415207533763,
            0.541774508623095,
            0.134065164928921,
            0.634093564547107,
            0.981026718908818,
            0.29864258176132,
            1.29321982986182,
            2.14226826821187,
            -1.57541477899575,
            -1.20864031274097,
            0.614259969810645,
            -1.11273947093321,
            -0.747582520955759,
            0.742939152591961,
            0.367035148375779,
            0.629260294753607,
            -2.90764791321527,
            1.81674051159666,
            -1.27652692983198,
            -1.94290907058012,
            2.5208012003232,
            -0.871450106365531,
            0.272189306719476,
            1.01227462627796,
            -0.356579330585395,
            0.481004283284028,
            0.165976176377298,
            0.822063375479486,
            -0.245353149162764,
            -1.40974327898294,
            -0.334709204672301,
            -0.00460477602051997,
            0.0117210317817887,
            2.69171682068671,
            0.359824874802531,
            0.821234081234943,
            -0.318909828758849,
            -1.88722434288848,
            -1.01377986818573,
            0.400700584291665,
            -0.141615483262696,
            0.128123583066683,
            -1.59321040126916,
            0.136218360404787,
            0.112778041636902,
            0.0942204942429378,
            2.20921149756541,
            0.882698443188986,
            0.852817759799762,
            -2.73007802370526,
            -1.21615404372871,
            0.633442434384728,
        ],
    )
    .unwrap();
    let y = dat.slice(ndarray::s![.., 0]).to_owned();
    let x = dat.slice(ndarray::s![.., 1..]).to_owned();
    (x, y)
}
