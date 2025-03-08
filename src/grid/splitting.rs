use ndarray::ArrayView2;
use rand::{seq::index::sample, Rng};

pub enum SplitStrategy {
    Random(RandomSplit),
    Interval(IntervalRandomSplit),
}

impl SplitStrategy {
    pub fn sample_splits<R: Rng + ?Sized>(
        &self,
        rng: &mut R,
        x: ArrayView2<f64>,
        intervals: &[Vec<(f64, f64)>],
    ) -> Vec<(usize, f64)> {
        match self {
            SplitStrategy::Random(random_split) => random_split.sample_splits(rng, x),
            SplitStrategy::Interval(interval_split) => {
                interval_split.sample_splits(rng, x, intervals)
            }
        }
    }
}
#[derive(Debug, Clone)]
pub struct RandomSplit {
    pub split_try: usize,
    pub colsample_bytree: f64,
}

#[derive(Debug, Clone)]
pub struct IntervalRandomSplit {
    pub split_try: usize,
    pub colsample_bytree: f64,
}

impl RandomSplit {
    fn sample_splits<R: Rng + ?Sized>(&self, rng: &mut R, x: ArrayView2<f64>) -> Vec<(usize, f64)> {
        let nrows = x.nrows();
        let ncols = x.ncols();
        let ncols_to_sample = (self.colsample_bytree * ncols as f64) as usize;

        let mut splits = vec![];

        for col in sample(rng, ncols, ncols_to_sample) {
            for idx in sample(rng, nrows, self.split_try) {
                splits.push((col, x[[idx, col]]));
            }
        }

        splits
    }
}

impl IntervalRandomSplit {
    fn sample_splits<R: Rng + ?Sized>(
        &self,
        rng: &mut R,
        x: ArrayView2<f64>,
        intervals: &[Vec<(f64, f64)>],
    ) -> Vec<(usize, f64)> {
        let ncols = x.ncols();
        let ncols_to_sample = (self.colsample_bytree * ncols as f64) as usize;

        let cols = sample(rng, ncols, ncols_to_sample);

        let mut splits = vec![];
        for col in cols {
            let intervals = &intervals[col];
            for (a, b) in intervals {
                let a = if a.is_infinite() {
                    *x.column(col)
                        .iter()
                        .min_by(|a, b| a.partial_cmp(b).unwrap())
                        .unwrap()
                } else {
                    *a
                };
                let b = if b.is_infinite() {
                    *x.column(col)
                        .iter()
                        .max_by(|a, b| a.partial_cmp(b).unwrap())
                        .unwrap()
                } else {
                    *b
                };

                for _ in 0..self.split_try {
                    splits.push((col, rng.gen_range(a..b)));
                }
            }
        }
        splits
    }
}
