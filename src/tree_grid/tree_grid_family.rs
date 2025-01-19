use itertools::Itertools;
use ndarray::{Array1, ArrayView1, ArrayView2};
use rand::seq::SliceRandom;
use rand::Rng;
use std::collections::{BTreeSet, HashMap};

use crate::{
    tree_grid::tree_grid_fitter::{
        find_refine_candidate, find_slice_candidate, RefineCandidate, TreeGridFitter,
    },
    FitResult, FittedModel, ModelFitter,
};

use super::tree_grid::FittedTreeGrid;

#[derive(Debug)]
pub struct TreeGridFamily {
    tree_grids: HashMap<BTreeSet<usize>, Vec<FittedTreeGrid>>,
}

impl TreeGridFamily {
    pub fn new(tree_grids: HashMap<BTreeSet<usize>, Vec<FittedTreeGrid>>) -> Self {
        Self { tree_grids }
    }
}

impl FittedModel for TreeGridFamily {
    fn predict(&self, x: ArrayView2<f64>) -> Array1<f64> {
        let mut result = Array1::zeros(x.shape()[0]);
        for grids in self.tree_grids.values() {
            for grid in grids {
                result += &grid.predict(x);
            }
        }
        result
    }
}

pub struct TreeGridFamilyFitter<'a> {
    pub dims: usize,
    pub x: ArrayView2<'a, f64>,
    pub y: ArrayView1<'a, f64>,
    pub tg_fitters: HashMap<BTreeSet<usize>, Vec<TreeGridFitter<'a>>>,
    pub y_hat: Array1<f64>,
    pub residuals: Array1<f64>,
}

impl<'a> TreeGridFamilyFitter<'a> {
    pub fn new(x: ArrayView2<'a, f64>, y: ArrayView1<'a, f64>) -> Self {
        let dims = x.shape()[1];
        let intercept_grid = TreeGridFitter::new(x, y);
        let mut tg_fitters = HashMap::new();
        let y_hat = intercept_grid.y_hat.clone();
        tg_fitters.insert(BTreeSet::new(), vec![intercept_grid]);

        let residuals = y.to_owned() - &y_hat;

        Self {
            dims,
            x,
            y,
            tg_fitters,
            y_hat,
            residuals,
        }
    }

    fn update_y_hat(&mut self) {
        self.y_hat.fill(0.0);
        for grids in self.tg_fitters.values() {
            for grid in grids {
                self.y_hat += &grid.y_hat;
            }
        }
    }

    fn update_residuals(&mut self) {
        self.update_y_hat();
        self.residuals = self.y.to_owned() - &self.y_hat;
    }

    pub fn loss(&self) -> f64 {
        self.residuals.pow2().mean().unwrap()
    }

    pub fn potential_splits(&self) -> Vec<(BTreeSet<usize>, usize, usize)> {
        let mut possible_tree_structure_after_split: HashMap<
            BTreeSet<usize>,
            Vec<(BTreeSet<usize>, usize)>,
        > = HashMap::new();
        let mut rng = rand::thread_rng();

        for (s, j) in self.tg_fitters.keys().cartesian_product(0..self.dims) {
            let mut s_union_j = s.clone();
            s_union_j.insert(j);

            if !self.tg_fitters[s].is_empty()
                && (!self.tg_fitters.contains_key(&s_union_j) || s.contains(&j))
            {
                possible_tree_structure_after_split
                    .entry(s_union_j)
                    .or_default()
                    .push((s.clone(), j));
            }
        }

        let mut potential_splits = Vec::new();
        for (s_union_j, sjs) in possible_tree_structure_after_split {
            if let Some(&(ref s, j)) = sjs.choose(&mut rng) {
                let tree_grid_idx = rng.gen_range(0..self.tg_fitters[s].len());
                potential_splits.push((s.clone(), j, tree_grid_idx));
            }
        }

        potential_splits
    }

    pub fn update_estimator(
        &mut self,
        s: BTreeSet<usize>,
        sample_tg_idx: usize,
        refine_candidate: RefineCandidate,
    ) {
        let dim = refine_candidate.col;
        if let Some(tg_fitters) = self.tg_fitters.get_mut(&s) {
            if let Some(sample_tg) = tg_fitters.get_mut(sample_tg_idx) {
                sample_tg.update_tree(refine_candidate);
            }
        }

        if !s.contains(&dim) {
            if let Some(tg_fitters) = self.tg_fitters.get_mut(&s) {
                let mut new_s = s.clone();
                new_s.insert(dim);

                let sample_tg = tg_fitters.remove(sample_tg_idx);
                self.tg_fitters.entry(new_s).or_default().push(sample_tg);
            }

            if s.is_empty() {
                let new_tg_fitter = TreeGridFitter::new(self.x, self.y);

                self.tg_fitters
                    .entry(BTreeSet::new())
                    .or_default()
                    .push(new_tg_fitter);
            }
        }

        self.update_residuals();
    }

    pub fn fit(
        mut self,
        n_iter: usize,
        m_try: f64,
        split_try: usize,
    ) -> (FitResult, TreeGridFamily) {
        let mut rng = rand::thread_rng();

        for _ in 0..n_iter {
            let potential_splits = self.potential_splits();
            let n_potential_splits = potential_splits.len();
            let n_splits_to_try = (n_potential_splits as f64 * m_try).ceil() as usize;

            let mut candidates = Vec::new();
            let split_indices: Vec<usize> = (0..n_potential_splits).collect();
            let selected_splits: Vec<usize> = split_indices
                .choose_multiple(&mut rng, n_splits_to_try)
                .cloned()
                .collect();

            for &p_idx in &selected_splits {
                let (s, dim, sample_tg_idx) = potential_splits[p_idx].clone();

                let mut best_split = None;
                let mut best_err_diff = f64::NEG_INFINITY;

                for _ in 0..split_try {
                    let split_idx = rng.gen_range(0..self.x.nrows());
                    let split_val = self.x[[split_idx, dim]];

                    if let Some(tg_fitters) = self.tg_fitters.get(&s) {
                        if let Some(sample_tg) = tg_fitters.get(sample_tg_idx) {
                            let slice_candidate = find_slice_candidate(
                                &sample_tg.splits,
                                &sample_tg.intervals,
                                dim,
                                split_val,
                            );
                            let (err_new, err_old, refine_candidate) = find_refine_candidate(
                                slice_candidate,
                                sample_tg.x,
                                &sample_tg.leaf_points,
                                &sample_tg.grid_values,
                                &sample_tg.intervals,
                                self.residuals.view(),
                                self.y_hat.view(),
                            );
                            let err_diff = err_old - err_new;

                            if err_diff > best_err_diff {
                                best_err_diff = err_diff;
                                best_split = Some((s.clone(), sample_tg_idx, refine_candidate));
                            }
                        }
                    }
                }

                if let Some(split) = best_split {
                    candidates.push((best_err_diff, split));
                }
            }

            if let Some((_, (s, sample_tg_idx, refine_candidate))) =
                candidates.into_iter().max_by(|a, b| {
                    let a_diff = a.0;
                    let b_diff = b.0;
                    a_diff.partial_cmp(&b_diff).unwrap()
                })
            {
                self.update_estimator(s, sample_tg_idx, refine_candidate);
            }
        }

        (
            FitResult {
                err: self.loss(),
                residuals: self.residuals,
                y_hat: self.y_hat,
            },
            TreeGridFamily::new(
                self.tg_fitters
                    .into_iter()
                    .map(|(s, tg_fitters)| (s, tg_fitters.into_iter().map_into().collect()))
                    .collect(),
            ),
        )
    }
}

#[cfg(test)]
mod tests {
    use csv::ReaderBuilder;
    use ndarray::Array2;

    use crate::forest::forest_fitter::MPFFitter;

    use super::*;
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

    #[test]
    fn test_tgf_fit() {
        let (x, y) = setup_data();
        let tgf_fitter = TreeGridFamilyFitter::new(x.view(), y.view());
        let (fit_result, _) = tgf_fitter.fit(100, 1.0, 10);
        let mean = y.mean().unwrap();
        let base_err = (y - mean).powi(2).mean().unwrap();
        println!("Base error: {:?}, Error: {:?}", base_err, fit_result.err);
        assert!(
            fit_result.err < base_err,
            "Error is not less than mean error"
        );
    }

    #[test]
    fn test_tgf_predict() {
        let (x, y) = setup_data();
        let tgf_fitter = TreeGridFamilyFitter::new(x.view(), y.view());
        let (fit_result, tgf) = tgf_fitter.fit(100, 1.0, 10);

        let pred = tgf.predict(x.view());
        let diff = fit_result.y_hat - pred;
        assert!(diff.iter().all(|&x| x < 1e-6));
    }

    #[test]
    fn test_mpf_fit() {
        let (x, y) = setup_data();
        let mpf_fitter = MPFFitter::new(x.view(), y.view());
        let (fit_result, _) = mpf_fitter.fit(100, 100, 1.0, 10);

        let mean = y.mean().unwrap();
        let base_err = (y - mean).powi(2).mean().unwrap();
        println!("Base error: {:?}, Error: {:?}", base_err, fit_result.err);
        assert!(
            fit_result.err < base_err,
            "Error is not less than mean error"
        );
    }

    #[test]
    fn test_mpf_predict() {
        let (x, y) = setup_data();
        let mpf_fitter = MPFFitter::new(x.view(), y.view());
        let (fit_result, mpf) = mpf_fitter.fit(100, 100, 1.0, 10);
        let pred = mpf.predict(x.view());
        let diff = fit_result.y_hat - pred;
        assert!(diff.iter().all(|&x| x < 1e-6));
    }
}
