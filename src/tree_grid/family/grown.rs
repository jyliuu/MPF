use itertools::Itertools;
use ndarray::{Array1, ArrayView1, ArrayView2};
use rand::seq::SliceRandom;
use rand::Rng;
use std::collections::{BTreeSet, HashMap};

use crate::{
    tree_grid::grid::fitter::{
        find_refine_candidate, find_slice_candidate, RefineCandidate, TreeGridFitter,
    },
    FitResult, FittedModel, ModelFitter,
};

use super::{Aggregation, AggregationMethod, TreeGridFamily};

pub fn fit(
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
    hyperparameters: &TreeGridFamilyGrownParams,
) -> (FitResult, TreeGridFamily<GrownVariant>) {
    let dims = x.shape()[1];
    let intercept_grid = TreeGridFitter::new(x.view(), y.view());
    let mut tg_fitters = HashMap::new();
    let y_hat = intercept_grid.y_hat.clone();
    let residuals = y.to_owned() - &y_hat;
    tg_fitters.insert(BTreeSet::new(), vec![intercept_grid]);

    let mut fitter = TreeGridFamilyGrownFitter {
        dims,
        x: x.view(),
        y: y.view(),
        tg_fitters,
        y_hat,
        residuals,
    };

    let TreeGridFamilyGrownParams {
        n_iter,
        m_try,
        split_try,
    } = *hyperparameters;
    let mut rng = rand::thread_rng();

    for _ in 0..n_iter {
        let potential_splits = fitter.potential_splits();
        let n_potential_splits = potential_splits.len();
        let n_splits_to_try = (n_potential_splits as f64 * m_try).ceil() as usize;

        let mut candidates = Vec::new();
        let split_indices: Vec<usize> = (0..n_potential_splits).collect();
        let selected_splits: Vec<usize> = split_indices
            .choose_multiple(&mut rng, n_splits_to_try)
            .copied()
            .collect();

        for &p_idx in &selected_splits {
            let (s, dim, sample_tg_idx) = potential_splits[p_idx].clone();

            let mut best_split = None;
            let mut best_err_diff = f64::NEG_INFINITY;

            for _ in 0..split_try {
                let split_idx = rng.gen_range(0..x.nrows());
                let split_val = x[[split_idx, dim]];

                if let Some(tg_fitters) = fitter.tg_fitters.get(&s) {
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
                            fitter.residuals.view(),
                            fitter.y_hat.view(),
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
            fitter.update_estimator(s, sample_tg_idx, refine_candidate);
        }
    }

    let fit_result = FitResult {
        err: fitter.loss(),
        residuals: fitter.residuals,
        y_hat: fitter.y_hat,
    };

    let fitted_tree_grids = fitter
        .tg_fitters
        .into_values()
        .flatten()
        .map_into()
        .collect();

    (fit_result, TreeGridFamily(fitted_tree_grids, GrownVariant))
}

#[derive(Debug)]
pub struct GrownVariant;

impl AggregationMethod for GrownVariant {
    const AGGREGATION_METHOD: Aggregation = Aggregation::Average;
}

impl FittedModel for TreeGridFamily<GrownVariant> {
    fn predict(&self, x: ArrayView2<f64>) -> Array1<f64> {
        let mut result = Array1::zeros(x.shape()[0]);
        for grid in self.0.iter() {
            result += &grid.predict(x);
        }
        result
    }
}

#[derive(Debug)]
pub struct TreeGridFamilyGrownParams {
    pub n_iter: usize,
    pub m_try: f64,
    pub split_try: usize,
}

impl Default for TreeGridFamilyGrownParams {
    fn default() -> Self {
        Self {
            n_iter: 100,
            m_try: 1.0,
            split_try: 10,
        }
    }
}

struct TreeGridFamilyGrownFitter<'a> {
    dims: usize,
    x: ArrayView2<'a, f64>,
    y: ArrayView1<'a, f64>,
    tg_fitters: HashMap<BTreeSet<usize>, Vec<TreeGridFitter<'a>>>,
    y_hat: Array1<f64>,
    residuals: Array1<f64>,
}

impl TreeGridFamilyGrownFitter<'_> {
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
}
