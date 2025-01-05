use itertools::Itertools;
use ndarray::{Array1, ArrayView1, ArrayView2};
use rand::seq::SliceRandom;
use rand::Rng;
use std::collections::{BTreeSet, HashMap};

use crate::tree_grid::model::{TreeGrid, TreeGridParams};

pub struct ForestFitter<'a> {
    pub dims: usize,
    pub points: ArrayView2<'a, f64>,
    pub labels: ArrayView1<'a, f64>,
    pub tree_grids: HashMap<BTreeSet<usize>, Vec<TreeGrid>>,
    pub y_hat: Array1<f64>,
    pub residuals: Array1<f64>,
}

impl<'a> ForestFitter<'a> {
    pub fn new(points: ArrayView2<'a, f64>, labels: ArrayView1<'a, f64>) -> Self {
        let dims = points.shape()[1];
        let intercept_grid = TreeGrid::new(TreeGridParams {
            n_iter: 50,
            split_try: 10,
            colsample_bytree: 1.0,
        });
        let mut tree_grids = HashMap::new();
        tree_grids.insert(BTreeSet::new(), vec![]);
        tree_grids
            .get_mut(&BTreeSet::new())
            .unwrap()
            .push(intercept_grid);

        let y_hat = Array1::zeros(labels.len());
        let residuals = labels.to_owned() - &y_hat;

        Self {
            dims,
            points,
            labels,
            tree_grids,
            y_hat,
            residuals,
        }
    }

    pub fn update_y_hat(&mut self) {
        self.y_hat.fill(0.0);
        for grids in self.tree_grids.values() {
            for grid in grids {
                self.y_hat += &grid.predict(self.points);
            }
        }
        self.residuals = self.labels.to_owned() - &self.y_hat;
    }

    pub fn loss(&self) -> f64 {
        self.residuals.iter().map(|x| x * x).sum::<f64>() / self.residuals.len() as f64
    }

    pub fn potential_splits(&self) -> Vec<(BTreeSet<usize>, usize, usize)> {
        let mut possible_tree_structure_after_split: HashMap<
            BTreeSet<usize>,
            Vec<(BTreeSet<usize>, usize)>,
        > = HashMap::new();
        let mut rng = rand::thread_rng();

        for (s, j) in self.tree_grids.keys().cartesian_product(0..self.dims) {
            let mut s_union_j = s.clone();
            s_union_j.insert(j);

            if !self.tree_grids[s].is_empty()
                && (!self.tree_grids.contains_key(&s_union_j) || s.contains(&j))
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
                let tree_grid_idx = rng.gen_range(0..self.tree_grids[s].len());
                potential_splits.push((s.clone(), j, tree_grid_idx));
            }
        }

        potential_splits
    }

    pub fn update_estimator(
        &mut self,
        candidate: (BTreeSet<usize>, usize, usize),
        split: (f64, f64, usize, f64),
    ) {
        let (s, dim, sample_tg_idx) = candidate;
        let (_, _, split_dim, split_val) = split;

        if let Some(tree_grids) = self.tree_grids.get_mut(&s) {
            if let Some(sample_tg) = tree_grids.get_mut(sample_tg_idx) {
                sample_tg.fit(self.points, self.labels);
            }
        }

        if !s.contains(&dim) {
            if let Some(tree_grids) = self.tree_grids.get_mut(&s) {
                let mut new_s = s.clone();
                new_s.insert(dim);

                if let Some(sample_tg) = tree_grids.get(sample_tg_idx) {
                    let mut new_grid = TreeGrid::new(TreeGridParams {
                        n_iter: 50,
                        split_try: 10,
                        colsample_bytree: 1.0,
                    });
                    new_grid.fit(self.points, self.labels);

                    tree_grids.remove(sample_tg_idx);
                    self.tree_grids.entry(new_s).or_default().push(new_grid);
                }
            }

            if s.is_empty() {
                let mut new_grid = TreeGrid::new(TreeGridParams {
                    n_iter: 50,
                    split_try: 10,
                    colsample_bytree: 1.0,
                });
                new_grid.fit(self.points, self.labels);

                self.tree_grids
                    .entry(BTreeSet::new())
                    .or_default()
                    .push(TreeGrid::new(TreeGridParams {
                        n_iter: 50,
                        split_try: 10,
                        colsample_bytree: 1.0,
                    }));
            }
        }

        self.update_y_hat();
    }

    pub fn fit(&mut self, n_iter: usize, m_try: f64, split_try: usize) -> f64 {
        let mut rng = rand::thread_rng();
        let mut best_loss = self.loss();

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
                    let split_idx = rng.gen_range(0..self.points.shape()[0]);
                    let split_val = self.points[[split_idx, dim]];

                    if let Some(tree_grids) = self.tree_grids.get(&s) {
                        if let Some(sample_tg) = tree_grids.get(sample_tg_idx) {
                            let old_loss = self.loss();
                            let mut new_grid = TreeGrid::new(TreeGridParams {
                                n_iter: 50,
                                split_try: 10,
                                colsample_bytree: 1.0,
                            });
                            new_grid.fit(self.points, self.labels);
                            let new_loss = self.loss();
                            let err_diff = old_loss - new_loss;

                            if err_diff > best_err_diff {
                                best_err_diff = err_diff;
                                best_split = Some((old_loss, new_loss, dim, split_val));
                            }
                        }
                    }
                }

                if let Some(split) = best_split {
                    candidates.push(((s, sample_tg_idx, dim), split));
                }
            }

            if let Some((candidate, split)) = candidates.into_iter().max_by(|a, b| {
                let a_diff = a.1 .0 - a.1 .1;
                let b_diff = b.1 .0 - b.1 .1;
                a_diff.partial_cmp(&b_diff).unwrap()
            }) {
                self.update_estimator(candidate, split);
                best_loss = self.loss();
            }
        }

        best_loss
    }
}
