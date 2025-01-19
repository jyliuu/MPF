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

use super::{grid::FittedTreeGrid, tree_grid_fitter::TreeGridParams};

pub struct BaggedVariant;
pub struct AveragedVariant;

impl Default for BaggedVariant {
    fn default() -> Self {
        BaggedVariant
    }
}

impl Default for AveragedVariant {
    fn default() -> Self {
        AveragedVariant
    }
}

#[derive(Debug)]
pub struct TreeGridFamily<T>(T, Vec<FittedTreeGrid>);

impl FittedModel for TreeGridFamily<BaggedVariant> {
    fn predict(&self, x: ArrayView2<f64>) -> Array1<f64> {
        let mut result = Array1::ones(x.shape()[0]);
        let mut signs = Array1::from_elem(x.shape()[0], 0.0);
        for grids in &self.1 {
            let pred = grids.predict(x.view());

            result *= &pred;
            signs += &pred.signum();
        }

        signs = signs.signum();

        result.zip_mut_with(&signs, |v, sign| {
            *v = sign * (*v).abs().powf(1.0 / self.1.len() as f64);
        });

        result
    }
}

impl FittedModel for TreeGridFamily<AveragedVariant> {
    fn predict(&self, x: ArrayView2<f64>) -> Array1<f64> {
        let mut result = Array1::zeros(x.shape()[0]);
        for grid in self.1.iter() {
            result += &grid.predict(x);
        }
        result
    }
}

#[derive(Debug)]
pub struct TreeGridFamilyAveragedParams {
    pub n_iter: usize,
    pub m_try: f64,
    pub split_try: usize,
}

#[derive(Debug)]
pub struct TreeGridFamilyBaggedParams {
    pub B: usize,
    pub tg_params: TreeGridParams,
}

pub trait FitAndPredictStrategy<'a> {
    type HyperParameters;
    type Model: FittedModel;
    type Features: 'a;
    type Labels: 'a;

    fn fit(
        x: Self::Features,
        y: Self::Labels,
        hyperparameters: Self::HyperParameters,
    ) -> (FitResult, Self::Model);
    fn predict(model: Self::Model, x: Self::Features) -> Array1<f64>;
}

impl<'a> FitAndPredictStrategy<'a> for BaggedVariant {
    type HyperParameters = TreeGridFamilyBaggedParams;
    type Model = TreeGridFamily<BaggedVariant>;
    type Features = ArrayView2<'a, f64>;
    type Labels = ArrayView1<'a, f64>;

    fn fit(
        x: Self::Features,
        y: Self::Labels,
        hyperparameters: Self::HyperParameters,
    ) -> (FitResult, Self::Model) {
        TreeGridFamilyBaggedFitter::new(x, y).fit(hyperparameters)
    }
    fn predict(model: Self::Model, x: Self::Features) -> Array1<f64> {
        model.predict(x)
    }
}

impl<'a> FitAndPredictStrategy<'a> for AveragedVariant {
    type HyperParameters = TreeGridFamilyAveragedParams;
    type Model = TreeGridFamily<AveragedVariant>;
    type Features = ArrayView2<'a, f64>;
    type Labels = ArrayView1<'a, f64>;

    fn fit(
        x: Self::Features,
        y: Self::Labels,
        hyperparameters: Self::HyperParameters,
    ) -> (FitResult, Self::Model) {
        TreeGridFamilyAveragedFitter::new(x, y).fit(hyperparameters)
    }
    fn predict(model: Self::Model, x: Self::Features) -> Array1<f64> {
        model.predict(x)
    }
}

pub struct TreeGridFamilyFitter<'a, T: FitAndPredictStrategy<'a>> {
    variant: T,
    x: T::Features,
    y: T::Labels,
}

impl<'a, T> ModelFitter<'a> for TreeGridFamilyFitter<'a, T>
where
    TreeGridFamily<T>: FittedModel,
    T: FitAndPredictStrategy<'a> + Default,
{
    type Features = T::Features;
    type Labels = T::Labels;
    type Model = T::Model;
    type HyperParameters = T::HyperParameters;

    fn new(x: Self::Features, y: Self::Labels) -> Self {
        Self {
            variant: T::default(),
            x,
            y,
        }
    }

    fn fit(self, hyperparameters: Self::HyperParameters) -> (FitResult, Self::Model) {
        T::fit(self.x, self.y, hyperparameters)
    }
}

struct TreeGridFamilyBaggedFitter<'a> {
    pub x: ArrayView2<'a, f64>,
    pub y: ArrayView1<'a, f64>,
}

impl<'a> ModelFitter<'a> for TreeGridFamilyBaggedFitter<'a> {
    type Features = ArrayView2<'a, f64>;
    type Labels = ArrayView1<'a, f64>;
    type Model = TreeGridFamily<BaggedVariant>;
    type HyperParameters = TreeGridFamilyBaggedParams;

    fn new(x: Self::Features, y: Self::Labels) -> Self {
        Self { x, y }
    }

    fn fit(
        self,
        hyperparameters: Self::HyperParameters,
    ) -> (FitResult, TreeGridFamily<BaggedVariant>) {
        let TreeGridFamilyBaggedParams { B, tg_params } = hyperparameters;
        let TreeGridParams {
            n_iter,
            split_try,
            colsample_bytree,
        } = tg_params;
        let mut rng = rand::thread_rng();
        let mut tree_grids = vec![];

        let n = self.x.nrows();

        for b in 0..B {
            let sample_indices: Vec<usize> = (0..n).map(|_| rng.gen_range(0..n)).collect();
            let x_sample = self.x.select(ndarray::Axis(0), &sample_indices);
            let y_sample = self.y.select(ndarray::Axis(0), &sample_indices);
            let tg_fitter = TreeGridFitter::new(x_sample.view(), y_sample.view());
            let (fit_res, tg): (FitResult, FittedTreeGrid) = tg_fitter.fit(TreeGridParams {
                n_iter,
                split_try,
                colsample_bytree,
            });
            println!("b: {:?}, err: {:?}", b, fit_res.err);
            tree_grids.push(tg);
        }

        let tgf = TreeGridFamily(BaggedVariant, tree_grids);

        let preds = tgf.predict(self.x);
        let residuals = &self.y - &preds;
        let err = residuals.pow2().mean().unwrap();

        (
            FitResult {
                err,
                residuals: residuals.to_owned(),
                y_hat: preds,
            },
            tgf,
        )
    }
}
struct TreeGridFamilyAveragedFitter<'a> {
    dims: usize,
    x: ArrayView2<'a, f64>,
    y: ArrayView1<'a, f64>,
    tg_fitters: HashMap<BTreeSet<usize>, Vec<TreeGridFitter<'a>>>,
    y_hat: Array1<f64>,
    residuals: Array1<f64>,
}

impl TreeGridFamilyAveragedFitter<'_> {
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

impl<'a> ModelFitter<'a> for TreeGridFamilyAveragedFitter<'a> {
    type HyperParameters = TreeGridFamilyAveragedParams;
    type Model = TreeGridFamily<AveragedVariant>;
    type Features = ArrayView2<'a, f64>;
    type Labels = ArrayView1<'a, f64>;

    fn new(x: Self::Features, y: Self::Labels) -> Self {
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

    fn fit(
        mut self,
        hyperparameters: Self::HyperParameters,
    ) -> (FitResult, TreeGridFamily<AveragedVariant>) {
        let TreeGridFamilyAveragedParams {
            n_iter,
            m_try,
            split_try,
        } = hyperparameters;
        let mut rng = rand::thread_rng();

        for _ in 0..n_iter {
            let potential_splits = self.potential_splits();
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
            TreeGridFamily(
                AveragedVariant,
                self.tg_fitters.into_values().flatten().map_into().collect(),
            ),
        )
    }
}

#[cfg(test)]
mod tests {
    use csv::ReaderBuilder;
    use ndarray::Array2;

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
    fn test_tgf_bagged_fit() {
        let (x, y) = setup_data();
        let tgf_fitter: TreeGridFamilyBaggedFitter<'_> =
            TreeGridFamilyBaggedFitter::new(x.view(), y.view());
        let (fit_result, _) = tgf_fitter.fit(TreeGridFamilyBaggedParams {
            B: 100,
            tg_params: TreeGridParams {
                n_iter: 100,
                split_try: 10,
                colsample_bytree: 1.0,
            },
        });
        let mean = y.mean().unwrap();
        let base_err = (y - mean).powi(2).mean().unwrap();
        println!("Base error: {:?}, Error: {:?}", base_err, fit_result.err);
        assert!(
            fit_result.err < base_err,
            "Error is not less than mean error"
        );
    }

    #[test]
    fn test_tgf_averaged_fit() {
        let (x, y) = setup_data();
        let tgf_fitter = TreeGridFamilyAveragedFitter::new(x.view(), y.view());
        let (fit_result, _) = tgf_fitter.fit(TreeGridFamilyAveragedParams {
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

    #[test]
    fn test_tgf_fit() {
        let (x, y) = setup_data();
        let tgf_fitter: TreeGridFamilyFitter<'_, AveragedVariant> =
            TreeGridFamilyFitter::new(x.view(), y.view());

        let (fit_result, _) = tgf_fitter.fit(TreeGridFamilyAveragedParams {
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

    #[test]
    fn test_tgf_predict() {
        let (x, y) = setup_data();
        let tgf_fitter: TreeGridFamilyFitter<'_, AveragedVariant> =
            TreeGridFamilyFitter::new(x.view(), y.view());
        let (fit_result, tgf) = tgf_fitter.fit(TreeGridFamilyAveragedParams {
            n_iter: 100,
            m_try: 1.0,
            split_try: 10,
        });

        let pred = tgf.predict(x.view());
        let diff = fit_result.y_hat - pred;
        assert!(diff.iter().all(|&x| x < 1e-6));
    }
}
