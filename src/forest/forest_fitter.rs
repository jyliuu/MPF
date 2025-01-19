use ndarray::{ArrayView1, ArrayView2};

use crate::{tree_grid::tree_grid_family::{TreeGridFamily, TreeGridFamilyFitter}, FitResult};

use super::mpf::MPF;

pub struct MPFFitter<'a> {
    pub x: ArrayView2<'a, f64>,
    pub y: ArrayView1<'a, f64>,
}

impl<'a> MPFFitter<'a> {
    pub fn new(x: ArrayView2<'a, f64>, y: ArrayView1<'a, f64>) -> Self {
        Self { x, y }
    }

    pub fn fit(
        self,
        n_families: usize,
        n_iter: usize,
        m_try: f64,
        split_try: usize,
    ) -> (FitResult, MPF<TreeGridFamily>) {
        let mut fitted_tree_grid_families = Vec::new();
        let mut fit_results = Vec::new();
        for _ in 0..n_families {
            let tg_family_fitter = TreeGridFamilyFitter::new(self.x, self.y);
            let (tgf_fit_result, tree_grid_family) = tg_family_fitter.fit(n_iter, m_try, split_try);
            fitted_tree_grid_families.push(tree_grid_family);
            fit_results.push(tgf_fit_result);
        }

        let mut fit_result = fit_results
            .into_iter()
            .reduce(|a, b| FitResult {
                err: 0.0,
                residuals: a.residuals + b.residuals,
                y_hat: a.y_hat + b.y_hat,
            })
            .unwrap();

        fit_result.residuals /= n_families as f64;
        fit_result.y_hat /= n_families as f64;
        fit_result.err = fit_result.residuals.pow2().mean().unwrap();

        (fit_result, MPF::new(fitted_tree_grid_families))
    }
}
