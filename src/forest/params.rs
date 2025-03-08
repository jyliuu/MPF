use crate::{
    family::params::{TreeGridFamilyBoostedParams, TreeGridFamilyBoostedParamsBuilder},
    grid::params::{CombinationStrategyParams, SplitStrategyParams},
};

#[derive(Debug, Clone)]
pub struct MPFBoostedParams {
    pub epochs: usize,
    pub tgf_params: TreeGridFamilyBoostedParams,
    pub seed: u64,
}

// Builder for MPFBoostedParams
pub struct MPFBoostedParamsBuilder {
    epochs: usize,
    tgf_params_builder: TreeGridFamilyBoostedParamsBuilder,
    seed: u64,
}

impl MPFBoostedParamsBuilder {
    pub fn new() -> Self {
        Self {
            epochs: 5,
            tgf_params_builder: TreeGridFamilyBoostedParamsBuilder::new(),
            seed: 42,
        }
    }

    pub fn epochs(mut self, epochs: usize) -> Self {
        self.epochs = epochs;
        self
    }

    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    // Convenience methods for nested parameters
    pub fn n_trees(mut self, n_trees: usize) -> Self {
        self.tgf_params_builder = self.tgf_params_builder.n_trees(n_trees);
        self
    }

    pub fn n_iter(mut self, n_iter: usize) -> Self {
        self.tgf_params_builder = self.tgf_params_builder.n_iter(n_iter);
        self
    }

    pub fn split_strategy(mut self, strategy: SplitStrategyParams) -> Self {
        self.tgf_params_builder = self.tgf_params_builder.split_strategy(strategy);
        self
    }

    pub fn reproject_grid_values(mut self, reproject_grid_values: bool) -> Self {
        self.tgf_params_builder = self
            .tgf_params_builder
            .reproject_grid_values(reproject_grid_values);
        self
    }

    pub fn combination_strategy(
        mut self,
        combination_strategy: CombinationStrategyParams,
    ) -> Self {
        self.tgf_params_builder = self
            .tgf_params_builder
            .combination_strategy(combination_strategy);
        self
    }

    pub fn build(self) -> MPFBoostedParams {
        MPFBoostedParams {
            epochs: self.epochs,
            tgf_params: self.tgf_params_builder.build(),
            seed: self.seed,
        }
    }
}

impl Default for MPFBoostedParamsBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for MPFBoostedParams {
    fn default() -> Self {
        MPFBoostedParamsBuilder::new().build()
    }
}
