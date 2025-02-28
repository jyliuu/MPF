#[derive(Debug, Clone)]
pub enum SplitStrategyParams {
    RandomSplit {
        split_try: usize,
        colsample_bytree: f64,
    },
    // Add other strategies here as needed
}

#[derive(Debug, Clone, PartialEq)]
pub enum IdentificationStrategyParams {
    L2_ARITH_MEAN,
    L2_MEDIAN,
    None,
}

#[derive(Debug, Clone)]
pub struct TreeGridParams {
    pub n_iter: usize,
    pub split_strategy_params: SplitStrategyParams,
    pub reproject_grid_values: bool,
    pub identification_strategy_params: IdentificationStrategyParams,
}

// Builder for TreeGridParams
#[derive(Debug, Clone)]
pub struct TreeGridParamsBuilder {
    n_iter: usize,
    split_try: usize,
    colsample_bytree: f64,
    identification_strategy_params: IdentificationStrategyParams,
    reproject_grid_values: bool,
}

impl TreeGridParamsBuilder {
    pub fn new() -> Self {
        Self {
            n_iter: 25,
            split_try: 10,
            colsample_bytree: 1.0,
            identification_strategy_params: IdentificationStrategyParams::L2_ARITH_MEAN,
            reproject_grid_values: true,
        }
    }

    pub fn n_iter(mut self, n_iter: usize) -> Self {
        self.n_iter = n_iter;
        self
    }

    pub fn split_try(mut self, split_try: usize) -> Self {
        self.split_try = split_try;
        self
    }

    pub fn colsample_bytree(mut self, colsample_bytree: f64) -> Self {
        self.colsample_bytree = colsample_bytree;
        self
    }

    pub fn identified(mut self, identified: bool) -> Self {
        self.identification_strategy_params = if identified {
            IdentificationStrategyParams::L2_ARITH_MEAN
        } else {
            IdentificationStrategyParams::None
        };
        self
    }

    pub fn identification_strategy(mut self, strategy: IdentificationStrategyParams) -> Self {
        self.identification_strategy_params = strategy;
        self
    }

    pub fn reproject_grid_values(mut self, reproject_grid_values: bool) -> Self {
        self.reproject_grid_values = reproject_grid_values;
        self
    }

    pub fn build(self) -> TreeGridParams {
        TreeGridParams {
            n_iter: self.n_iter,
            split_strategy_params: SplitStrategyParams::RandomSplit {
                split_try: self.split_try,
                colsample_bytree: self.colsample_bytree,
            },
            identification_strategy_params: self.identification_strategy_params,
            reproject_grid_values: self.reproject_grid_values,
        }
    }
}

impl Default for TreeGridParamsBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for TreeGridParams {
    fn default() -> Self {
        TreeGridParams {
            n_iter: 25,
            split_strategy_params: SplitStrategyParams::RandomSplit {
                split_try: 10,
                colsample_bytree: 1.0,
            },
            identification_strategy_params: IdentificationStrategyParams::L2_ARITH_MEAN,
            reproject_grid_values: true,
        }
    }
}
