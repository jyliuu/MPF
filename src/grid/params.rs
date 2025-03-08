#[derive(Debug, Clone)]
pub enum SplitStrategyParams {
    RandomSplit {
        split_try: usize,
        colsample_bytree: f64,
    },
    IntervalRandomSplit {
        split_try: usize,
        colsample_bytree: f64,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub enum CombinationStrategyParams {
    ArithMean,
    Median,
    ArithmeticGeometricMean,
    GeometricMean,
    None,
}

#[derive(Debug, Clone)]
pub struct TreeGridParams {
    pub n_iter: usize,
    pub split_strategy_params: SplitStrategyParams,
    pub reproject_grid_values: bool,
    pub combination_strategy_params: CombinationStrategyParams,
}

// Builder for TreeGridParams
#[derive(Debug, Clone)]
pub struct TreeGridParamsBuilder {
    n_iter: usize,
    split_strategy_params: SplitStrategyParams,
    combination_strategy_params: CombinationStrategyParams,
    reproject_grid_values: bool,
}

impl TreeGridParamsBuilder {
    pub fn new() -> Self {
        Self {
            n_iter: 25,
            split_strategy_params: SplitStrategyParams::RandomSplit {
                split_try: 10,
                colsample_bytree: 1.0,
            },
            combination_strategy_params: CombinationStrategyParams::ArithMean,
            reproject_grid_values: true,
        }
    }

    pub fn n_iter(mut self, n_iter: usize) -> Self {
        self.n_iter = n_iter;
        self
    }

    pub fn split_strategy(mut self, strategy: SplitStrategyParams) -> Self {
        self.split_strategy_params = strategy;
        self
    }

    pub fn combination_strategy(mut self, strategy: CombinationStrategyParams) -> Self {
        self.combination_strategy_params = strategy;
        self
    }

    pub fn reproject_grid_values(mut self, reproject_grid_values: bool) -> Self {
        self.reproject_grid_values = reproject_grid_values;
        self
    }

    pub fn build(self) -> TreeGridParams {
        TreeGridParams {
            n_iter: self.n_iter,
            split_strategy_params: self.split_strategy_params,
            combination_strategy_params: self.combination_strategy_params,
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
        TreeGridParamsBuilder::new().build()
    }
}
