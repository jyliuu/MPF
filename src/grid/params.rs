#[derive(Debug, Clone)]
pub enum SplitStrategyParams {
    RandomSplit {
        split_try: usize,
        colsample_bytree: f64,
    },
    // Add other strategies here as needed
}

#[derive(Debug, Clone)]
pub enum CandidateStrategyParams {
    GreedySelection,
    // Add other strategies here as needed
}

#[derive(Debug, Clone)]
pub struct TreeGridParams {
    pub n_iter: usize,
    pub split_strategy_params: SplitStrategyParams,
    pub candidate_strategy_params: CandidateStrategyParams,
    pub identified: bool,
}

// Builder for TreeGridParams
#[derive(Debug, Clone)]
pub struct TreeGridParamsBuilder {
    n_iter: usize,
    split_try: usize,
    colsample_bytree: f64,
    candidate_strategy_params: CandidateStrategyParams,
    identified: bool,
}

impl TreeGridParamsBuilder {
    pub fn new() -> Self {
        Self {
            n_iter: 25,
            split_try: 10,
            colsample_bytree: 1.0,
            candidate_strategy_params: CandidateStrategyParams::GreedySelection,
            identified: true,
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

    pub fn candidate_strategy_params(mut self, strategy: CandidateStrategyParams) -> Self {
        self.candidate_strategy_params = strategy;
        self
    }

    pub fn identified(mut self, identified: bool) -> Self {
        self.identified = identified;
        self
    }

    pub fn build(self) -> TreeGridParams {
        TreeGridParams {
            n_iter: self.n_iter,
            split_strategy_params: SplitStrategyParams::RandomSplit {
                split_try: self.split_try,
                colsample_bytree: self.colsample_bytree,
            },
            candidate_strategy_params: self.candidate_strategy_params,
            identified: self.identified,
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
            candidate_strategy_params: CandidateStrategyParams::GreedySelection,
            identified: true,
        }
    }
}
