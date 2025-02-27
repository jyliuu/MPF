#[derive(Debug, Clone)]
pub enum SplitStrategy {
    RandomSplit {
        split_try: usize,
        colsample_bytree: f64,
    },
    // Add other strategies here as needed
}

#[derive(Debug, Clone)]
pub enum CandidateStrategy {
    GreedySelection,
    // Add other strategies here as needed
}

#[derive(Debug, Clone)]
pub struct TreeGridParams {
    pub n_iter: usize,
    pub split_strategy: SplitStrategy,
    pub candidate_strategy: CandidateStrategy,
    pub identified: bool,
}

// Builder for TreeGridParams
#[derive(Debug, Clone)]
pub struct TreeGridParamsBuilder {
    n_iter: usize,
    split_try: usize,
    colsample_bytree: f64,
    candidate_strategy: CandidateStrategy,
    identified: bool,
}

impl TreeGridParamsBuilder {
    pub fn new() -> Self {
        Self {
            n_iter: 25,
            split_try: 10,
            colsample_bytree: 1.0,
            candidate_strategy: CandidateStrategy::GreedySelection,
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

    pub fn candidate_strategy(mut self, strategy: CandidateStrategy) -> Self {
        self.candidate_strategy = strategy;
        self
    }

    pub fn identified(mut self, identified: bool) -> Self {
        self.identified = identified;
        self
    }

    pub fn build(self) -> TreeGridParams {
        TreeGridParams {
            n_iter: self.n_iter,
            split_strategy: SplitStrategy::RandomSplit {
                split_try: self.split_try,
                colsample_bytree: self.colsample_bytree,
            },
            candidate_strategy: self.candidate_strategy,
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
            split_strategy: SplitStrategy::RandomSplit {
                split_try: 10,
                colsample_bytree: 1.0,
            },
            candidate_strategy: CandidateStrategy::GreedySelection,
            identified: true,
        }
    }
}
