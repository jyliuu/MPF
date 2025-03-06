use crate::grid::params::{
    IdentificationStrategyParams, SplitStrategyParams, TreeGridParams, TreeGridParamsBuilder,
};

#[derive(Debug)]
pub struct TreeGridFamilyBoostedParams {
    pub B: usize,
    pub bootstrap: bool,
    pub tg_params: TreeGridParams,
}

// Builder for TreeGridFamilyBoostedParams
#[derive(Debug)]
pub struct TreeGridFamilyBoostedParamsBuilder {
    B: usize,
    bootstrap: bool,
    tg_params_builder: TreeGridParamsBuilder,
}

impl TreeGridFamilyBoostedParamsBuilder {
    pub fn new() -> Self {
        Self {
            B: 100,
            bootstrap: false,
            tg_params_builder: TreeGridParamsBuilder::new(),
        }
    }

    pub fn B(mut self, B: usize) -> Self {
        self.B = B;
        self
    }

    pub fn bootstrap(mut self, bootstrap: bool) -> Self {
        self.bootstrap = bootstrap;
        self
    }

    // Convenience methods for TreeGridParams configuration
    pub fn n_iter(mut self, n_iter: usize) -> Self {
        self.tg_params_builder = self.tg_params_builder.n_iter(n_iter);
        self
    }

    pub fn split_strategy(mut self, strategy: SplitStrategyParams) -> Self {
        self.tg_params_builder = self.tg_params_builder.split_strategy(strategy);
        self
    }

    pub fn identified(self, identified: bool) -> Self {
        self.identification_strategy(if identified {
            IdentificationStrategyParams::L2_ARITH_MEAN
        } else {
            IdentificationStrategyParams::None
        })
    }

    pub fn identification_strategy(
        mut self,
        identification_strategy: IdentificationStrategyParams,
    ) -> Self {
        self.tg_params_builder = self
            .tg_params_builder
            .identification_strategy(identification_strategy);
        self
    }

    pub fn build(self) -> TreeGridFamilyBoostedParams {
        TreeGridFamilyBoostedParams {
            B: self.B,
            bootstrap: self.bootstrap,
            tg_params: self.tg_params_builder.build(),
        }
    }
}

impl Default for TreeGridFamilyBoostedParamsBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for TreeGridFamilyBoostedParams {
    fn default() -> Self {
        TreeGridFamilyBoostedParamsBuilder::new().build()
    }
}
