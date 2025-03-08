# MPF - Multi-Partitioned Forest (Work In Progress)

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)


**MPF** is a machine learning library for Python (written in Rust) that implements an interpretable machine learning model by fitting sums of multiplicative (separable) structures. It tries to learn a regression model $\hat{m}(x_1, \dots, x_p)$ as a sum of rank‑1, separable components:
  
$$
\hat{m}(x_1, \dots, x_p) \approx \sum_{k=1}^{K} \lambda_k \prod_{j=1}^{p} \hat{m}_{j,k}(x_j)
$$

where each $\hat{m}_{j,k}(x_j)$ is a univariate function capturing the effect of variable $x_j$ in the $k$'th component $\lambda_k$ is a scaling factor for the $k$th component.

This structure naturally decomposes the function into interpretable main effects and interactions.

## Key Features

- 🌳 **Functional Partitioning**: Models decision boundaries as products of functions
- 🐍 **Python Integration**: Seamless NumPy integration through `mpf-py` package
- 🚀 **High Performance**: Built in Rust for optimal speed and memory efficiency
- 🎯 **Reproducibility**: Deterministic results through seeded random number generation
- 📊 **Model Diagnostics**: Built-in tools for model interpretation and visualization

## Installation

### Rust Installation

1. Install Rust using [rustup](https://rustup.rs/):
   ```sh
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   ```

2. Clone and build the project:
   ```sh
   git clone https://github.com/jyliuu/MPF.git
   cd MPF
   cargo build --release
   ```

### Python Package Installation

1. Navigate to the Python package directory:
   ```sh
   cd mpf-py
   ```

2. Install the package:
   ```sh
   pip install .
   ```

## Usage

### Python Example

```python
import numpy as np
from mpf_py import MPF, TreeGrid
from sklearn.model_selection import train_test_split

# Prepare data
X = np.random.rand(1000, 2)
y = 2*X[:,1] + X[:,0] - 0.5 * X[:,0]* X[:,1] + 34
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit MPF Boosted model with reproducible results
model, fit_result = MPF.fit_boosted(
    X_train, y_train,
    epochs=3,
    n_trees=1,
    n_iter=260,
    split_try=20,
    colsample_bytree=1.0,
    combination_strategy='l2_arith_geom_mean',
    reproject_grid_values=True,
    seed=1  # Set seed for reproducibility
)

# Make predictions
predictions = model.predict(X_test)
test_error = np.mean((y_test - predictions) ** 2)
print(f"Test MSE: {test_error}")

# Access and visualize model components
tree_grid_families = model.tree_grid_families
for family in tree_grid_families:
    # Get individual tree grids
    tree_grids = family.tree_grids
    for grid in tree_grids:
        # Plot component functions for each dimension
        grid = TreeGrid(grid)
        grid.plot_components()
        
        # Visualize grid predictions
        from utils import plot_2d_model_predictions
        plot_2d_model_predictions(
            lambda x: grid.predict(x), 
            title=f"Tree Grid (scaling: {grid.scaling})"
        )

```

### Rust Example

```rust
use mpf::{FitResult, FittedModel};
use ndarray::{Array1, Array2};

fn main() {
    // Load your data
    let x: Array2<f64> = /* your features */;
    let y: Array1<f64> = /* your targets */;

    // Fit MPF model with reproducible results
    let params = MPFBoostedParamsBuilder::new()
        .epochs(40)
        .n_iter(120) // Using default, but explicitly stated for clarity
        .n_trees(4)
        .combination_strategy(CombinationStrategyParams::L2ArithmeticGeometricMean)
        .split_strategy(SplitStrategyParams::RandomSplit {
            split_try: 12,
            colsample_bytree: 1.0,
        })
        .build();
    
    let (fit_result, model) = fit_boosted(x.view(), y.view(), &params);

    // Make predictions
    let predictions = model.predict(x.view());
}
```

## Project Structure

```
MPF/
├── benches/             # Benchmarking code
│   └── tree_grid_fitter.rs
├── clippy.toml
├── data/                # Sample datasets
│   ├── dat.csv
│   └── housing.csv
├── mpf-py/              # Python interface
│   ├── Cargo.lock
│   ├── Cargo.toml
│   ├── notebooks/       # Example notebooks and scripts
│   ├── pyproject.toml   # Python package configuration
│   ├── python/          # Python package code
│   │   ├── example.py
│   │   ├── main.py
│   │   └── mpf_py/
│   ├── src/             # Rust-Python bindings
│   │   └── lib.rs
│   └── tests/           # Python interface tests
│       ├── test_models.py
│       └── test_reproducibility.py
├── src/                 # Core Rust implementation
│   ├── family/          # Family implementation
│   │   ├── combine_grids.rs
│   │   ├── fitter.rs
│   │   └── params.rs
│   ├── family.rs        # Family module exports
│   ├── forest/          # Forest implementation
│   │   ├── fitter.rs
│   │   └── params.rs
│   ├── forest.rs        # Forest module exports
│   ├── grid/            # Grid implementation
│   │   ├── candidates.rs      # Handles generation of candidate split points for trees
│   │   ├── fitter.rs          # Implements core algorithm for fitting tree grid models
│   │   ├── grid_index.rs      # Manages grid data structure and indexing operations
│   │   ├── identification.rs  # Identifies grid components
│   │   ├── params.rs          # Defines hyperparameters for grid models
│   │   ├── reproject_values.rs # Implements value reprojection for model refinement
│   │   └── splitting.rs       # Implements strategies for optimal grid splitting
│   ├── grid.rs          # Grid module exports
│   └── lib.rs           # Main library exports
└── tests/               # Rust tests
    ├── family.rs
    ├── forest.rs
    ├── test_data.rs
    └── tree_grid.rs
```

## API Documentation
Coming soon...

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
