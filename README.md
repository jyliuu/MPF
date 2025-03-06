# MPF - Multi-Partitioned Forest

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
from mpf_py import MPF
from sklearn.model_selection import train_test_split

# Prepare data
X = np.random.rand(1000, 2)
y = 2*X[:,1] + X[:,0] - 0.5 * X[:,0]* X[:,1] + 34
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit MPF Boosted model with reproducible results
model, fit_result = MPF.fit_boosted(
    X_train, y_train,
    epochs=3,
    n_trees1,
    n_iter=260,
    split_try=20,
    colsample_bytree=1.0,
    identified=True,
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
    let params = MPFBoostedParams {
        epochs: 5,
        tgf_params: TreeGridFamilyBoostedParams {
            B: 100,
            tg_params: TreeGridParams {
                n_iter: 100,
                split_try: 10,
                colsample_bytree: 1.0,
                identified: true,
            },
        },
        seed: 42,  // Set seed for reproducibility
    };
    
    let (fit_result, model) = fit_boosted(x.view(), y.view(), &params);

    // Make predictions
    let predictions = model.predict(x.view());
}
```

## Project Structure

```
MPF/
├── src/                # Core Rust implementation
│   ├── forest/         # MPF algorithm implementation
│   │   ├── forest_fitter.rs  # Main fitting algorithms
│   │   └── mpf.rs      # MPF model definition
│   ├── tree_grid/      # Tree grid implementation
│   │   ├── family/     # Tree grid family variants
│   │   └── grid/       # Base tree grid functionality
│   └── lib.rs          # Main library exports
├── mpf-py/             # Python interface
│   ├── src/            # Rust-Python bindings
│   ├── tests/          # Python interface tests
│   └── python/         # Python package code
└── README.md           # Project documentation
```

## API Documentation

### Python API

#### MPF
- `fit_boosted(X, y, epochs, B, n_iter, split_try, colsample_bytree, identified, seed)`: Fit model to data
- `predict(X)`: Make predictions
- `get_tree_grid_families()`: Access internal tree grid families

#### TreeGridFamily
- `get_tree_grids()`: Get individual tree grids
- `predict(X)`: Make predictions using this family

### Rust API

#### Main Functions
- `fit_boosted()`: Fit MPF model with bagging

#### Types
- `MPFBoostedParams`: Parameters for boosted MPF
- `TreeGridFamilyBoostedParams`: Parameters for boosted tree grid family
- `TreeGridParams`: Base tree grid parameters

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
