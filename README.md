# MPF - Multi-Partitioned Forest

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

MPF is a supervised learning algorithm that models using sums of products of univariate functions. This approach provides a different way to model decision boundaries compared to traditional decision trees, offering potentially better interpretability while maintaining predictive performance.

## Key Features

- 🌳 **Functional Partitioning**: Models decision boundaries as products of functions
- 🐍 **Python Integration**: Seamless NumPy integration through `mpf-py` package
- 🚀 **High Performance**: Built in Rust for optimal speed and memory efficiency
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

### Rust Example

```rust
use mpf::{FitResult, FittedModel};
use ndarray::{Array1, Array2};

fn main() {
    // Load your data
    let x: Array2<f64> = /* your features */;
    let y: Array1<f64> = /* your targets */;

    // Fit TreeGrid model
    let (tg_fit_result, tg_model) = mpf::tree_grid::fit(
        x.view(),
        y.view(),
        mpf::tree_grid::TreeGridParams {
            n_iter: 100,
            split_try: 10,
            colsample_bytree: 1.0,
        }
    );

    // Fit MPF model
    let (mpf_fit_result, mpf_model) = mpf::fit_bagged(
        x.view(),
        y.view(),
        mpf::MPFBaggedParams {
            epochs: 5,
            tgf_params: mpf::tree_grid::TreeGridFamilyBaggedParams {
                B: 100,
                tg_params: mpf::tree_grid::TreeGridParams {
                    n_iter: 100,
                    split_try: 10,
                    colsample_bytree: 1.0,
                },
            },
        }
    );

    // Make predictions
    let tg_predictions = tg_model.predict(x.view());
    let mpf_predictions = mpf_model.predict(x.view());
}
```

### Python Example

```python
import numpy as np
from mpf_py import TreeGrid, MPF

# Prepare data
X = np.random.rand(100, 2)
y = np.random.rand(100)

# Fit TreeGrid model
tg_model, tg_fit_result = TreeGrid.fit(
    X, y,
    n_iter=100,
    split_try=10,
    colsample_bytree=1.0
)

# Fit MPF model
mpf_model, mpf_fit_result = MPF.fit_bagged(
    X, y,
    epochs=5,
    B=100,
    n_iter=100,
    split_try=10,
    colsample_bytree=1.0
)

# Make predictions
tg_predictions = tg_model.predict(X)
mpf_predictions = mpf_model.predict(X)

# Visualize first component
tg_model.plot(axis=0)
```

## Project Structure

```
MPF/
├── src/                  # Core Rust implementation
│   ├── forest/           # MPF algorithm implementation
│   ├── tree_grid/        # Tree grid fitting algorithms
│   ├── lib.rs            # Main library exports
│   └── main.rs           # Example usage
├── mpf-py/               # Python interface
│   ├── python/           # Python package
│   └── src/              # Rust-Python bindings
├── examples/             # Usage examples
├── Cargo.toml            # Rust project configuration
└── README.md             # Project documentation
```

## API Documentation

### Core Rust API

- `mpf::fit()` - Main fitting function
- `mpf::FitResult` - Model fitting results
- `mpf::FittedModel` - Trait for fitted models
- `mpf::ModelFitter` - Trait for model fitting

### Python API

- `MPF()` - Main model class
- `fit(X, y)` - Fit model to data
- `predict(X)` - Make predictions

## Contributing

We welcome contributions! Please see our [contribution guidelines](CONTRIBUTING.md) for details.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
