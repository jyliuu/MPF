# MPF

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)


MPF is a decision-tree like, forest model for regression tasks. The primary goal of this project is to provide explainability in regression modeling by modeling decision trees as products of functions. This approach aims to offer more interpretable models compared to traditional decision tree methods.

## Features
 - [x] Python/NumPy integration (see `mpf-py`) [WIP]
 - [ ] Model diagnostic plots and other plots for interpretability 


## Installation

To use this project, you need to have Rust installed. You can install Rust from [here](https://www.rust-lang.org/tools/install).

Clone the repository and navigate to the project directory:

```sh
git clone https://github.com/jyliuu/MPF.git
cd MPF
```

Build the project:

```sh
cargo build --release
```

## Usage

To run the main program, use the following command:

```sh
cargo run --release
```

This will execute the main function in `src/main.rs`, which reads data from `dat.csv`, fits the MPF model, and prints the error.

## Project Structure

- `mpf-py`: Python/NumPy interface.
- `src/forest/`: Contains the implementation of the MPF algorithm.
- `src/tree_grid/`: Contains the implementation of tree grid fitting and family fitting algorithms.
- `Cargo.toml`: Project configuration file.

## Example

Ensure you have a `dat.csv` file in the project directory with the following format:

```csv
y,x1,x2
0.5,0.1,0.2
0.5,0.3,0.5
1.5,0.7,0.6
1.5,1.1,1.2
1.5,1.3,1.5
1.5,1.7,1.6
```

Run the project:

```sh
cargo run --release
```

## License

This project is licensed under the Apache License.
