# %% [markdown]
# # Correlated Features Impact Analysis
# This notebook explores how feature correlation affects model performance.

# %% Imports
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from utils import plot_2d_model_predictions, random_hyperparam_search_parallel, true_model3
import mpf_py
from scipy.stats import randint

# %% Data generation function
def gen_data(n=5000, seed=1, var=5, cov=0.3, model=true_model3):
    """Generate data with controlled correlation between features."""
    np.random.seed(seed)
    mean = np.array([0, 0])  # Mean vector
    cov_matrix = np.array([[var, cov], [cov, var]])  # Covariance matrix
    X = np.random.multivariate_normal(mean, cov_matrix, size=n)
    y = model(X) + np.random.normal(scale=0.5, size=n)
    return X, y

# %% Utility functions
def plot_correlation(X, title="Feature Correlation"):
    """Plot the correlation between features."""
    plt.figure(figsize=(8, 8))
    plt.scatter(X[:, 0], X[:, 1], alpha=0.5)
    plt.title(title)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.grid(True)
    plt.show()

def evaluate_model(model, X_test, y_test, title="Model Evaluation"):
    """Evaluate model performance and print metrics."""
    predictions = model.predict(X_test)
    mse = np.mean((y_test - predictions) ** 2)
    r2 = 1 - np.sum((y_test - predictions) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)
    
    print(f"\n{title}")
    print(f"MSE: {mse:.4f}")
    print(f"R²: {r2:.4f}")
    
    return predictions, mse, r2

# %% Setup experiment parameters
correlations = [2, 3, 6]
results = []

# %% Run experiments with different correlation values
for rho in correlations:
    print(f"\nTesting with correlation ρ = {rho}")
    X, y = gen_data(n=10000, cov=rho, seed=42)

    # Plot correlation
    plot_correlation(X, f"Feature Correlation (ρ = {rho})")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    best_model, best_fr, best_params, best_error = random_hyperparam_search_parallel(
        X_train, y_train, n_splits=2, n_candidates=10, n_jobs=1,
        param_distributions={
            "epochs": lambda: randint(2, 9).rvs(),       # 1 to 8 inclusive
            "n_iter": lambda: randint(5, 101).rvs(),     # 5 to 100 inclusive
            "split_try": lambda: randint(5, 26).rvs(),   # 5 to 25 inclusive
            "B": lambda: randint(1, 101).rvs(),          # 1 to 100 inclusive
            "colsample_bytree": lambda: 1.0,
            "identified": lambda: True
        }
    )
    print(f"Best params: {best_params}, best error: {best_error}")

    predictions, mse, r2 = evaluate_model(
        best_model, X_test, y_test, 
        f"MPF Performance (ρ = {rho})"
    )

    results.append({
        'correlation': rho,
        'mse': mse,
        'r2': r2
    })

    print("\nVisualizing model components...")
    plot_2d_model_predictions(lambda x: best_model.predict(x), title=f"MPF Performance (ρ = {rho})")
    for tgf in best_model.tree_grid_families:
        combined_grid = mpf_py.TreeGrid(tgf.combined_tree_grid)
        combined_grid.plot_components()

# %% Plot performance metrics vs correlation
correlations = [r['correlation'] for r in results]
mses = [r['mse'] for r in results]
r2s = [r['r2'] for r in results]

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(correlations, mses, 'o-')
plt.title('MSE vs Correlation')
plt.xlabel('Correlation (ρ)')
plt.ylabel('Mean Squared Error')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(correlations, r2s, 'o-')
plt.title('R² vs Correlation')
plt.xlabel('Correlation (ρ)')
plt.ylabel('R² Score')
plt.grid(True)

plt.tight_layout()
plt.show()
