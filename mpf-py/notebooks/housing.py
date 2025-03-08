# %% [markdown]
# # California Housing Dataset Analysis
# This notebook demonstrates the use of MPF and XGBoost models on the California housing dataset,
# focusing on geographic prediction patterns using latitude and longitude features.

# %% Imports
import mpf_py
import numpy as np
from sklearn.datasets import fetch_california_housing
from xgboost import XGBRegressor
from scipy.stats import randint
from utils import plot_2d_model_predictions, plot_2d_model_predictions_map, random_hyperparam_search_parallel

# %% Load the California housing dataset
california = fetch_california_housing()

# %% Examine dataset structure
X = np.ascontiguousarray(california.data)
y = np.ascontiguousarray(california.target) 

print(f"Dataset shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"Feature names: {california.feature_names}")

# %% Extract geographic features (latitude and longitude)
X = np.ascontiguousarray(X[:, [-2, -1]])  # Latitude and Longitude columns
lat_range = (X[:, 0].min(), X[:, 0].max())
lon_range = (X[:, 1].min(), X[:, 1].max())
print(f"Latitude range: {lat_range}")
print(f"Longitude range: {lon_range}")

# %% [markdown]
# ## XGBoost Model
# First, we'll train an XGBoost model with optimal hyperparameters

# %% Configure and train XGBoost model
best_xgboost_params = {
    'learning_rate': 0.05,
    'max_depth': 7,
    'n_estimators': 300,
    'gamma': 0,
    'min_child_weight': 1,
}

best_model_xgboost = XGBRegressor(**best_xgboost_params)
best_model_xgboost.fit(X, y)

# %% Visualize XGBoost predictions on California map
plot_2d_model_predictions_map(
    lambda x: best_model_xgboost.predict(x), 
    title="Best XGBoost model",
    data=(X, y),
    cbar_max=5
)

# %% [markdown]
# ## MPF Model
# Now, we'll train an MPF model and compare its performance with XGBoost

# %% Cross-validation for MPF hyperparameters (commented out)
best_model, best_fr, best_params, best_error = random_hyperparam_search_parallel(
    X, y, n_splits=2, n_candidates=10, n_jobs=1, param_distributions= {
        "epochs": lambda: randint(2, 40).rvs(),       # 1 to 8 inclusive
        "n_iter": lambda: randint(5, 100).rvs(),      # 5 to 40 inclusive
        "split_try": lambda: randint(5, 21).rvs(),   # 5 to 20 inclusive
        "n_trees": lambda: randint(10, 100).rvs(),          # 10 to 70 inclusive
        "colsample_bytree": lambda: 1.0,
        "combination_strategy": lambda: np.random.choice(["arith_mean","arith_geom_mean","median"]),
        "identification_strategy": lambda: np.random.choice(["l1","l2"]),
        "split_strategy": lambda: np.random.choice(["interval_random","random"])
    })
print("Best hyperparameters for MPF:", best_params)
print("Best CV MSE for MPF:", best_error)

# %% Train MPF model with best hyperparameters
best_params = {
    'epochs': 13, 
    'n_iter': 39,
    'split_try': 12,
    'n_trees': 72,
    'colsample_bytree': 1.0, 
    'combination_strategy': 'arith_geom_mean',
    'identification_strategy': 'l2',
    'split_strategy': 'interval_random'
}
print("Using MPF hyperparameters:", best_params)

best_model, best_fr = mpf_py.MPF.fit_boosted(
    X, y,
    **best_params
)

# %%
mse_mpf = np.mean((y - best_model.predict(X)) ** 2)
print(f"MSE of MPF: {mse_mpf}")


# %% Visualize MPF predictions on California map
plot_2d_model_predictions(
    lambda x: best_model.predict(x), 
    x_bounds=lat_range,
    y_bounds=lon_range,
    title="Best MPF model",
    # data=(X, y),
    cbar_max=5
)

# %% Visualize MPF components
print("MPF Model Components:")
for i, tgf in enumerate(best_model.tree_grid_families):
    print(f"Tree Grid Family {i+1}:")
    combined_tg = mpf_py.TreeGrid(tgf.combined_tree_grid)
    combined_tg.plot_components(individual_plots=True)
    print(f"Grid Values: {combined_tg.grid_values}")

# %% [markdown]
# ## Model Comparison
# Compare performance metrics between XGBoost and MPF

# %% Compute and compare performance metrics
mse_xgboost = np.mean((y - best_model_xgboost.predict(X)) ** 2)

print(f"MSE of XGBoost: {mse_xgboost:.4f}")
print(f"Relative improvement: {((mse_xgboost - mse_mpf) / mse_xgboost) * 100:.2f}%")

# %%
