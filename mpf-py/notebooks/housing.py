# %% [markdown]
# # California Housing Dataset Analysis
# This notebook demonstrates the use of MPF and XGBoost models on the California housing dataset,
# focusing on geographic prediction patterns using latitude and longitude features.

# %% Imports
import mpf_py
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from scipy.stats import uniform, randint
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from utils import random_hyperparam_search_parallel

# %% Load the California housing dataset
california = fetch_california_housing()

# %% Examine dataset structure
X = np.ascontiguousarray(california.data)
y = np.ascontiguousarray(california.target) 

print(f"Dataset shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"Feature names: {california.feature_names}")

# %% Extract geographic features (latitude and longitude)
X = np.ascontiguousarray(X[:, [-1, -2]])  # Latitude and Longitude columns
lat_range = (X[:, 0].min(), X[:, 0].max())
lon_range = (X[:, 1].min(), X[:, 1].max())
print(f"Latitude range: {lat_range}")
print(f"Longitude range: {lon_range}")

# %% Define visualization function for map-based predictions
def plot_2d_model_predictions(model, x_bounds=(-4,4), y_bounds=(-4,4), grid_points=100, cmap='viridis',
                           title="Model Predictions"):
    """
    Constructs a grid of (longitude, latitude) coordinates and uses the provided ML model to generate predictions.
    The predictions are then visualized as a colored grid overlaid on a map of California.

    Parameters:
        model (callable): A function or machine learning model that accepts an input array of shape (N, 2)
                          and returns prediction values as an array of shape (N,).
        x_bounds (tuple): A tuple (lon_min, lon_max) defining the longitude range.
        y_bounds (tuple): A tuple (lat_min, lat_max) defining the latitude range.
        grid_points (int): The number of points to generate along each axis.
        cmap (str): The colormap to use when displaying the predictions.
        title (str): The title for the plot.
    """
    # Create linearly spaced values for longitude and latitude
    lon_min, lon_max = x_bounds
    lat_min, lat_max = y_bounds

    lon_vals = np.linspace(lon_min, lon_max, grid_points)
    lat_vals = np.linspace(lat_min, lat_max, grid_points)

    # Create a mesh grid of points
    LON, LAT = np.meshgrid(lon_vals, lat_vals)

    # Flatten the grid to form (longitude, latitude) pairs
    grid = np.column_stack([LON.ravel(), LAT.ravel()])

    # Obtain predictions from the ML model
    predictions = model(grid)

    # Reshape predictions back into the grid format
    Z = predictions.reshape(LON.shape)

    # Set up the map projection
    plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # Add map features
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.STATES)
    
    # Set the extent to California (with a bit of margin)
    margin = 0.5  # degrees
    ax.set_extent([lon_min - margin, lon_max + margin, 
                   lat_min - margin, lat_max + margin], crs=ccrs.PlateCarree())
    
    # Plot the prediction contours
    contour = ax.contourf(LON, LAT, Z, levels=20, cmap=cmap, transform=ccrs.PlateCarree(), alpha=0.7)
    
    # Add colorbar
    cbar = plt.colorbar(contour, ax=ax, shrink=0.6, pad=0.05)
    cbar.set_label('Prediction Value')
    
    # Add gridlines
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    
    # Add title
    plt.title(title)
    
    plt.show()

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
    'reg_alpha': 0.01,
    'reg_lambda': 1
}

best_model_xgboost = XGBRegressor(**best_xgboost_params)
best_model_xgboost.fit(X, y)

# %% Visualize XGBoost predictions on California map
plot_2d_model_predictions(
    lambda x: best_model_xgboost.predict(x), 
    x_bounds=lat_range,
    y_bounds=lon_range,
    title="Best XGBoost model"
)

# %% [markdown]
# ## MPF Model
# Now, we'll train an MPF model and compare its performance with XGBoost

# %% Cross-validation for MPF hyperparameters (commented out)
"""
best_model, best_fr, best_params, best_error = random_hyperparam_search_parallel(
    X, y, n_splits=2, n_candidates=10, n_jobs=1, param_distributions= {
        "epochs": lambda: randint(2, 9).rvs(),       # 1 to 8 inclusive
        "n_iter": lambda: randint(5, 40).rvs(),      # 5 to 40 inclusive
        "split_try": lambda: randint(5, 21).rvs(),   # 5 to 20 inclusive
        "B": lambda: randint(10, 70).rvs(),          # 10 to 70 inclusive
        "colsample_bytree": lambda: 1.0,
        "identified": lambda: True
    })
print("Best hyperparameters for MPF:", best_params)
print("Best CV MSE for MPF:", best_error)
"""

# %% Train MPF model with best hyperparameters
best_params = {'epochs': 6, 'n_iter': 23, 'split_try': 6, 'B': 44, 'colsample_bytree': 1.0, 'identified': True}
print("Using MPF hyperparameters:", best_params)

best_model, best_fr = mpf_py.MPF.fit_boosted(
    X, y,
    **best_params
)

# %% Visualize MPF predictions on California map
plot_2d_model_predictions(
    lambda x: best_model.predict(x), 
    x_bounds=lat_range,
    y_bounds=lon_range,
    title="Best MPF model"
)

# %% Visualize MPF components
print("MPF Model Components:")
for i, tgf in enumerate(best_model.tree_grid_families):
    print(f"Tree Grid Family {i+1}:")
    combined_tg = mpf_py.TreeGrid(tgf.combined_tree_grid)
    combined_tg.plot_components()
    print(f"Grid Values: {combined_tg.grid_values}")

# %% [markdown]
# ## Model Comparison
# Compare performance metrics between XGBoost and MPF

# %% Compute and compare performance metrics
mse_xgboost = np.mean((y - best_model_xgboost.predict(X)) ** 2)
mse_mpf = np.mean((y - best_model.predict(X)) ** 2)

print(f"MSE of XGBoost: {mse_xgboost:.4f}")
print(f"MSE of MPF: {mse_mpf:.4f}")
print(f"Relative improvement: {((mse_xgboost - mse_mpf) / mse_xgboost) * 100:.2f}%")

# %%
