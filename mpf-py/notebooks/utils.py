import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from joblib import Parallel, delayed
import mpf_py
from scipy.stats import randint
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np

def bump_function(center, radius, scale=1):
    def bump(X):
        # Vectorized calculation with safety margin
        squared_dist = np.sum((X - center)**2, axis=1)
        denominator = 1 - (squared_dist / (radius**2))
        
        # Apply safety margin to avoid division by zero
        denominator = np.clip(denominator, 1e-9, None)  # Prevent negative/zero values
        
        return scale * np.exp(-1 / denominator)
    return bump

bump_neg4_1 = bump_function(np.array([-4, 1]), 1, scale=1)
bump_1_0 =  bump_function(np.array([1, 0]), 0.5, scale=3)
bump_neg2_neg2 = bump_function(np.array([-2, -2]), 1, scale=2)

true_model = lambda x: 2*x[:,1] * x[:,0] + x[:,0]
true_model2 = lambda x: np.sin(x[:,0]) * np.cos(x[:,1]) + 2 * x[:,0] - 2 * x[:,1]
true_model3 = lambda X: np.exp(np.sin(X[:,0]) * np.cos(X[:,1])) + X[:,0]
true_model4 = lambda X: bump_neg4_1(X) + bump_1_0(X) + bump_neg2_neg2(X) + bump_neg2_neg2(X)


def gen_data(n=5000, seed=1, unif = True, noise=True, model=true_model3):
    np.random.seed(seed)
    mean = np.array([0, 0])  # Mean vector
    cov = np.array([[1, 0], [0, 1]])  # Covariance matrix
    if unif:
        X = np.random.uniform(-4, 4, size=(n, 2))
    else:
        X = np.random.multivariate_normal(mean, cov, size=n)
    if noise:
        y = model(X) + np.random.normal(scale=0.5, size=n)
    else:
        y = model(X)
    return X, y

def plot_2d_model_predictions_map(
        model, 
        x_bounds=(-4,4), y_bounds=(-4,4),
        grid_points=100, cmap='viridis',
        title="Model Predictions", data=None, cbar_max=5
    ):
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
        data (tuple): Optional tuple of (X,y) where X contains coordinates and y contains target values
        cbar_max (float): The maximum value for the colorbar
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
    predictions = np.clip(model(grid), a_min=-3, a_max=cbar_max)

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
    cs = ax.contourf(LON, LAT, Z, levels=20, cmap=cmap, transform=ccrs.PlateCarree(), alpha=0.7, vmax=cbar_max)
    
    # Add colorbar
    cbar = plt.colorbar(cs, ax=ax, shrink=0.6, pad=0.05)
    cbar.set_label('Prediction Value')
    cbar.set_ticks(np.linspace(Z.min(), cbar_max, 5))
    
    # Add gridlines
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    
    # Add title
    plt.title(title)

    # If data is provided, plot the actual points
    if data is not None:
        X, y = data
        scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, 
                           transform=ccrs.PlateCarree(), alpha=1, s=3)
    
    plt.show()



def plot_2d_model_map_errors(model, data, grid_points=100, cmap='coolwarm',
                           title="Model Errors", cbar_max=None):
    """
    Visualizes mean prediction errors (y - ŷ) in geographic grid cells with marginal error histograms.
    """
    X, y = data
    predictions = model(X)
    errors = y - predictions
    squared_errors = errors ** 2

    # Create bin edges
    lon_min, lon_max = (X[:, 0].min(), X[:, 0].max())
    lat_min, lat_max = (X[:, 1].min(), X[:, 1].max())
    print(lon_min, lon_max, lat_min, lat_max)
    lon_edges = np.linspace(lon_min, lon_max, grid_points + 1)
    lat_edges = np.linspace(lat_min, lat_max, grid_points + 1)

    # Calculate 2D histograms
    H_counts, _, _ = np.histogram2d(X[:, 0], X[:, 1], bins=[lon_edges, lat_edges])  # Swapped order of edges
    H_errors, _, _ = np.histogram2d(X[:, 0], X[:, 1], bins=[lon_edges, lat_edges], weights=errors)
    H_errors_squared, _, _ = np.histogram2d(X[:, 0], X[:, 1], bins=[lon_edges, lat_edges], weights=squared_errors)
        # Compute mean errors and handle empty bins
    with np.errstate(divide='ignore', invalid='ignore'):
        H_mean = H_errors / H_counts
    H_mean[H_counts == 0] = np.nan

    # Calculate marginal sums
    lon_sse = np.nansum(H_errors_squared, axis=1)  # Sum along latitude
    lat_sse = np.nansum(H_errors_squared, axis=0)  # Sum along longitude

    # Create plot with marginal histograms
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2, width_ratios=[4, 1], height_ratios=[4, 1],
                        hspace=0.05, wspace=0.05)
    
    # Main map axis
    ax = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.STATES)
    
    margin = 0.5
    ax.set_extent([-124.35, -114.31, 32.54, 41.95], crs=ccrs.PlateCarree())

    # Plot error grid
    vmax = cbar_max or np.nanmax(np.abs(H_mean))
    vmin = -vmax if cbar_max is None else -cbar_max
    mesh = ax.pcolormesh(lon_edges, lat_edges, H_mean.T, 
                        cmap=cmap, vmin=vmin, vmax=vmax,
                        transform=ccrs.PlateCarree(), alpha=0.7)

    # Add colorbar
    cbar = plt.colorbar(mesh, ax=ax, shrink=0.6, pad=0.05)
    cbar.set_label('Mean Error')
    
    # Add grid lines and title
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    plt.title(title)

    # X-axis histogram (longitude SSE)
    ax_histx = fig.add_subplot(gs[1, 0])
    bar_widths = np.diff(lon_edges)
    ax_histx.bar(lon_edges[:-1], lon_sse, width=bar_widths, 
                align='edge', color=plt.cm.get_cmap(cmap)(0.5), alpha=0.7)
    ax_histx.set_ylabel('Longitudinal SSE')
    ax_histx.grid(True, alpha=0.3)
    ax_histx.set_xlim(lon_min, lon_max)

    # Y-axis histogram (latitude SSE)
    ax_histy = fig.add_subplot(gs[0, 1])
    bar_heights = np.diff(lat_edges)
    ax_histy.barh(lat_edges[:-1], lat_sse, height=bar_heights,
                 align='edge', color=plt.cm.get_cmap(cmap)(0.5), alpha=0.7)
    ax_histy.set_xlabel('Latitudinal SSE')
    ax_histy.grid(True, alpha=0.3)
    ax_histy.set_ylim(lat_min, lat_max)

    plt.show()


def plot_2d_model_predictions(model, x_bounds=(-4,4), y_bounds=(-4,4), grid_points=100, cmap='seismic',
                           title="Model Predictions"):
    """
    Constructs a grid of (x1, x2) coordinates and uses the provided ML model to generate predictions.
    The predictions are then visualized as a colored grid, with the color scale centered at zero.

    Parameters:
        model (callable): A function or machine learning model that accepts an input array of shape (N, 2)
                          and returns prediction values as an array of shape (N,).
        x_bounds (tuple): A tuple (x_min, x_max) defining the minimum and maximum x1 values.
        y_bounds (tuple): A tuple (y_min, y_max) defining the minimum and maximum x2 values.
        grid_points (int): The number of points to generate along each axis.
        cmap (str): The colormap to use when displaying the predictions.
        title (str): The title for the plot.
    """
    # Create linearly spaced values for x1 and x2
    x_min, x_max = x_bounds
    y_min, y_max = y_bounds

    x1_vals = np.linspace(x_min, x_max, grid_points)
    x2_vals = np.linspace(y_min, y_max, grid_points)

    # Create a mesh grid of points
    X1, X2 = np.meshgrid(x1_vals, x2_vals)

    # Flatten the grid to form (x1, x2) pairs
    grid = np.column_stack([X1.ravel(), X2.ravel()])

    # Obtain predictions from the ML model (model should accept an array of shape (N, 2))
    predictions = model(grid)

    # Reshape predictions back into the grid format
    Z = predictions.reshape(X1.shape)

    # Compute symmetric normalization for predictions
    abs_max = max(abs(Z.min()), abs(Z.max()))

    # Create the plot with pcolormesh, applying the norm so that 0 is centered
    plt.figure(figsize=(8, 6))
    plt.contourf(X1, X2, Z, levels=20, cmap=cmap)

    plt.colorbar(label='Prediction')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title(title)
    plt.show()


def plot_intervals_with_values_all(axes_intervals, axes_values, title='Treegrid One-Dimensional Components'):
    """
    Plots horizontal lines for each axis's intervals and corresponding constant values on the same plot.
    Each axis's intervals are plotted in a different color.

    Parameters:
    - axes_intervals: List of lists of tuples. Each inner list corresponds to one axis and
                      contains tuples (start, end) for that axis.
    - axes_values: List of lists of floats. Each inner list corresponds to the grid values
                   (i.e. the constant value) on the corresponding intervals for one axis.
    """
    # Define a list of colors (or use a colormap if more axes are present)
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']

    plt.figure(figsize=(10, 6))

    # For each axis, plot its intervals
    for axis_idx, (intervals, values) in enumerate(zip(axes_intervals, axes_values)):
        # Choose a color (cycling if there are more axes than colors)
        color = colors[axis_idx % len(colors)]
        for (x_start, x_end), y in zip(intervals, values):
            # Replace -infinity and +infinity if necessary:
            if x_start == -np.inf:
                x_start = x_end - np.abs(x_end) * 0.2  # default replacement
            if x_end == np.inf:
                x_end = x_start + np.abs(x_start) * 0.2  # default replacement
            plt.hlines(y, x_start, x_end, color=color, lw=2,
                       label=f"Axis {axis_idx}" if y == values[0] else None)


    plt.xlabel('X-axis')
    plt.ylabel('Value')
    plt.title(title)
    plt.grid(True)
    # Only add one legend entry per axis.
    plt.legend()
    plt.show()



def l2_identification(tg, x):
    weights = [[] for _ in range(len(tg.grid_values))]
    for i in range(len(tg.intervals)):
        for j in range(len(tg.intervals[i])):
            interval = tg.intervals[i][j]
            num_obs_in_interval = np.sum((x[:, i] >= interval[0]) & (x[:, i] < interval[1]))
            weights[i].append(num_obs_in_interval)

    grid_values = tg.grid_values.copy()
    scaling = 1
    for dim in range(len(grid_values)):
        grid_values[dim] = np.array(grid_values[dim])
        weights_sum = np.sum(weights[dim])
        norm = np.sqrt(np.sum(weights[dim] * grid_values[dim] ** 2)/weights_sum)
        grid_values[dim] = grid_values[dim] / norm
        scaling *= norm
    return scaling, grid_values



def evaluate_params(x_train, y_train, kf, **params):
    """
    Evaluates one hyperparameter combination using cross-validation.
    Returns a tuple (epochs, B, n_iter, split_try, avg_mse).
    """
    errors = []
    for train_index, val_index in kf.split(x_train):
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_val = x_train[val_index]
        y_val = y_train[val_index]

        # Fit the boosted model on the training split.
        print(params)
        try:
            model, fr = mpf_py.MPF.fit_boosted(
                x_tr, y_tr, **params
            )
        except Exception as e:
            print(e)
            return np.inf
        # Predict on the validation split.
        y_pred = model.predict(x_val)
        mse = np.mean((y_val - y_pred) ** 2)
        errors.append(mse)
    avg_error = np.mean(errors)
    return avg_error

def random_hyperparam_search_parallel(x_train, y_train, n_splits=2, n_candidates=50, n_jobs=-1, param_distributions=None):
    """
    Performs a random grid search cross-validation over hyperparameters for mpf_py.MPF.fit_boosted.
    
    Parameters:
        x_train, y_train: Training data.
        n_splits: Number of cross-validation splits.
        n_candidates: Number of hyperparameter combinations to evaluate.
        n_jobs: Number of parallel jobs.
        param_distributions: A dictionary specifying the hyperparameter distributions.
                           Each key is a hyperparameter name, and each value is a distribution
                           (e.g., scipy.stats.randint, scipy.stats.uniform).
    
    Returns the best model, additional fitted result (fr), best parameters, and best error.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Default parameter distributions if none are provided.
    if param_distributions is None:
        param_distributions = {
            "epochs": lambda: randint(2, 9).rvs(),       # 1 to 8 inclusive
            "n_iter": lambda: randint(5, 101).rvs(),         # 5 to 100 inclusive
            "split_try": lambda: randint(5, 21).rvs(),         # 5 to 20 inclusive
            "B": lambda: randint(10, 101).rvs(),         # 10 to 100 inclusive
            "colsample_bytree": lambda: 1.0,
            "identified": lambda: True
        }
    
    # Generate n_candidates random hyperparameter combinations.
    param_list = []
    for _ in range(n_candidates):
        params = {
            name: dist() for name, dist in param_distributions.items()
        }
        param_list.append(params)
        
    # Evaluate candidates in parallel.
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(evaluate_params)(x_train, y_train, kf, **params)
        for params in param_list
    )
    
    best_params = None
    best_error = np.inf
    best_model = None
    for (avg_error, params) in zip(results, param_list):
        print(f"Params: {params} --> MSE={avg_error:.4f}")
        if avg_error < best_error:
            best_error = avg_error
            best_params = params
    
    # Refit the model on the full training set with the best hyperparameters.
    best_model, best_fr = mpf_py.MPF.fit_boosted(
        x_train, y_train,
        **best_params
    )
    
    return best_model, None, best_params, best_error
