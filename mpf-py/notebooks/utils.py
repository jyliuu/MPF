import numpy as np
import matplotlib.pyplot as plt


true_model = lambda x: 2*x[:,1] * x[:,0] + x[:,0]
true_model2 = lambda x: np.sin(x[:,0]) * np.cos(x[:,1]) + 2 * x[:,0] - 2 * x[:,1]
true_model3 = lambda X: np.exp(np.sin(X[:,0]) * np.cos(X[:,1])) + X[:,0]


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
