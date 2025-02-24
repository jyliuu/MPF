# %% [markdown]
# # Hyperparameter Search for MPF Bagged Model
# 
# In this cell we define helper functions for evaluating one hyperparameter candidate and performing a randomized search in parallel. We then use our random search function on training data.

# %%
import numpy as np
from sklearn.model_selection import KFold
from joblib import Parallel, delayed
import mpf_py
from utils import gen_data, true_model3, plot_2d_model_predictions  # Adjust import according to your project structure
from scipy.stats import randint

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

        # Fit the bagged model on the training split.
        print(params)
        model, fr = mpf_py.MPF.fit_bagged(
            x_tr, y_tr, **params
        )
        # Predict on the validation split.
        y_pred = model.predict(x_val)
        mse = np.mean((y_val - y_pred) ** 2)
        errors.append(mse)
    avg_error = np.mean(errors)
    return avg_error

def random_hyperparam_search_parallel(x_train, y_train, n_splits=2, n_candidates=50, n_jobs=-1, param_distributions=None):
    """
    Performs a random grid search cross-validation over hyperparameters for mpf_py.MPF.fit_bagged.
    
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
    for (avg_error, params) in zip(results, param_list):
        print(f"Params: {params} --> MSE={avg_error:.4f}")
        if avg_error < best_error:
            best_error = avg_error
            best_params = params
    
    # Refit the model on the full training set with the best hyperparameters.
    best_model, best_fr = mpf_py.MPF.fit_bagged(
        x_train, y_train,
        **best_params
    )
    
    return best_model, best_fr, best_params, best_error

# %% [markdown]
# ### Example Usage for MPF
# 
# Here we generate training data (using a hypothetical `gen_data` and `true_model3` from your utils) and run the random hyperparameter search.

# %%
# Generate data
x_train, y_train = gen_data(n=10000, seed=3, model=true_model3)
# %%
# Fit the best MPF model
best_model, best_fr, best_params, best_error = random_hyperparam_search_parallel(
    x_train, y_train, n_splits=2, n_candidates=50, n_jobs=-1, param_distributions= {
        "epochs": lambda: randint(2, 9).rvs(),       # 1 to 8 inclusive
        "n_iter": lambda: randint(5, 101).rvs(),         # 5 to 100 inclusive
        "split_try": lambda: randint(5, 21).rvs(),         # 5 to 20 inclusive
        "B": lambda: randint(10, 101).rvs(),         # 10 to 100 inclusive
        "colsample_bytree": lambda: 1.0,
        "identified": lambda: False
    })
print("Best hyperparameters for MPF:", best_params)
print("Best CV MSE for MPF:", best_error)

# %% [markdown]
# # Hyperparameter Search for XGBoost
# 
# In this section we use scikit‑learn’s RandomizedSearchCV with continuous and discrete ranges. We fix 2‑fold (or 4‑fold as set below) cross‑validation and use random sampling over the following ranges:
# 
# - `max_depth`: integers from 3 to 9,
# - `learning_rate`: continuous values in [0.001, 0.6],
# - `n_estimators`: integers from 200 to 800.
# 
# We then print the best hyperparameters and CV MSE, and retrieve the best model.

# %%
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint

# Define hyperparameter distributions.
param_distributions = {
    'max_depth': randint(3, 10),           # Integers from 3 to 9.
    'learning_rate': uniform(0.001, 0.599),  # Continuous values in [0.001, 0.6].
    'n_estimators': randint(200, 801)        # Integers from 200 to 800.
}

# Create an XGBRegressor.
xgb_model = XGBRegressor(random_state=42)

# Set up RandomizedSearchCV with 4-fold CV.
random_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_distributions,
    n_iter=1000,   # number of random candidates to try
    scoring='neg_mean_squared_error',
    cv=4,         # use 4-fold cross-validation (adjust as needed)
    n_jobs=-1,    # use all available cores
    verbose=1,
    random_state=42
)

# Run the randomized hyperparameter search.
random_search.fit(x_train, y_train)
print("Best xgboost hyperparameters:", random_search.best_params_)
print("Best xgboost CV MSE:", -random_search.best_score_)

best_model_xgboost = random_search.best_estimator_

# %% [markdown]
# # Comparing Models
# 
# Finally, we plot the true model, the best MPF model, and the best XGBoost model using a provided plotting function `plot_model_predictions`.

# %%
# Assuming plot_model_predictions is defined in your environment
plot_2d_model_predictions(true_model3, title="True model")
plot_2d_model_predictions(lambda x: best_model.predict(x), title="Best MPF model")
plot_2d_model_predictions(lambda x: best_model_xgboost.predict(x), title="Best XGBoost model")

# %%
# Fit the best MPF model with identified=True
best_params = {'epochs': 6,
 'n_iter': 33,
 'split_try': 16,
 'B': 63,
 'colsample_bytree': 1.0,
 'identified': True}
best_model_identified, best_fr_identified = mpf_py.MPF.fit_bagged(
    x_train, y_train,
    **best_params
)

# %%
# Plot the best MPF model with identified=True
plot_2d_model_predictions(lambda x: best_model_identified.predict(x), title="Best MPF model with identified=True")

# %%
tgf = best_model_identified.tree_grid_families[0]

for tg in tgf:
    tg.plot_components()
    plot_2d_model_predictions(lambda x: tg.predict(x), title="TG with identified=True")


# %%
