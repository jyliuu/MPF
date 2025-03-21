import numpy as np
import pytest
from mpf_py import TreeGrid, MPF

def gen_data(n=5000, seed=1):
    np.random.seed(seed)
    X = np.random.uniform(-5, 5, size=(n, 2))
    # y = 3*np.sin(3 * X[:,0])*np.cos(5*X[:,1]) + np.random.normal(scale=0.5, size=n)
    y = np.exp(np.sin(X[:, 0]) * np.cos(X[:, 1])) + X[:, 0] + np.random.normal(scale=0.5, size=n)
    return X, y

@pytest.fixture(scope="module")
def training_data():
    X, y = gen_data(seed=1)
    return X, y.ravel()

@pytest.fixture(scope="module")
def test_data():
    X_test, y_test = gen_data(seed=2)
    return X_test, y_test.ravel()


def test_tree_grid_fit(training_data, test_data):
    X, y = training_data
    X_test, y_test = test_data

    # Train the MPF estimator
    tg, fr = TreeGrid.fit(X, y, n_iter=100, split_try=15, colsample_bytree=1.0)

    print("Fit result: ", fr)
    # MPF predictions and loss
    y_pred = tg.predict(X_test)
    tg_test_loss = np.mean((y_test - y_pred) ** 2)

    # Baseline: loss of predicting the mean of y_test
    baseline = np.mean(y_test)
    mean_test_loss = np.mean((y_test - baseline) ** 2)

    # Print losses for debugging (optional)
    print(f"Tree grid test loss: {tg_test_loss}")
    print(f"Mean test loss: {mean_test_loss}")

    print(f"Tree grid scaling: {tg.scaling}")

    assert tg_test_loss < mean_test_loss, "Tree grid should beat the mean predictor"


def test_mpf_boosted_fit(training_data, test_data):
    X, y = training_data
    X_test, y_test = test_data

    # Train the MPF estimator
    mpf, fr = MPF.Boosted.fit(X, y, epochs=3, n_trees=37, n_iter=30, split_try=16, colsample_bytree=1.0)

    print("Fit result: ", fr)
    # MPF predictions and loss
    y_pred = mpf.predict(X_test)
    mpf_test_loss = np.mean((y_test - y_pred) ** 2)

    # Baseline: loss of predicting the mean of y_test
    baseline = np.mean(y_test)
    mean_test_loss = np.mean((y_test - baseline) ** 2)

    # Print losses for debugging (optional)
    print(f"MPF test loss: {mpf_test_loss}")
    print(f"Mean test loss: {mean_test_loss}")

    assert mpf_test_loss < mean_test_loss, "MPF should beat the mean predictor"
