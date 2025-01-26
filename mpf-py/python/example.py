import numpy as np
import time  # Add import for time module

import mpf_py

import xgboost as xgb
import pandas as pd

"""
n = 5000
x1 = rnorm(n)
x2 = rnorm(n)
y = sin(x1) * cos(x2) + 2 * x1 - 2 * x2 + rnorm(n, sd=0.3)
dat = data.frame(y, x1, x2)
write.csv(dat, "dat.csv", row.names = FALSE)
"""
df = pd.read_csv("../../dat.csv")
x = df.iloc[:, 1:].values
y = df.iloc[:, 0].values

n_rows = x.shape[0]
x_train = x[:n_rows//2, :]
y_train = y[:n_rows//2]
x_test = x[n_rows//2:, :]
y_test = y[n_rows//2:]

start_time = time.time()  # Start timing for TreeGrid
tg, fr = mpf_py.TreeGrid.fit(x_train, y_train, n_iter = 100, split_try = 10, colsample_bytree = 1.0)
end_time = time.time()  # End timing for TreeGrid
print("TreeGrid fitting time: ", end_time - start_time, "seconds")

start_time = time.time()  # Start timing for MPF
mpf_bagged, mpf_fr = mpf_py.MPF.fit_bagged(x_train, y_train, epochs = 2, B = 50, n_iter = 100, split_try = 10, colsample_bytree = 1.0)
end_time = time.time()  # End timing for TreeGrid
print("MPFBagged fitting time: ", end_time - start_time, "seconds")

xgb_model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
xgb_model.fit(x_train, y_train)

xgb_error = np.mean((xgb_model.predict(x_test) - y_test) ** 2)
tg_error = np.mean((tg.predict(x_test) - y_test) ** 2)
mpf_error = np.mean((mpf_bagged.predict(x_test) - y_test) ** 2)
baseline_err = np.mean((y_test - np.mean(x_test)) ** 2)

print(f"Mean squared error (TG): {tg_error}, (MPF): {mpf_error}, (XGBoost): {xgb_error}, (Baseline): {baseline_err}")
