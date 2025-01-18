
import xgboost as xgb
import pandas as pd
import numpy as np

import mpf_py


n = 1000
x = np.random.randn(n, 2)
y = 2 * x[:, 0] * x[:, 1]


tg = mpf_py.TreeGrid.fit(x, y, n_iter = 100, split_try = 10, colsample_bytree = 1.0)
fr = tg.fit(x, y)
fr
y_mean = np.mean(y)

print("Mean squared error: ", fr.err)
print("Mean squared error (mean): ", np.mean((y_mean - y) ** 2))


# dat.csv

df = pd.read_csv("dat.csv")
x = df.iloc[:, 1:].values
y = df.iloc[:, 0].values


xgb_model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
xgb_model.fit(x, y)

print("Mean squared error (xgb): ", np.mean((xgb_model.predict(x) - y) ** 2))
