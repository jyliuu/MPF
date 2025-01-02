
import numpy as np
import mpf

n = 1000
x = np.random.randn(n, 2)
y = 2 * x[:, 0] * x[:, 1]


tg = mpf.RTGrid(n_iter = 100, split_try = 10, colsample_bytree = 1.0)
tg.fit(x, y)

y_mean = np.mean(y)
preds = tg.predict(x)

print("Mean squared error: ", np.mean((preds - y) ** 2))
print("Mean squared error (mean): ", np.mean((y_mean - y) ** 2))
