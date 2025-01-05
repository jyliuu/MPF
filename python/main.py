
import numpy as np
import mpf

n = 1000
x = np.random.randn(n, 2)
y = 2 * x[:, 0] * x[:, 1]


tg = mpf.TreeGrid(n_iter = 100, split_try = 10, colsample_bytree = 1.0)
fr = tg.fit(x, y)
fr
y_mean = np.mean(y)

print("Mean squared error: ", fr.err)
print("Mean squared error (mean): ", np.mean((y_mean - y) ** 2))
