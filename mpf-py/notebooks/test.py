from collections import namedtuple
import mpf_py  # Assuming mpf_py.MPF is available
from utils import gen_data, plot_2d_model_predictions, plot_intervals_with_values_all, true_model3, l2_identification  # Adjust import according to your project structure
import numpy as np

x_train, y_train = gen_data(n=5000, seed=3, model=true_model3)

# Fit the best MPF model with identified=True
best_params = {'epochs': 3,
 'n_iter': 28,
 'split_try': 16,
 'B': 40,
 'colsample_bytree': 1.0,
 'identified': True}
best_model_identified, _ = mpf_py.MPF.fit_bagged(
    x_train, y_train,
    **best_params
)

plot_2d_model_predictions(lambda x: best_model_identified.predict(x), title="TGF with identified=True")
combined_tgs = []

for tgf in best_model_identified.tree_grid_families:
    combined_tg = mpf_py.TreeGrid(tgf.combined_tree_grid)
    plot_2d_model_predictions(lambda x: tgf.predict(x), title="TGF with identified=True")
    plot_2d_model_predictions(lambda x: combined_tg.predict(x), title="Combined TG with identified=True")
    combined_tg.plot_components()
    combined_tgs.append(combined_tg)
    print(f"GV: {combined_tg.grid_values}")

combined_pred_fun = lambda x: sum(tg.predict(x) for tg in combined_tgs)
plot_2d_model_predictions(combined_pred_fun, title="Combined TGF with identified=True")
