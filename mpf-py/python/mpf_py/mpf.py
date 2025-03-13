import numpy as np
from mpf_py._mpf_py import TreeGrid as _TreeGrid
from mpf_py._mpf_py import MPFBoosted as _MPFBoosted
from mpf_py._mpf_py import FitResult
from mpf_py.wrapper import PythonWrapperClassBase


COMBINATION_STRATEGY_MAP = {
    "none": 0,
    "arith_mean": 1,
    "median": 2,
    "arith_geom_mean": 3,
    "geom_mean": 4,
}

IDENTIFICATION_STRATEGY_MAP = {
    "none": 0,
    "l1": 1,
    "l2": 2,
}

SPLIT_STRATEGY_MAP = {
    "random": 1,
    "interval_random": 2,
}

class TreeGrid(PythonWrapperClassBase, RustClass=_TreeGrid):
    
    def __init__(self, tg):
        super().__init__()
        self._rust_instance = tg

    @classmethod
    def fit(
        cls, 
        x: np.typing.NDArray[np.float64], 
        y: np.typing.NDArray[np.float64],
        n_iter: int,
        split_try: int, 
        colsample_bytree: float,
        identification_strategy: str = "l2",
        seed: int = 42
    ) -> tuple["TreeGrid", "FitResult"]:
        identification_strategy = int(IDENTIFICATION_STRATEGY_MAP[identification_strategy])
        tg, fr = _TreeGrid.fit(x, y, n_iter, split_try, colsample_bytree, identification_strategy, seed)
        return cls(tg), fr


    def predict(self, x: np.typing.NDArray[np.float64]) -> np.typing.NDArray[np.float64]:
        return self._rust_instance.predict(x)
    
    def get_component(self, axis: int):
        intervals = self.intervals[axis]
        values = self.grid_values[axis]

        return [*zip(intervals, values)]

    def plot_components(self, individual_plots: bool = False):
        try: 
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "Matplotlib is required to use the 'plot' function. "
                "Please install it using: pip install matplotlib"
            )

        # Define a list of high contrast colors
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']

        # Create single figure for combined plot upfront
        if not individual_plots:
            plt.figure(figsize=(10, 6))

        for axis_idx, (intervals, values) in enumerate(zip(self.intervals, self.grid_values)):
            if individual_plots:
                plt.figure(figsize=(10, 6))  # Create new figure per axis
            
            color = colors[axis_idx % len(colors)]
            
            # Create connected step function instead of separate horizontal lines
            x_points = []
            y_points = []
            
            # Use intervals directly without sorting
            for (x_start, x_end), y in zip(intervals, values):
                # Skip infinite intervals for plotting
                if x_start == float('-inf') or x_end == float('inf'):
                    continue
                    
                # Add points for step function
                x_points.extend([x_start, x_end])
                y_points.extend([y, y])
            
            if x_points:  # Only plot if we have valid points
                plt.step(x_points, y_points, where='post', lw=1, color=color, 
                         label=f"Axis {axis_idx}")

            if individual_plots:
                plt.xlabel('X-axis')
                plt.ylabel('Value')
                plt.title(f'TreeGrid Component for Axis {axis_idx}, Scaling: {self.scaling}')
                plt.grid(True)
                plt.legend()
                plt.show()  # Show individual plots immediately

        if not individual_plots:
            # Final setup for combined plot
            plt.xlabel('X-axis')
            plt.ylabel('Value')
            plt.title(f'TreeGrid One-Dimensional Components, Scaling: {self.scaling}')
            plt.grid(True)
            plt.legend()
            plt.show()


class MPF:

    class Boosted(PythonWrapperClassBase, RustClass=_MPFBoosted):
        def __init__(self, mpf_boosted):
            super().__init__()
            self._rust_instance = mpf_boosted

        @classmethod
        def fit(
            cls, 
            x: np.typing.NDArray[np.float64], 
            y: np.typing.NDArray[np.float64],
            epochs: int,
            n_trees: int,
            n_iter: int,
            split_try: int, 
            colsample_bytree: float,
            split_strategy: str = "random",
            identification_strategy: str = "l2",
            combination_strategy: str = "arith_mean",
            reproject_grid_values: bool = True,
            similarity_threshold: float = 0.0,
            seed: int = 42
        ) -> tuple["MPF.Boosted", "FitResult"]:
            split_strategy = SPLIT_STRATEGY_MAP[split_strategy]
            combination_strategy = int(COMBINATION_STRATEGY_MAP[combination_strategy])
            identification_strategy = int(IDENTIFICATION_STRATEGY_MAP[identification_strategy])
            mpf_boosted, fr = _MPFBoosted.fit(
                x, 
                y, 
                epochs, 
                n_trees,
                n_iter,
                split_try, 
                colsample_bytree, 
                split_strategy,
                identification_strategy,
                combination_strategy, 
                reproject_grid_values,
                similarity_threshold,
                seed
            )
            instance = cls(mpf_boosted)
            instance._tree_grid_families_lst = [[TreeGrid(tg) for tg in tf.tree_grids] for tf in instance.tree_grid_families]
            return instance, fr

        def predict(self, x: np.typing.NDArray[np.float64]) -> np.typing.NDArray[np.float64]:
            return self._rust_instance.predict(x)


    def __init__(self, mpf):
        self._mpf = mpf
        self.variant = self._mpf.__class__.__name__

        for attr in dir(self._mpf):
            if attr.startswith("__"):
                continue
            if not callable(getattr(self._mpf, attr)):
                setattr(self, attr, getattr(self._mpf, attr))

    @classmethod
    def fit_boosted(
        cls, 
        *args, **kwargs
    ) -> tuple["MPF.Boosted", "FitResult"]:
        mpf_boosted, fr = cls.Boosted.fit(*args, **kwargs)
        return cls.Boosted(mpf_boosted), fr
    