import numpy as np
from mpf_py._mpf_py import TreeGrid as _TreeGrid
from mpf_py._mpf_py import MPFBoosted as _MPFBoosted
from mpf_py._mpf_py import FitResult
from mpf_py.wrapper import PythonWrapperClassBase



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
        identified: bool = True,
        seed: int = 42
    ) -> tuple["TreeGrid", "FitResult"]:
        tg, fr = _TreeGrid.fit(x, y, n_iter, split_try, colsample_bytree, identified, seed)
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
            for (x_start, x_end), y in zip(intervals, values):
                # Replace -infinity and +infinity if necessary:
                if x_start == -np.inf or x_end == np.inf:
                    continue
                plt.hlines(y, x_start, x_end, lw=2, color=color,
                           label=f"Axis {axis_idx}" if x_start == float('-inf') else None)

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


IDENTIFICATION_STRATEGY_MAP = {
    "none": 0,
    "l2_arith_mean": 1,
    "l2_median": 2,
}

SPLIT_STRATEGY_MAP = {
    "random": 1,
    "interval_random": 2,
}

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
            B: int,
            n_iter: int,
            split_try: int, 
            colsample_bytree: float,
            split_strategy: str = "random",
            identification: str = "l2_arith_mean",
            identified: bool = False,
            seed: int = 42
        ) -> tuple["MPF.Boosted", "FitResult"]:
            split_strategy = SPLIT_STRATEGY_MAP[split_strategy]
            identification_strategy = int(IDENTIFICATION_STRATEGY_MAP[identification] or identified) # short circuit trick
            
            mpf_boosted, fr = _MPFBoosted.fit(
                x, y, 
                epochs, B, n_iter, split_try, 
                colsample_bytree, 
                split_strategy,
                identification_strategy, 
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
    