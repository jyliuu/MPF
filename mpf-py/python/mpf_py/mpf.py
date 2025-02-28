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

        # For each axis, plot its intervals using a dedicated high contrast color
        for axis_idx, (intervals, values) in enumerate(zip(self.intervals, self.grid_values)):
            if individual_plots:
                plt.figure(figsize=(10, 6))
            color = colors[axis_idx % len(colors)]
            for (x_start, x_end), y in zip(intervals, values):
                # Replace -infinity and +infinity if necessary:
                if x_start == -np.inf:
                    print(f"x_start: {x_start}, x_end: {x_end}")
                    x_start = x_end - np.abs(x_end) * 0.2  # default replacement
                    continue
                elif x_end == np.inf:
                    print(f"x_start: {x_start}, x_end: {x_end}")
                    x_end = x_start + np.abs(x_start) * 0.2  # default replacement
                    continue
                plt.hlines(y, x_start, x_end, lw=2, color=color,
                           label=f"Axis {axis_idx}" if y == values[0] else None)

            if individual_plots:
                plt.xlabel('X-axis')
                plt.ylabel('Value')
                plt.title(f'TreeGrid Component for Axis {axis_idx}, Scaling: {self.scaling}')
                plt.grid(True)
                plt.legend()
                plt.show()

        if not individual_plots:
            plt.figure(figsize=(10, 6))
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
            identification: str,
            identified: bool = True,
            seed: int = 42
        ) -> tuple["MPF.Boosted", "FitResult"]:
            
            identification_strategy = IDENTIFICATION_STRATEGY_MAP[identification] or identified # short circuit trick
            mpf_boosted, fr = _MPFBoosted.fit(x, y, epochs, B, n_iter, split_try, colsample_bytree, identification_strategy, seed)
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
    