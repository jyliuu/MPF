
import numpy as np
from mpf_py._mpf_py import TreeGrid as _TreeGrid
from mpf_py._mpf_py import MPFBagged as _MPFBagged
from mpf_py._mpf_py import MPFGrown as _MPFGrown

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
    ) -> tuple["TreeGrid", "FitResult"]:
        tg, fr = _TreeGrid.fit(x, y, n_iter, split_try, colsample_bytree, identified)
        return cls(tg), fr


    def predict(self, x: np.typing.NDArray[np.float64]) -> np.typing.NDArray[np.float64]:
        return self._rust_instance.predict(x)
    
    def get_component(self, axis: int):
        intervals = self.intervals[axis]
        values = self.grid_values[axis]

        return [*zip(intervals, values)]

    def plot_components(self):
        try: 
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "Matplotlib is required to use the 'plot' function. "
                "Please install it using: pip install matplotlib"
            )

        plt.figure(figsize=(10, 6))
        # Define a list of high contrast colors
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']

        # For each axis, plot its intervals using a dedicated high contrast color
        for axis_idx, (intervals, values) in enumerate(zip(self.intervals, self.grid_values)):
            color = colors[axis_idx % len(colors)]
            for (x_start, x_end), y in zip(intervals, values):
            # Replace -infinity and +infinity if necessary:
                if x_start == -np.inf:
                    x_start = x_end - np.abs(x_end) * 0.2  # default replacement
                if x_end == np.inf:
                    x_end = x_start + np.abs(x_start) * 0.2  # default replacement
                plt.hlines(y, x_start, x_end, lw=2, color=color,
                    label=f"Axis {axis_idx}" if y == values[0] else None)

        plt.xlabel('X-axis')
        plt.ylabel('Value')
        plt.title(f'TreeGrid One-Dimensional Components, Scaling: {self.scaling}')
        plt.grid(True)
        # Only add one legend entry per axis.
        plt.legend()
        
        plt.show()


class MPF:

    class Bagged(PythonWrapperClassBase, RustClass=_MPFBagged):
        def __init__(self, mpf_bagged):
            super().__init__()
            self._rust_instance = mpf_bagged

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
            identified: bool = True,
            seed: int = 42
        ) -> tuple["MPF.Bagged", "FitResult"]:
            mpf_bagged, fr = _MPFBagged.fit(x, y, epochs, B, n_iter, split_try, colsample_bytree, identified, seed)
            instance = cls(mpf_bagged)
            instance._tree_grid_families_lst = [[TreeGrid(tg) for tg in tf.tree_grids] for tf in instance.tree_grid_families]
            return instance, fr

        def predict(self, x: np.typing.NDArray[np.float64]) -> np.typing.NDArray[np.float64]:
            return self._rust_instance.predict(x)


    class Grown(PythonWrapperClassBase, RustClass=_MPFGrown):
        def __init__(self, mpf_grown):
            super().__init__()
            self._rust_instance = mpf_grown
        
        @classmethod
        def fit(
            cls, 
            x: np.typing.NDArray[np.float64], 
            y: np.typing.NDArray[np.float64],
            n_iter: int,
            split_try: int, 
            colsample_bytree: float,
            seed: int = 42
        ) -> tuple["MPF.Grown", "FitResult"]:
            mpf_grown, fr = _MPFGrown.fit(x, y, n_iter, split_try, colsample_bytree, seed)
            instance = cls(mpf_grown)
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
    def fit_bagged(
        cls, 
        *args, **kwargs
    ) -> tuple["MPF.Bagged", "FitResult"]:
        mpf_bagged, fr = cls.Bagged.fit(*args, **kwargs)
        return cls.Bagged(mpf_bagged), fr
    
    @classmethod
    def fit_grown(
        cls, 
        *args, **kwargs
    ) -> tuple["MPF.Grown", "FitResult"]:
        mpf_grown, fr = cls.Grown.fit(*args, **kwargs)
        return cls.Grown(mpf_grown), fr
    