
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
        colsample_bytree: float
    ) -> tuple["TreeGrid", "FitResult"]:
        tg, fr = _TreeGrid.fit(x, y, n_iter, split_try, colsample_bytree)
        return cls(tg), fr


    def predict(self, x: np.typing.NDArray[np.float64]) -> np.typing.NDArray[np.float64]:
        return self._rust_instance.predict(x)
    
    def get_component(self, axis: int):
        intervals = self.intervals[axis]
        values = self.grid_values[axis]

        return [*zip(intervals, values)]

    def plot(self, axis: int):

        component = self.get_component(axis)
        try: 
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "Matplotlib is required to use the 'plot' function. "
                "Please install it using: pip install matplotlib"
            )
    
        plt.figure(figsize=(10, 6))

        # Iterate through the intervals and corresponding values
        for (x_start, x_end), y in component:
            # Handle infinity
            if x_start == -np.inf:
                x_start = x_end - 2 
            if x_end == np.inf:
                x_end = x_start + 2

            plt.hlines(y, x_start, x_end, color='b', lw=2)  # Draw horizontal lines

        # Set labels and title
        plt.xlabel('X-axis')
        plt.ylabel('Values')
        plt.title(f'Component values on Axis: {axis + 1}')
        plt.grid(True)
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
            colsample_bytree: float
        ) -> tuple["MPF.Bagged", "FitResult"]:
            mpf_bagged, fr = _MPFBagged.fit(x, y, epochs, B, n_iter, split_try, colsample_bytree)
            instance = cls(mpf_bagged)
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
            colsample_bytree: float
        ) -> tuple["MPF.Grown", "FitResult"]:
            mpf_grown, fr = _MPFGrown.fit(x, y, n_iter, split_try, colsample_bytree)
            instance = cls(mpf_grown)
            return instance, fr


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
        return cls(cls.Bagged(mpf_bagged)), fr
    
    @classmethod
    def fit_grown(
        cls, 
        *args, **kwargs
    ) -> tuple["MPF.Grown", "FitResult"]:
        mpf_grown, fr = cls.Grown.fit(*args, **kwargs)
        return cls(cls.Grown(mpf_grown)), fr
    
    def predict(self, x: np.typing.NDArray[np.float64]) -> np.typing.NDArray[np.float64]:
        return self._mpf.predict(x)
