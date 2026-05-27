from enum import Enum
from typing import Callable

import numpy as np
from scipy.interpolate import interp1d, make_lsq_spline


class InterpolationKind(Enum):
    LUT = "lut"
    SPLINE = "spline"
    CONSTANT = "constant"


class Interpolant:
    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        kind: InterpolationKind | str = InterpolationKind.LUT,
        options: dict | None = None,
    ):
        if isinstance(kind, str):
            kind = InterpolationKind(kind)
        if options is None:
            options = {}

        _DISPATCH = {
            InterpolationKind.LUT: self._make_lut,
            InterpolationKind.SPLINE: self._make_spline,
            InterpolationKind.CONSTANT: self._make_constant,
        }

        sorted_indices = np.argsort(x)
        self._kind = kind
        self._x = x[sorted_indices]
        self._y = y[sorted_indices]
        self._options = options
        self._interpolator = _DISPATCH[self._kind]()
        return

    @property
    def x(self) -> np.ndarray:
        return self._x

    @property
    def y(self) -> np.ndarray:
        return self._y

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self._interpolator(x)

    def _make_spline(self) -> Callable:
        n = self._options.get("n", 10)
        k = self._options.get("k", 2)
        t = np.r_[
            [self._x.min()] * (k + 1),
            np.linspace(self._x.min(), self._x.max(), n)[1:-1],
            [self._x.max()] * (k + 1),
        ]
        return make_lsq_spline(self._x, self._y, t, k=k)

    def _make_lut(self) -> Callable:
        return interp1d(
            self._x,
            self._y,
            kind=self._options.get("kind", "linear"),
            fill_value="extrapolate",
        )

    def _make_constant(self) -> Callable:
        def wrapper(x_):
            return np.full(x_.shape, self._y.mean())

        return wrapper
