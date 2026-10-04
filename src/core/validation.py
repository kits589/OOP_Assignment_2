"""Input coercion and the decorators that guard model entry points.

Rather than hiding validation inside a template method, this build applies it
with decorators. ``fit`` on a concrete model therefore contains *only* the
parameter estimation; storing the training data and rejecting bad input happen
once, here, and are shared by every model in every mini-project.
"""
from __future__ import annotations

import functools
from typing import Callable, TypeVar

import numpy as np
from numpy.typing import ArrayLike

F = TypeVar("F", bound=Callable)


def as_series(values: ArrayLike, label: str = "values", *, minimum: int = 1) -> np.ndarray:
    """Coerce ``values`` to a 1-D float array, rejecting unusable input.

    Parameters
    ----------
    values : array-like
        The data to coerce.
    label : str
        Name used in error messages, so the caller knows which argument failed.
    minimum : int
        Fewest elements the caller can work with.

    Raises
    ------
    ValueError
        If the input is empty, shorter than ``minimum``, or holds NaN/inf.
    """
    arr = np.asarray(values, dtype=float).ravel()
    if arr.size < minimum:
        raise ValueError(f"{label} needs at least {minimum} value(s), got {arr.size}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{label} must be finite (no NaN or inf)")
    return arr


def validates_series(fit: F) -> F:
    """Wrap a model's ``fit`` so it receives clean arrays and keeps the state.

    The wrapped method is called as ``fit(self, y, t)`` with both arguments
    already coerced, and only has to estimate parameters. The wrapper stores
    ``self.y_`` / ``self.t_`` and returns ``self``, so calls chain:
    ``model.fit(y).predict(5)``.
    """

    @functools.wraps(fit)
    def wrapper(self, y: ArrayLike, t: ArrayLike | None = None):
        y_arr = as_series(y, "y", minimum=self.min_obs)
        t_arr = (np.arange(y_arr.size, dtype=float) if t is None
                 else as_series(t, "t", minimum=self.min_obs))
        if t_arr.size != y_arr.size:
            raise ValueError(
                f"t ({t_arr.size}) and y ({y_arr.size}) must have the same length")
        self.y_ = y_arr
        self.t_ = t_arr
        fit(self, y_arr, t_arr)
        return self

    return wrapper  # type: ignore[return-value]


def forecasts(predict: F) -> F:
    """Wrap a model's ``predict`` so the horizon and the fitted state are checked.

    The symmetric partner of :func:`validates_series`: between them, no concrete
    model repeats a guard clause. The wrapped method is called with a validated
    positive ``horizon`` and its return value is coerced to a float array.
    """

    @functools.wraps(predict)
    def wrapper(self, horizon: int) -> np.ndarray:
        if getattr(self, "y_", None) is None:
            raise RuntimeError(f"call fit() before predict() on {type(self).__name__}")
        return np.asarray(predict(self, positive_int(horizon, "horizon")), dtype=float)

    return wrapper  # type: ignore[return-value]


def requires_fit(method: F) -> F:
    """Wrap a method so it raises ``RuntimeError`` unless ``fit`` has run."""

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        if getattr(self, "y_", None) is None:
            raise RuntimeError(
                f"call fit() before {method.__name__}() on {type(self).__name__}")
        return method(self, *args, **kwargs)

    return wrapper  # type: ignore[return-value]


def positive_int(value: object, label: str) -> int:
    """Return ``value`` as a positive int or raise ``ValueError``."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{label} must be an integer, got {type(value).__name__}")
    if value < 1:
        raise ValueError(f"{label} must be >= 1, got {value}")
    return int(value)
