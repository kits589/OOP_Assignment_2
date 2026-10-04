"""The abstract forecasting model shared by mini-projects 1 and 5.

Design notes
------------
Concrete models declare themselves with class keywords::

    class LinearTrendModel(Forecaster, label="linear-trend", family="population"):

``__init_subclass__`` files each one into a per-family catalogue, so a caller
asks for "every population model" rather than maintaining a hand-written
factory dictionary that drifts out of step with the classes.

``fit`` and ``predict`` are fully abstract here: a subclass writes its own
``fit``, wraps it in :func:`~src.core.validation.validates_series` and writes
only the parameter estimation. The decorators supply the shared behaviour, so
there is no ``_fit``/``_predict`` hook pair to remember.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import ArrayLike

from src.core.metrics import ErrorReport
from src.core.validation import requires_fit


class Forecaster(ABC):
    """Base class for every univariate forecasting model in this repository.

    Attributes
    ----------
    min_obs : int
        Fewest observations the model can be fitted on; enforced by the
        ``validates_series`` decorator on each subclass's ``fit``.
    label : str
        Short human-readable name, used in catalogues, tables and legends.
    family : str
        Which mini-project the model belongs to.
    """

    min_obs: int = 2
    label: str = "forecaster"
    family: str = "general"

    #: family -> {label -> class}, populated by ``__init_subclass__``.
    _catalogue: dict[str, dict[str, type["Forecaster"]]] = {}

    def __init_subclass__(cls, *, label: str | None = None, family: str | None = None,
                          register: bool = True, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        if label is not None:
            cls.label = label
        if family is not None:
            cls.family = family
        if register:
            if cls.label in Forecaster._catalogue.get(cls.family, {}):
                raise TypeError(
                    f"label {cls.label!r} is already registered in family {cls.family!r}")
            Forecaster._catalogue.setdefault(cls.family, {})[cls.label] = cls

    def __init__(self) -> None:
        self.y_: np.ndarray | None = None
        self.t_: np.ndarray | None = None

    # ----- the interface the brief asks for ----------------------------
    @abstractmethod
    def fit(self, y: ArrayLike, t: ArrayLike | None = None) -> "Forecaster":
        """Estimate parameters from ``y``. Returns ``self`` so calls chain."""

    @abstractmethod
    def predict(self, horizon: int) -> np.ndarray:
        """Forecast the next ``horizon`` values beyond the training data."""

    # ----- shared behaviour -------------------------------------------
    @requires_fit
    def fitted(self) -> np.ndarray:
        """In-sample fitted values; NaN where the model does not define them."""
        return np.full_like(self.y_, np.nan, dtype=float)

    @requires_fit
    def residuals(self) -> np.ndarray:
        """Training residuals (actual - fitted), used for bootstrap intervals."""
        return self.y_ - self.fitted()

    def evaluate(self, actual: ArrayLike) -> ErrorReport:
        """Fit-then-score helper: forecast ``len(actual)`` steps and score them."""
        actual_arr = np.asarray(actual, dtype=float).ravel()
        return ErrorReport.score(self.label, actual_arr, self.predict(actual_arr.size))

    # ----- catalogue ---------------------------------------------------
    @classmethod
    def catalogue(cls, family: str) -> dict[str, type["Forecaster"]]:
        """Every registered model class in ``family``, keyed by label."""
        if family not in Forecaster._catalogue:
            known = sorted(Forecaster._catalogue)
            raise KeyError(f"no models registered for family {family!r}; known: {known}")
        return dict(Forecaster._catalogue[family])

    # ----- dunder methods ---------------------------------------------
    def __repr__(self) -> str:
        state = "fitted" if self.y_ is not None else "unfitted"
        return f"{type(self).__name__}(label={self.label!r}, {state})"

    def __len__(self) -> int:
        """Number of observations the model was trained on (0 if unfitted)."""
        return 0 if self.y_ is None else int(self.y_.size)
