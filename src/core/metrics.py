"""Error reporting.

The three accuracy measures the brief asks for (MAE, RMSE, MAPE) are bundled
into one immutable :class:`ErrorReport` object instead of three loose
functions. Because the object knows how to compare itself, model selection is
just ``min(reports)`` and the comparison table is built by formatting them.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

#: Metric used to rank models when they are compared or sorted.
RANK_BY = "mape"


@dataclass(frozen=True)
class ErrorReport:
    """Out-of-sample accuracy of one model on one series.

    Actual and predicted values are stored as tuples so the report stays
    hashable and immutable: a result that has been computed cannot be edited
    afterwards, only recomputed.

    Parameters
    ----------
    model : str
        Label of the model being scored.
    actual, predicted : tuple of float
        Equal-length test values and forecasts.
    """

    model: str
    actual: tuple[float, ...]
    predicted: tuple[float, ...]

    @classmethod
    def score(cls, model: str, actual: ArrayLike, predicted: ArrayLike) -> "ErrorReport":
        """Build a report from any array-likes, validating them first."""
        a = np.asarray(actual, dtype=float).ravel()
        p = np.asarray(predicted, dtype=float).ravel()
        if a.size == 0:
            raise ValueError("actual must not be empty")
        if a.shape != p.shape:
            raise ValueError(
                f"actual ({a.size}) and predicted ({p.size}) must be the same length")
        if not (np.all(np.isfinite(a)) and np.all(np.isfinite(p))):
            raise ValueError("actual and predicted must be finite")
        return cls(model, tuple(a.tolist()), tuple(p.tolist()))

    # ----- the three measures the brief asks for -----------------------
    @cached_property
    def _residuals(self) -> np.ndarray:
        return np.asarray(self.actual) - np.asarray(self.predicted)

    @cached_property
    def mae(self) -> float:
        """Mean absolute error, in the units of the series."""
        return float(np.mean(np.abs(self._residuals)))

    @cached_property
    def rmse(self) -> float:
        """Root mean squared error; penalises large misses more than MAE does."""
        return float(np.sqrt(np.mean(self._residuals ** 2)))

    @cached_property
    def mape(self) -> float:
        """Mean absolute percentage error. Undefined if any actual value is zero."""
        a = np.asarray(self.actual)
        if np.any(a == 0):
            raise ValueError("MAPE is undefined when an actual value is zero")
        return float(np.mean(np.abs(self._residuals / a)) * 100)

    def as_dict(self) -> dict[str, float | str]:
        """The report as a plain mapping, ready for a DataFrame row."""
        return {"model": self.model, "MAE": self.mae,
                "RMSE": self.rmse, "MAPE_%": self.mape}

    # ----- dunder methods ---------------------------------------------
    def __lt__(self, other: "ErrorReport") -> bool:
        """Order reports by :data:`RANK_BY`, so ``min(reports)`` is the best model."""
        if not isinstance(other, ErrorReport):
            return NotImplemented
        return getattr(self, RANK_BY) < getattr(other, RANK_BY)

    def __len__(self) -> int:
        return len(self.actual)

    def __format__(self, spec: str) -> str:
        """``f"{report:.2f}"`` renders an aligned one-line summary."""
        spec = spec or ".3f"
        return (f"{self.model:<22} MAE {self.mae:{spec}}  "
                f"RMSE {self.rmse:{spec}}  MAPE {self.mape:{spec}}%")

    def __repr__(self) -> str:
        return (f"ErrorReport(model={self.model!r}, n={len(self)}, "
                f"mape={self.mape:.2f}%)")


def comparison_table(reports: list[ErrorReport]) -> pd.DataFrame:
    """One tidy table of every model's accuracy, best (lowest MAPE) first."""
    if not reports:
        raise ValueError("need at least one report to tabulate")
    frame = pd.DataFrame([r.as_dict() for r in sorted(reports)])
    return frame.set_index("model")
