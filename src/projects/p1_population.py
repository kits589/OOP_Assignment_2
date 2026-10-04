"""Mini-Project 1: UBOS District Population Forecaster.

Answers the planning unit's question -- how many extra primary-school
classrooms will each district need by 2029? -- by validating three forecasting
models on a held-out period and planning from the winner.

Contents
--------
:class:`DistrictPopulation`
    Validated container for one district's series, with both stdlib and NumPy
    descriptive statistics so the ``ddof`` difference can be demonstrated.
:class:`LinearTrendModel`, :class:`CAGRModel`, :class:`FibonacciRatioModel`
    The three models the brief specifies, registered into the ``population``
    family of :class:`~src.core.forecaster.Forecaster`.
:class:`ModelSelector`
    Trains every registered model on 2015-2021, scores it on 2022-2024 and
    picks a winner per district.
:class:`ClassroomPlanner`
    Turns forecast population growth into a classroom requirement.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass
from enum import Enum
from typing import Iterator

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from src.core.forecaster import Forecaster
from src.core.metrics import ErrorReport, comparison_table
from src.core.validation import as_series, forecasts, positive_int, validates_series

FAMILY = "population"

#: Calendar years covered by the illustrative series.
YEARS: list[int] = list(range(2015, 2025))

#: Population estimates in thousands. Kampala, Wakiso and Gulu are the figures
#: given in the brief. Jinja and Mbale are the two additional districts chosen
#: here: both are long-established municipalities growing more slowly than the
#: Kampala metropolitan belt, which gives the comparison a useful contrast.
DISTRICTS: dict[str, list[float]] = {
    "Kampala": [1200, 1250, 1300, 1350, 1420, 1500, 1580, 1650, 1720, 1800],
    "Wakiso": [950, 1000, 1070, 1150, 1220, 1300, 1390, 1480, 1570, 1670],
    "Gulu": [320, 330, 345, 360, 375, 390, 410, 430, 455, 480],
    "Jinja": [300, 312, 321, 335, 347, 357, 372, 386, 398, 414],
    "Mbale": [520, 531, 546, 557, 573, 585, 601, 614, 631, 645],
}

#: Boundary of the validation split: train on 2015-LAST_TRAIN_YEAR.
LAST_TRAIN_YEAR = 2021


class Phase(str, Enum):
    """Which side of the validation split a year falls on."""

    TRAIN = "train"
    TEST = "test"

    def __str__(self) -> str:
        return self.value


class DistrictPopulation:
    """One district's population series, in thousands of people.

    Parameters
    ----------
    name : str
        District name; must be a non-empty string.
    years : array-like of int
        Calendar years, strictly increasing.
    people : array-like of float
        Population in thousands; finite and non-negative, same length as
        ``years``.

    Raises
    ------
    ValueError
        On empty input, mismatched lengths, negative or non-finite population,
        or years that are not strictly increasing.
    """

    def __init__(self, name: str, years: ArrayLike, people: ArrayLike) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("name must be a non-empty string")
        pop = as_series(people, "people", minimum=1)
        yrs = np.asarray(years, dtype=int).ravel()
        if yrs.size != pop.size:
            raise ValueError(
                f"years ({yrs.size}) and people ({pop.size}) differ in length")
        if np.any(pop < 0):
            raise ValueError("population must be non-negative")
        if yrs.size > 1 and np.any(np.diff(yrs) <= 0):
            raise ValueError("years must be strictly increasing")
        self.name = name.strip()
        self.years = yrs
        self.people = pop

    # ----- dunder methods ---------------------------------------------
    def __repr__(self) -> str:
        return (f"DistrictPopulation({self.name!r}, {self.years[0]}-{self.years[-1]}, "
                f"n={len(self)}, latest={self.people[-1]:,.0f}k)")

    def __len__(self) -> int:
        return int(self.years.size)

    def __contains__(self, year: object) -> bool:
        """``2019 in district`` -- is that year covered by the series?"""
        return bool(np.any(self.years == year))

    # ----- task 2: the same statistics computed two ways ---------------
    def describe_stdlib(self) -> dict[str, float]:
        """Descriptive statistics from the ``statistics`` module.

        ``statistics.variance`` is the *sample* variance: it divides by n-1.
        """
        data = self.people.tolist()
        if len(data) < 2:
            raise ValueError("variance needs at least two observations")
        return {"mean": statistics.mean(data), "median": statistics.median(data),
                "variance": statistics.variance(data), "stdev": statistics.stdev(data)}

    def describe_numpy(self, ddof: int = 0) -> dict[str, float]:
        """The same statistics from NumPy.

        ``np.var`` divides by ``n - ddof``, and ``ddof`` defaults to 0, so the
        default is the *population* variance. Pass ``ddof=1`` to reproduce
        :meth:`describe_stdlib`.
        """
        p = self.people
        return {"mean": float(np.mean(p)), "median": float(np.median(p)),
                "variance": float(np.var(p, ddof=ddof)),
                "stdev": float(np.std(p, ddof=ddof))}

    # ----- task 3: growth ---------------------------------------------
    def yoy_growth(self) -> np.ndarray:
        """Year-on-year growth as fractions; length ``len(self) - 1``."""
        if len(self) < 2:
            raise ValueError("need at least two years to measure growth")
        if np.any(self.people[:-1] == 0):
            raise ValueError("growth is undefined after a zero population")
        return self.people[1:] / self.people[:-1] - 1

    def compound_growth(self) -> float:
        """Compound annual growth rate between the first and last year."""
        span = int(self.years[-1] - self.years[0])
        if span <= 0:
            raise ValueError("CAGR needs at least two distinct years")
        if self.people[0] == 0:
            raise ValueError("CAGR is undefined when the first value is zero")
        return float((self.people[-1] / self.people[0]) ** (1 / span) - 1)

    # ----- task 5: validation split ------------------------------------
    def split_at(self, last_train_year: int) -> tuple["DistrictPopulation",
                                                      "DistrictPopulation"]:
        """Split into (train, test); ``last_train_year`` stays in train."""
        mask = self.years <= last_train_year
        if not mask.any() or mask.all():
            raise ValueError(
                f"splitting at {last_train_year} leaves one side empty "
                f"(series covers {self.years[0]}-{self.years[-1]})")
        train = DistrictPopulation(self.name, self.years[mask], self.people[mask])
        test = DistrictPopulation(self.name, self.years[~mask], self.people[~mask])
        return train, test

    def phase_of(self, year: int, last_train_year: int = LAST_TRAIN_YEAR) -> Phase:
        """Label a year as train or test."""
        return Phase.TRAIN if year <= last_train_year else Phase.TEST


# ----------------------------------------------------------------------
# Task 4: the three forecasting models
# ----------------------------------------------------------------------
class LinearTrendModel(Forecaster, label="linear-trend", family=FAMILY):
    """Straight line through the series, fitted with ``np.polyfit`` (degree 1).

    Appropriate when a district adds roughly the same number of people every
    year. It cannot bend, so it under-forecasts a compounding series.
    """

    min_obs = 2

    def __init__(self) -> None:
        super().__init__()
        self.slope_: float | None = None
        self.intercept_: float | None = None

    @validates_series
    def fit(self, y: np.ndarray, t: np.ndarray) -> None:
        self.slope_, self.intercept_ = (float(c) for c in np.polyfit(t, y, 1))

    @forecasts
    def predict(self, horizon: int) -> np.ndarray:
        future = self.t_[-1] + np.arange(1, horizon + 1, dtype=float)
        return self.slope_ * future + self.intercept_

    def fitted(self) -> np.ndarray:
        if self.y_ is None:
            raise RuntimeError("call fit() before fitted() on LinearTrendModel")
        return self.slope_ * self.t_ + self.intercept_


class CAGRModel(Forecaster, label="cagr", family=FAMILY):
    """Constant-percentage growth, anchored on the first and last observation.

    Population tends to compound rather than add, so this is the natural
    counterpart to the linear trend.
    """

    min_obs = 2

    def __init__(self) -> None:
        super().__init__()
        self.rate_: float | None = None

    @validates_series
    def fit(self, y: np.ndarray, t: np.ndarray) -> None:
        span = float(t[-1] - t[0])
        if span <= 0:
            raise ValueError("CAGR needs a positive time span")
        if y[0] <= 0:
            raise ValueError("CAGR needs a positive first observation")
        self.rate_ = float((y[-1] / y[0]) ** (1 / span) - 1)

    @forecasts
    def predict(self, horizon: int) -> np.ndarray:
        steps = np.arange(1, horizon + 1, dtype=float)
        return self.y_[-1] * (1 + self.rate_) ** steps

    def fitted(self) -> np.ndarray:
        if self.y_ is None:
            raise RuntimeError("call fit() before fitted() on CAGRModel")
        return self.y_[0] * (1 + self.rate_) ** (self.t_ - self.t_[0])


class FibonacciRatioModel(Forecaster, label="fibonacci-ratio", family=FAMILY):
    """Scales the last observation by successive Fibonacci ratios.

    Carried over from the previous cohort of this assignment so that it can be
    assessed rather than assumed. ``start_index`` chooses where in the
    Fibonacci sequence the ratios are taken from; this build starts at index
    10, well inside the region where F(n+1)/F(n) has converged on the golden
    ratio, so the model implies roughly 61.8% growth a year.
    """

    min_obs = 1

    def __init__(self, start_index: int = 10) -> None:
        super().__init__()
        self.start_index = positive_int(start_index, "start_index")
        self.ratios_: np.ndarray | None = None

    @validates_series
    def fit(self, y: np.ndarray, t: np.ndarray) -> None:
        self.ratios_ = None  # the ratios depend on the horizon; built in predict()

    @forecasts
    def predict(self, horizon: int) -> np.ndarray:
        seq = fibonacci(self.start_index + horizon + 2)
        window = seq[self.start_index: self.start_index + horizon + 1]
        self.ratios_ = window[1:] / window[:-1]
        return self.y_[-1] * np.cumprod(self.ratios_)


def fibonacci(n: int) -> np.ndarray:
    """First ``n`` Fibonacci numbers as floats, starting 1, 1, 2, 3, 5, ..."""
    n = positive_int(n, "n")
    seq = np.empty(n, dtype=float)
    a, b = 1.0, 1.0
    for i in range(n):
        seq[i], a, b = a, b, a + b
    return seq


# ----------------------------------------------------------------------
# Tasks 5-6: validation, selection and forecasting
# ----------------------------------------------------------------------
class ModelSelector:
    """Trains every registered population model and picks the best per district.

    The models are discovered through
    :meth:`~src.core.forecaster.Forecaster.catalogue`, so adding a new subclass
    puts it into the comparison automatically -- there is no factory dictionary
    to keep in step with the classes.

    Parameters
    ----------
    last_train_year : int
        Final year kept in the training set.
    """

    def __init__(self, last_train_year: int = LAST_TRAIN_YEAR) -> None:
        self.last_train_year = int(last_train_year)
        self.models = Forecaster.catalogue(FAMILY)

    def __repr__(self) -> str:
        return (f"ModelSelector(split={self.last_train_year}, "
                f"models={sorted(self.models)})")

    def __len__(self) -> int:
        return len(self.models)

    def __iter__(self) -> Iterator[str]:
        return iter(sorted(self.models))

    def reports(self, district: DistrictPopulation) -> list[ErrorReport]:
        """Score every model on the held-out years for one district."""
        train, test = district.split_at(self.last_train_year)
        out = []
        for label, cls in sorted(self.models.items()):
            model = cls()
            model.fit(train.people, train.years.astype(float))
            out.append(ErrorReport.score(label, test.people, model.predict(len(test))))
        return out

    def table(self, district: DistrictPopulation) -> pd.DataFrame:
        """MAE/RMSE/MAPE for every model on one district, best first."""
        return comparison_table(self.reports(district))

    def best(self, district: DistrictPopulation) -> tuple[str, ErrorReport]:
        """The lowest-error model label for one district, and its report."""
        winner = min(self.reports(district))
        return winner.model, winner

    def fit_best(self, district: DistrictPopulation) -> Forecaster:
        """A model of the winning type, refitted on the district's full series."""
        label, _ = self.best(district)
        model = self.models[label]()
        return model.fit(district.people, district.years.astype(float))


def forecast_frame(district: DistrictPopulation, model: Forecaster,
                   horizon: int = 5) -> pd.DataFrame:
    """Tidy table of forecast years and values for one fitted model."""
    horizon = positive_int(horizon, "horizon")
    years = district.years[-1] + np.arange(1, horizon + 1)
    return pd.DataFrame({"year": years, "forecast": model.predict(horizon)}
                        ).set_index("year")


def variance_comparison(district: DistrictPopulation,
                        forecast: ArrayLike) -> dict[str, float]:
    """Task 6: variance of the observed series against the forecast series.

    A deterministic model produces a path with no noise in it, so the two
    variances are not measuring the same thing: one mixes the trend with
    year-to-year scatter, the other reflects only the trend.
    """
    f = as_series(forecast, "forecast", minimum=2)
    actual_var = float(np.var(district.people, ddof=1))
    forecast_var = float(np.var(f, ddof=1))
    return {"actual_variance": actual_var, "forecast_variance": forecast_var,
            "ratio": forecast_var / actual_var if actual_var else float("inf")}


# ----------------------------------------------------------------------
# Extension: bootstrap prediction intervals
# ----------------------------------------------------------------------
def bootstrap_intervals(model: Forecaster, horizon: int = 5, n_boot: int = 1000,
                        level: float = 0.95, seed: int = 7) -> dict[str, np.ndarray]:
    """95% prediction intervals by resampling the training residuals.

    The point forecast carries no uncertainty of its own, so the spread is
    taken from how badly the model missed the years it was trained on: draw
    residuals with replacement, add them to the forecast path, and read the
    percentiles off the resulting distribution.

    Parameters
    ----------
    model : Forecaster
        An already-fitted model.
    horizon, n_boot : int
        Forecast length and number of resamples (the brief asks for >= 1,000).
    level : float
        Coverage, strictly between 0 and 1.
    seed : int
        Seed for ``np.random.default_rng``, so the interval is reproducible.
    """
    horizon = positive_int(horizon, "horizon")
    n_boot = positive_int(n_boot, "n_boot")
    if not 0 < level < 1:
        raise ValueError("level must be strictly between 0 and 1")
    point = model.predict(horizon)
    resid = model.residuals()
    resid = resid[np.isfinite(resid)]
    if resid.size == 0:
        raise ValueError("model has no finite residuals to resample")
    rng = np.random.default_rng(seed)
    draws = rng.choice(resid, size=(n_boot, horizon), replace=True)
    paths = point + draws
    tail = (1 - level) / 2 * 100
    return {"point": point,
            "lower": np.percentile(paths, tail, axis=0),
            "upper": np.percentile(paths, 100 - tail, axis=0)}


# ----------------------------------------------------------------------
# Task 8: planning output
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class ClassroomPlanner:
    """Converts population growth into a primary-classroom requirement.

    Parameters
    ----------
    age_share : float
        Fraction of the population of primary-school age (the brief says 18%).
    class_size : int
        Pupils per classroom (the brief says 53).
    """

    age_share: float = 0.18
    class_size: int = 53

    def __post_init__(self) -> None:
        if not 0 < self.age_share <= 1:
            raise ValueError("age_share must be in (0, 1]")
        if self.class_size < 1:
            raise ValueError("class_size must be at least 1")

    def pupils(self, people_thousands: float) -> float:
        """Primary-age pupils implied by a population given in thousands."""
        if people_thousands < 0:
            raise ValueError("population must be non-negative")
        return people_thousands * 1_000 * self.age_share

    def classrooms(self, people_thousands: float) -> int:
        """Classrooms needed to seat a whole population, rounded up."""
        return int(np.ceil(self.pupils(people_thousands) / self.class_size))

    def additional(self, now_thousands: float, later_thousands: float) -> int:
        """Extra classrooms needed for the growth between two populations."""
        growth = later_thousands - now_thousands
        if growth <= 0:
            return 0
        return int(np.ceil(self.pupils(growth) / self.class_size))


def load_districts(names: list[str] | None = None) -> list[DistrictPopulation]:
    """Build :class:`DistrictPopulation` objects from the illustrative data."""
    chosen = list(DISTRICTS) if names is None else list(names)
    unknown = [n for n in chosen if n not in DISTRICTS]
    if unknown:
        raise ValueError(f"unknown district(s): {unknown}; known: {sorted(DISTRICTS)}")
    return [DistrictPopulation(n, YEARS, DISTRICTS[n]) for n in chosen]
