"""Mini-Project 5: Taxi Route Revenue, Pricing & Fleet Planner.

A matatu association runs 14-seater vehicles on three routes out of Kampala.
It wants to forecast demand, understand its fares and decide how many vehicles
to deploy.

Contents
--------
:class:`Route`
    One route's passenger counts and fare, with the descriptive statistics.
:class:`MarketModel`
    Linear supply and demand, solved as a 2x2 system for the equilibrium fare.
:class:`MovingAverageModel`, :class:`ExponentialSmoothingModel`,
:class:`RouteLinearTrend`, :class:`DriftModel`, :class:`SeasonalNaiveModel`
    Forecasting models, registered into the ``taxi`` family.
:class:`WalkForward`
    Rolling-origin backtesting and the grid search over the smoothing constant.
:class:`FleetPlanner`
    Turns a passenger forecast into a vehicle count.

``RouteLinearTrend`` **inherits** mini-project 1's linear trend rather than
reimplementing it: the mathematics is identical, only the model family differs.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Callable, Iterator, Sequence

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy import linalg

from src.core.forecaster import Forecaster
from src.core.metrics import ErrorReport, comparison_table
from src.core.validation import as_series, forecasts, positive_int, validates_series
from src.projects.p1_population import LinearTrendModel

__all__ = ["FAMILY", "ROUTES", "DriftModel", "ExponentialSmoothingModel",
           "FleetPlanner", "MarketModel", "MovingAverageModel", "Route",
           "RouteLinearTrend", "SeasonalNaiveModel", "WalkForward",
           "load_routes", "simulate_weekly_demand"]

FAMILY = "taxi"

#: Daily passenger counts over ten days, with the fare in UGX. The Ntinda
#: figures come from the brief; the other two are illustrative.
ROUTES: dict[str, dict[str, object]] = {
    "Kampala-Ntinda": {"passengers": (35, 40, 42, 50, 55, 60, 48, 52, 47, 45),
                       "fare": 2_000.0},
    "Kampala-Entebbe": {"passengers": (60, 58, 65, 70, 72, 80, 75, 68, 66, 64),
                        "fare": 5_000.0},
    "Kampala-Mukono": {"passengers": (45, 47, 50, 49, 55, 62, 58, 53, 51, 50),
                       "fare": 3_000.0},
}


class Route:
    """One route's daily passenger counts and its fare.

    Parameters
    ----------
    name : str
        Route name.
    passengers : array-like of float
        Daily passenger counts; non-negative and finite.
    fare : float
        Fare per passenger, in UGX; strictly positive.

    Raises
    ------
    ValueError
        On an empty name, an empty or negative passenger series, or a
        non-positive fare.
    """

    def __init__(self, name: str, passengers: ArrayLike, fare: float) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("name must be a non-empty string")
        counts = as_series(passengers, "passengers", minimum=1)
        if np.any(counts < 0):
            raise ValueError("passenger counts cannot be negative")
        if not np.isfinite(fare) or fare <= 0:
            raise ValueError("fare must be a positive, finite number")
        self.name = name.strip()
        self.passengers = counts
        self.fare = float(fare)

    # ----- dunder methods ---------------------------------------------
    def __repr__(self) -> str:
        return (f"Route({self.name!r}, days={len(self)}, "
                f"fare=UGX {self.fare:,.0f}, "
                f"total=UGX {self.total_revenue:,.0f})")

    def __len__(self) -> int:
        return int(self.passengers.size)

    def __iter__(self) -> Iterator[tuple[int, float]]:
        """Iterate as ``(day number, passengers)`` pairs, days numbered from 1."""
        return enumerate(self.passengers.tolist(), start=1)

    # ----- task 1 ---------------------------------------------------------
    @property
    def daily_revenue(self) -> np.ndarray:
        """Revenue for each day, in UGX."""
        return self.passengers * self.fare

    @property
    def total_revenue(self) -> float:
        """Revenue across the whole period, in UGX."""
        return float(self.daily_revenue.sum())

    def stats(self) -> dict[str, float]:
        """Mean, median, variance and sd of passengers, via ``statistics``."""
        data = self.passengers.tolist()
        if len(data) < 2:
            raise ValueError("need at least two days for a variance")
        return {"mean": statistics.mean(data), "median": statistics.median(data),
                "variance": statistics.variance(data), "stdev": statistics.stdev(data),
                "cv": statistics.stdev(data) / statistics.mean(data)}


@dataclass(frozen=True)
class MarketModel:
    """Linear supply and demand for one route, solved for the equilibrium.

    With demand :math:`Q_d = a - bP` and supply :math:`Q_s = c + dP`, writing
    both as :math:`Q + bP = a` and :math:`Q - dP = c` gives the 2x2 system

    .. math::
        \\begin{pmatrix} b & 1 \\\\ -d & 1 \\end{pmatrix}
        \\begin{pmatrix} P \\\\ Q \\end{pmatrix} =
        \\begin{pmatrix} a \\\\ c \\end{pmatrix}

    which ``scipy.linalg.solve`` handles directly.

    Parameters
    ----------
    demand_intercept, demand_slope : float
        ``a`` and ``b`` in :math:`Q_d = a - bP`. The brief gives 120 and 0.02.
    supply_intercept, supply_slope : float
        ``c`` and ``d`` in :math:`Q_s = c + dP`. The brief gives 10 and 0.03.
    """

    demand_intercept: float = 120.0
    demand_slope: float = 0.02
    supply_intercept: float = 10.0
    supply_slope: float = 0.03

    def __post_init__(self) -> None:
        if self.demand_slope <= 0 or self.supply_slope <= 0:
            raise ValueError("both slopes must be positive "
                             "(demand falls and supply rises with price)")
        if self.demand_intercept <= self.supply_intercept:
            raise ValueError("demand must exceed supply at a zero price, "
                             "or there is no market")

    def system(self) -> tuple[np.ndarray, np.ndarray]:
        """The coefficient matrix and right-hand side, in (P, Q) order."""
        A = np.array([[self.demand_slope, 1.0], [-self.supply_slope, 1.0]])
        b = np.array([self.demand_intercept, self.supply_intercept])
        return A, b

    @property
    def determinant(self) -> float:
        return float(np.linalg.det(self.system()[0]))

    @property
    def condition_number(self) -> float:
        return float(np.linalg.cond(self.system()[0]))

    def equilibrium(self) -> tuple[float, float]:
        """The clearing fare and quantity, ``(P*, Q*)``."""
        A, b = self.system()
        if np.isclose(self.determinant, 0.0):
            raise ValueError("supply and demand are parallel; no equilibrium exists")
        price, quantity = linalg.solve(A, b)
        return float(price), float(quantity)

    def quantity_demanded(self, price: float) -> float:
        return self.demand_intercept - self.demand_slope * price

    def quantity_supplied(self, price: float) -> float:
        return self.supply_intercept + self.supply_slope * price

    def gap_at(self, price: float) -> dict[str, float]:
        """Excess demand (positive) or excess supply (negative) at a fare."""
        if price < 0:
            raise ValueError("price cannot be negative")
        demanded = self.quantity_demanded(price)
        supplied = self.quantity_supplied(price)
        clearing, _ = self.equilibrium()
        return {"price": price, "demanded": demanded, "supplied": supplied,
                "excess_demand": demanded - supplied,
                "distance_from_equilibrium": price - clearing}


# ----------------------------------------------------------------------
# Task 3: the forecasting models
# ----------------------------------------------------------------------
class MovingAverageModel(Forecaster, label="moving-average", family=FAMILY):
    """Mean of the last ``window`` observations, projected flat.

    The association's existing method. It is robust to a single odd day but
    it cannot see a trend, and it lags whatever it is following by about half
    the window.
    """

    def __init__(self, window: int = 3) -> None:
        super().__init__()
        self.window = positive_int(window, "window")
        self.min_obs = self.window
        self.level_: float | None = None

    @validates_series
    def fit(self, y: np.ndarray, t: np.ndarray) -> None:
        self.level_ = float(np.mean(y[-self.window:]))

    @forecasts
    def predict(self, horizon: int) -> np.ndarray:
        return np.full(horizon, self.level_)

    def fitted(self) -> np.ndarray:
        if self.y_ is None:
            raise RuntimeError("call fit() before fitted() on MovingAverageModel")
        out = np.full_like(self.y_, np.nan)
        for i in range(self.window, self.y_.size):
            out[i] = self.y_[i - self.window:i].mean()
        return out


class ExponentialSmoothingModel(Forecaster, label="ses", family=FAMILY):
    """Simple exponential smoothing with a tunable ``alpha``.

    The level is updated as ``l_t = alpha * y_t + (1 - alpha) * l_{t-1}``, so
    ``alpha`` sets how quickly the model forgets. At ``alpha = 1`` it becomes
    the naive forecast; at ``alpha`` near zero it is a long-run average.
    """

    min_obs = 2

    def __init__(self, alpha: float = 0.5) -> None:
        super().__init__()
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = float(alpha)
        self.level_: float | None = None
        self.levels_: np.ndarray | None = None

    @validates_series
    def fit(self, y: np.ndarray, t: np.ndarray) -> None:
        levels = np.empty_like(y)
        levels[0] = y[0]
        for i in range(1, y.size):
            levels[i] = self.alpha * y[i] + (1 - self.alpha) * levels[i - 1]
        self.levels_ = levels
        self.level_ = float(levels[-1])

    @forecasts
    def predict(self, horizon: int) -> np.ndarray:
        return np.full(horizon, self.level_)

    def fitted(self) -> np.ndarray:
        if self.y_ is None:
            raise RuntimeError("call fit() before fitted() on ExponentialSmoothingModel")
        out = np.full_like(self.y_, np.nan)
        out[1:] = self.levels_[:-1]            # the one-step-ahead forecasts
        return out

    @property
    def label(self) -> str:                    # type: ignore[override]
        return f"ses(a={self.alpha:.2f})"


class RouteLinearTrend(LinearTrendModel, label="linear-trend", family=FAMILY):
    """Mini-project 1's linear trend, reused unchanged on passenger counts.

    Nothing is reimplemented here. ``np.polyfit`` on a time index is the same
    operation whether the series is a district population or a daily passenger
    count, so the only thing this subclass changes is which family the model
    is registered under.
    """


class DriftModel(Forecaster, label="drift", family=FAMILY):
    """Extends the straight line drawn between the first and last observation.

    Cheaper than a least-squares trend and less sensitive to the middle of the
    series, which is useful when only the recent direction matters.
    """

    min_obs = 2

    def __init__(self) -> None:
        super().__init__()
        self.drift_: float | None = None

    @validates_series
    def fit(self, y: np.ndarray, t: np.ndarray) -> None:
        self.drift_ = float((y[-1] - y[0]) / (y.size - 1))

    @forecasts
    def predict(self, horizon: int) -> np.ndarray:
        return self.y_[-1] + self.drift_ * np.arange(1, horizon + 1)

    def fitted(self) -> np.ndarray:
        if self.y_ is None:
            raise RuntimeError("call fit() before fitted() on DriftModel")
        out = np.full_like(self.y_, np.nan)
        out[1:] = self.y_[:-1] + self.drift_
        return out


class SeasonalNaiveModel(Forecaster, label="seasonal-naive", family=FAMILY):
    """Repeats the value from ``period`` steps ago.

    Useless on ten days of data, and the obvious model once a weekly rhythm
    exists -- which is what the extension demonstrates.
    """

    def __init__(self, period: int = 7) -> None:
        super().__init__()
        self.period = positive_int(period, "period")
        self.min_obs = self.period

    @validates_series
    def fit(self, y: np.ndarray, t: np.ndarray) -> None:
        pass                                   # nothing to estimate

    @forecasts
    def predict(self, horizon: int) -> np.ndarray:
        season = self.y_[-self.period:]
        return np.array([season[i % self.period] for i in range(horizon)])

    def fitted(self) -> np.ndarray:
        if self.y_ is None:
            raise RuntimeError("call fit() before fitted() on SeasonalNaiveModel")
        out = np.full_like(self.y_, np.nan)
        out[self.period:] = self.y_[:-self.period]
        return out


# ----------------------------------------------------------------------
# Task 4: rolling-origin backtesting
# ----------------------------------------------------------------------
class WalkForward:
    """Rolling-origin backtesting over a stretch of a series.

    At each origin the model is refitted on everything observed up to that
    point and asked for one step, which is how the forecast would actually
    have been made at the time. Scoring on a fixed hold-out would instead let
    a model see data it could not have had.

    Parameters
    ----------
    y : array-like
        The full series.
    first_day : int
        One-based day number of the first forecast. The brief asks for days
        4-10, so this defaults to 4.
    """

    def __init__(self, y: ArrayLike, first_day: int = 4) -> None:
        self.y = as_series(y, "y", minimum=2)
        self.first_day = positive_int(first_day, "first_day")
        if not 1 < self.first_day <= self.y.size:
            raise ValueError(
                f"first_day must be in (1, {self.y.size}], got {self.first_day}")

    def __repr__(self) -> str:
        return (f"WalkForward(n={self.y.size}, days "
                f"{self.first_day}-{self.y.size})")

    def __len__(self) -> int:
        """How many one-step forecasts the backtest produces."""
        return int(self.y.size - self.first_day + 1)

    def forecasts(self, build: Callable[[], Forecaster]) -> pd.DataFrame:
        """One-step forecasts and actuals, day by day."""
        rows = []
        for origin in range(self.first_day - 1, self.y.size):
            history = self.y[:origin]
            model = build()
            if history.size < model.min_obs:
                rows.append({"day": origin + 1, "actual": self.y[origin],
                             "forecast": np.nan})
                continue
            model.fit(history)
            rows.append({"day": origin + 1, "actual": float(self.y[origin]),
                         "forecast": float(model.predict(1)[0])})
        return pd.DataFrame(rows).set_index("day")

    def score(self, build: Callable[[], Forecaster]) -> ErrorReport:
        """Accuracy of one model over the backtest window."""
        table = self.forecasts(build).dropna()
        if table.empty:
            raise ValueError("the model could not be fitted at any origin")
        return ErrorReport.score(build().label, table["actual"], table["forecast"])

    def compare(self, builders: dict[str, Callable[[], Forecaster]]) -> pd.DataFrame:
        """Every model's backtest accuracy, best first."""
        if not builders:
            raise ValueError("need at least one model to compare")
        reports = []
        for name, build in builders.items():
            report = self.score(build)
            reports.append(ErrorReport(name, report.actual, report.predicted))
        return comparison_table(reports)

    def tune_alpha(self, grid: ArrayLike | None = None) -> tuple[float, pd.Series]:
        """Grid-search the smoothing constant on backtest MAE."""
        candidates = (np.round(np.arange(0.1, 1.01, 0.05), 2) if grid is None
                      else as_series(grid, "grid", minimum=1))
        if np.any((candidates <= 0) | (candidates > 1)):
            raise ValueError("every alpha must be in (0, 1]")
        scores = pd.Series(
            {float(a): self.score(lambda a=a: ExponentialSmoothingModel(a)).mae
             for a in candidates}, name="MAE")
        scores.index.name = "alpha"
        return float(scores.idxmin()), scores


# ----------------------------------------------------------------------
# Task 6: fleet planning
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class FleetPlanner:
    """Turns a passenger forecast into a vehicle count.

    Parameters
    ----------
    seats : int
        Seats per vehicle. The brief says 14.
    trips_per_day : int
        One-way trips each vehicle makes. The brief says 8.
    buffer : float
        Spare capacity as a fraction. The brief says 15%.

    Notes
    -----
    Rounding is **up**, with a floor of one vehicle on any route that carries
    passengers at all. Rounding to nearest would leave a route short on a day
    it was slightly busier than forecast, and a stranded passenger costs the
    association more than an under-filled seat does.
    """

    seats: int = 14
    trips_per_day: int = 8
    buffer: float = 0.15

    def __post_init__(self) -> None:
        if self.seats < 1 or self.trips_per_day < 1:
            raise ValueError("seats and trips_per_day must be at least 1")
        if not 0 <= self.buffer < 5:
            raise ValueError("buffer must be a fraction in [0, 5)")

    @property
    def capacity(self) -> int:
        """Passengers one vehicle can carry in a day."""
        return self.seats * self.trips_per_day

    def vehicles_for(self, passengers: float) -> int:
        """Vehicles needed to carry a forecast number of passengers."""
        if passengers < 0:
            raise ValueError("passenger count cannot be negative")
        if passengers == 0:
            return 0
        return max(1, int(np.ceil(passengers * (1 + self.buffer) / self.capacity)))

    def plan(self, forecasts: dict[str, float]) -> pd.DataFrame:
        """A vehicle count for each route."""
        rows = [{"route": route,
                 "forecast passengers": value,
                 "with buffer": value * (1 + self.buffer),
                 "vehicles": self.vehicles_for(value)}
                for route, value in forecasts.items()]
        return pd.DataFrame(rows).set_index("route")


def load_routes(names: Sequence[str] | None = None) -> list[Route]:
    """Build :class:`Route` objects from the brief's data."""
    wanted = list(ROUTES) if names is None else list(names)
    missing = [n for n in wanted if n not in ROUTES]
    if missing:
        raise ValueError(f"unknown route(s) {missing}; have {sorted(ROUTES)}")
    return [Route(n, ROUTES[n]["passengers"], ROUTES[n]["fare"]) for n in wanted]


def simulate_weekly_demand(days: int = 60, base: float = 50.0, seed: int = 7,
                           weekday_effect: Sequence[float] | None = None,
                           noise: float = 0.07) -> np.ndarray:
    """Extension: synthetic daily demand with a weekly rhythm.

    Busier on Fridays, quieter on Sundays, as the brief describes, so that a
    seasonal-naive model has something to find.

    Parameters
    ----------
    days : int
        How many days to simulate.
    base : float
        Mean passengers per day before the weekly effect.
    seed : int
        Seed for ``np.random.default_rng``.
    weekday_effect : sequence of float, optional
        Seven multipliers, Monday first. Defaults to a plausible matatu week.
    noise : float
        Standard deviation of the multiplicative noise.
    """
    n = positive_int(days, "days")
    effect = np.asarray(
        (1.00, 1.02, 1.05, 1.10, 1.28, 0.95, 0.72) if weekday_effect is None
        else weekday_effect, dtype=float)
    if effect.size != 7:
        raise ValueError("weekday_effect needs exactly seven multipliers")
    if base <= 0 or noise < 0:
        raise ValueError("base must be positive and noise non-negative")
    rng = np.random.default_rng(seed)
    pattern = effect[np.arange(n) % 7]
    return np.round(base * pattern * rng.normal(1.0, noise, size=n), 0)
