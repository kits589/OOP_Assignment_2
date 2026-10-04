"""Mini-Project 3: Lake Victoria Fish Stock & Export Risk Model.

A fish-export cooperative in Jinja needs to know whether its harvesting rate
is sustainable, and how risky its revenue is. The previous version of this
question modelled the stock with Fibonacci numbers and flagged risk when
revenue variance exceeded 50,000; both are replaced here, and the notebook
explains why each was indefensible.

Contents
--------
:class:`FishStock`
    Discrete logistic growth with harvesting.
:class:`PriceModel`
    A seeded random walk for the weekly export price, with reflecting bounds.
:class:`RiskAssessor`
    Classifies revenue risk by coefficient of variation, and measures the
    tail with both Value-at-Risk and expected shortfall.
:class:`ExportScenario`
    Composes the three above into one runnable scenario.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterator, Sequence

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from src.core.validation import positive_int

__all__ = ["ExportScenario", "FishStock", "PriceModel", "RiskAssessor", "RiskClass",
           "StockPath", "closed_season_weeks", "fibonacci_stock"]

#: Kilograms in a tonne: the stock is in tonnes, the price is per kilogram.
KG_PER_TONNE = 1_000


class RiskClass(Enum):
    """Revenue risk bands, defined on the coefficient of variation."""

    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"

    def __str__(self) -> str:
        return self.value


def fibonacci_stock(count: int = 15) -> np.ndarray:
    """The baseline the brief asks for: ``count`` Fibonacci numbers as "stock".

    Kept only so the notebook can show what is wrong with it.
    """
    n = positive_int(count, "count")
    out = np.empty(n, dtype=float)
    a, b = 1.0, 1.0
    for i in range(n):
        out[i], a, b = a, b, a + b
    return out


@dataclass
class StockPath:
    """One simulated year: the stock week by week, and what was taken."""

    biomass: np.ndarray
    harvest: np.ndarray
    harvest_rate: float
    open_weeks: np.ndarray

    def __len__(self) -> int:
        return int(self.harvest.size)

    def __iter__(self) -> Iterator[tuple[float, float]]:
        """Iterate as ``(biomass at the start of the week, harvest taken)``."""
        return zip(self.biomass[:-1].tolist(), self.harvest.tolist())

    def __repr__(self) -> str:
        return (f"StockPath(h={self.harvest_rate:.2f}, weeks={len(self)}, "
                f"final={self.final_biomass:,.0f}t, "
                f"caught={self.total_harvest:,.0f}t)")

    @property
    def final_biomass(self) -> float:
        """Standing stock at the end of the simulation, in tonnes."""
        return float(self.biomass[-1])

    @property
    def total_harvest(self) -> float:
        """Everything landed over the simulation, in tonnes."""
        return float(self.harvest.sum())

    @property
    def closed_weeks(self) -> int:
        return int((~self.open_weeks).sum())

    def revenue(self, price_ugx_per_kg: ArrayLike) -> np.ndarray:
        """Weekly revenue in UGX: harvest in tonnes x 1,000 kg x price per kg."""
        price = np.asarray(price_ugx_per_kg, dtype=float).ravel()
        if price.size != len(self):
            raise ValueError(
                f"need one price per week: {price.size} prices, {len(self)} weeks")
        return self.harvest * KG_PER_TONNE * price


class FishStock:
    """Discrete logistic growth with proportional harvesting.

    The stock evolves as

    .. math:: N_{t+1} = N_t + r N_t (1 - N_t / K) - h N_t

    which replaces the Fibonacci baseline with a model that has a carrying
    capacity, so the biomass cannot grow without bound.

    Parameters
    ----------
    r : float
        Intrinsic weekly growth rate. The brief gives 0.4.
    K : float
        Carrying capacity in tonnes. The brief gives 10,000.
    N0 : float
        Starting biomass in tonnes. The brief gives 4,000.
    harvest_rate : float
        Fraction of the standing stock taken each open week.

    Raises
    ------
    ValueError
        If any parameter is non-positive, non-finite, or if the starting
        biomass exceeds the carrying capacity.
    """

    def __init__(self, r: float = 0.4, K: float = 10_000.0, N0: float = 4_000.0,
                 harvest_rate: float = 0.10) -> None:
        for name, value in (("r", r), ("K", K), ("N0", N0)):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a positive, finite number")
        if not np.isfinite(harvest_rate) or not 0 <= harvest_rate < 1:
            raise ValueError("harvest_rate must be in [0, 1)")
        if N0 > K:
            raise ValueError(f"N0 ({N0:,.0f}) cannot exceed K ({K:,.0f})")
        self.r = float(r)
        self.K = float(K)
        self.N0 = float(N0)
        self.harvest_rate = float(harvest_rate)

    def __repr__(self) -> str:
        return (f"FishStock(r={self.r}, K={self.K:,.0f}, N0={self.N0:,.0f}, "
                f"h={self.harvest_rate})")

    # ----- reference points ------------------------------------------------
    @property
    def msy(self) -> float:
        """Maximum sustainable yield, rK/4 tonnes per week."""
        return self.r * self.K / 4

    @property
    def h_msy(self) -> float:
        """The harvest rate that achieves MSY: h = r/2."""
        return self.r / 2

    def equilibrium_biomass(self, harvest_rate: float | None = None) -> float:
        """Stock the fishery settles at under a constant harvest rate.

        Setting :math:`N_{t+1} = N_t` gives :math:`N^* = K(1 - h/r)`, which
        collapses to zero once ``h`` reaches ``r``: the stock is fished out.
        """
        h = self.harvest_rate if harvest_rate is None else float(harvest_rate)
        return max(0.0, self.K * (1 - h / self.r))

    def equilibrium_yield(self, harvest_rate: float | None = None) -> float:
        """Sustainable catch per week at that equilibrium, ``h N*``."""
        h = self.harvest_rate if harvest_rate is None else float(harvest_rate)
        return h * self.equilibrium_biomass(h)

    # ----- simulation --------------------------------------------------------
    def step(self, biomass: float, harvest_rate: float | None = None) -> float:
        """Advance the stock by one week."""
        if biomass < 0:
            raise ValueError("biomass cannot be negative")
        h = self.harvest_rate if harvest_rate is None else float(harvest_rate)
        grown = biomass + self.r * biomass * (1 - biomass / self.K) - h * biomass
        return max(0.0, grown)

    def simulate(self, weeks: int = 52, open_weeks: ArrayLike | None = None,
                 harvest_rate: float | None = None) -> StockPath:
        """Run the model forward.

        Parameters
        ----------
        weeks : int
            How many weeks to simulate.
        open_weeks : array-like of bool, optional
            Which weeks the fishery is open. ``None`` means every week; this
            is how the closed-season extension is expressed.
        harvest_rate : float, optional
            Overrides the rate set on the instance.
        """
        n = positive_int(weeks, "weeks")
        h = self.harvest_rate if harvest_rate is None else float(harvest_rate)
        if not 0 <= h < 1:
            raise ValueError("harvest_rate must be in [0, 1)")
        if open_weeks is None:
            is_open = np.ones(n, dtype=bool)
        else:
            is_open = np.asarray(open_weeks, dtype=bool).ravel()
            if is_open.size != n:
                raise ValueError(
                    f"open_weeks must have {n} entries, got {is_open.size}")

        biomass = np.empty(n + 1, dtype=float)
        harvest = np.empty(n, dtype=float)
        biomass[0] = self.N0
        for week in range(n):
            rate = h if is_open[week] else 0.0
            harvest[week] = rate * biomass[week]
            biomass[week + 1] = self.step(biomass[week], rate)
        return StockPath(biomass, harvest, h, is_open)


class PriceModel:
    """Weekly export price in UGX/kg, as a bounded random walk.

    Each week the price takes a normal step. Rather than clipping at the
    bounds -- which piles probability mass onto the boundary and quietly
    biases long runs -- the walk **reflects** off them, which keeps the
    interior distribution undistorted.

    Parameters
    ----------
    start : float
        Opening price, UGX/kg. The brief gives 12,000.
    lower, upper : float
        Bounds on the price. The brief suggests 9,000 and 16,000.
    step_sd : float
        Standard deviation of the weekly step, in UGX.
    seed : int
        Seed for ``np.random.default_rng``.
    """

    def __init__(self, start: float = 12_000.0, lower: float = 9_000.0,
                 upper: float = 16_000.0, step_sd: float = 400.0,
                 seed: int = 7) -> None:
        if not lower < upper:
            raise ValueError(f"need lower < upper, got {lower} and {upper}")
        if not lower <= start <= upper:
            raise ValueError(f"start {start} must lie within [{lower}, {upper}]")
        if step_sd <= 0 or not np.isfinite(step_sd):
            raise ValueError("step_sd must be a positive, finite number")
        self.start = float(start)
        self.lower = float(lower)
        self.upper = float(upper)
        self.step_sd = float(step_sd)
        self.seed = int(seed)

    def __repr__(self) -> str:
        return (f"PriceModel(start={self.start:,.0f}, "
                f"bounds=({self.lower:,.0f}, {self.upper:,.0f}), "
                f"sd={self.step_sd:,.0f}, seed={self.seed})")

    def _reflect(self, values: np.ndarray) -> np.ndarray:
        """Fold values back inside the bounds, repeatedly if necessary."""
        span = self.upper - self.lower
        folded = np.abs(values - self.lower) % (2 * span)
        folded = np.where(folded > span, 2 * span - folded, folded)
        return self.lower + folded

    def paths(self, weeks: int = 52, n_paths: int = 1,
              seed: int | None = None) -> np.ndarray:
        """Simulate ``n_paths`` price paths of ``weeks`` steps each.

        Returns an ``(n_paths, weeks)`` array.
        """
        n = positive_int(weeks, "weeks")
        count = positive_int(n_paths, "n_paths")
        rng = np.random.default_rng(self.seed if seed is None else seed)
        steps = rng.normal(0.0, self.step_sd, size=(count, n))
        walk = self.start + np.cumsum(steps, axis=1)
        return self._reflect(walk)

    def path(self, weeks: int = 52, seed: int | None = None) -> np.ndarray:
        """A single price path."""
        return self.paths(weeks, 1, seed)[0]


@dataclass(frozen=True)
class RiskAssessor:
    """Classifies revenue risk by coefficient of variation.

    Why not variance? Revenue is in UGX, so its variance is in **UGX squared**
    -- a quantity with no interpretation and no natural scale. A threshold such
    as "variance > 50,000" is therefore meaningless: on weekly revenues of
    hundreds of millions of shillings, a variance of 50,000 corresponds to a
    standard deviation of about UGX 224, which is a rounding error. Worse, the
    number changes by a factor of a million if the same revenue is expressed in
    thousands of shillings, while the underlying risk is unchanged. The
    coefficient of variation (sd / mean) is dimensionless, so it survives a
    change of units and can be compared across fisheries.

    Parameters
    ----------
    moderate_above, high_above : float
        CV thresholds separating the three bands. The defaults -- 0.15 and
        0.30 -- sit either side of the 0.2-0.25 range typically reported for
        soft-commodity export earnings, so "high" means noticeably more
        volatile than a commodity exporter would normally expect.
    """

    moderate_above: float = 0.15
    high_above: float = 0.30

    def __post_init__(self) -> None:
        if not 0 < self.moderate_above < self.high_above:
            raise ValueError("need 0 < moderate_above < high_above")

    @staticmethod
    def describe(revenue: ArrayLike) -> dict[str, float]:
        """Mean, median, variance, sd and CV, computed with ``statistics``."""
        data = [float(v) for v in np.asarray(revenue, dtype=float).ravel()]
        if len(data) < 2:
            raise ValueError("need at least two observations")
        mean = statistics.mean(data)
        sd = statistics.stdev(data)
        return {"mean": mean, "median": statistics.median(data),
                "variance": statistics.variance(data), "stdev": sd,
                "cv": sd / mean if mean else float("inf")}

    def classify_cv(self, cv: float) -> RiskClass:
        """Band a coefficient of variation."""
        if cv < 0 or not np.isfinite(cv):
            raise ValueError("cv must be a finite, non-negative number")
        if cv < self.moderate_above:
            return RiskClass.LOW
        return RiskClass.MODERATE if cv < self.high_above else RiskClass.HIGH

    def classify(self, revenue: ArrayLike) -> RiskClass:
        """Band a revenue series."""
        return self.classify_cv(self.describe(revenue)["cv"])

    @staticmethod
    def tail_risk(outcomes: ArrayLike, alpha: float = 0.05) -> dict[str, float]:
        """Value-at-Risk and expected shortfall at the ``alpha`` quantile.

        VaR is the revenue level that is only undershot ``alpha`` of the time.
        Expected shortfall is the *average* revenue in those bad cases, which
        answers the question VaR cannot: how bad is bad?
        """
        draws = np.asarray(outcomes, dtype=float).ravel()
        if draws.size < 2:
            raise ValueError("need at least two outcomes")
        if not 0 < alpha < 1:
            raise ValueError("alpha must be strictly between 0 and 1")
        var = float(np.percentile(draws, alpha * 100))
        tail = draws[draws <= var]
        expected = float(tail.mean()) if tail.size else var
        mean = float(draws.mean())
        return {"mean": mean, "var": var, "expected_shortfall": expected,
                "var_shortfall": mean - var,
                "var_shortfall_pct": 100 * (mean - var) / mean if mean else float("nan")}


@dataclass
class ExportScenario:
    """One fishery, one price process and one risk rule, run together.

    Composition rather than inheritance: a scenario *has* a stock, a price
    model and an assessor, because none of the three is a kind of the others
    and each is useful on its own.
    """

    stock: FishStock = field(default_factory=FishStock)
    price: PriceModel = field(default_factory=PriceModel)
    assessor: RiskAssessor = field(default_factory=RiskAssessor)

    def __repr__(self) -> str:
        return (f"ExportScenario(h={self.stock.harvest_rate}, "
                f"price_start={self.price.start:,.0f})")

    def run(self, weeks: int = 52, open_weeks: ArrayLike | None = None,
            harvest_rate: float | None = None,
            price_path: ArrayLike | None = None) -> dict[str, object]:
        """Simulate one year and score it."""
        path = self.stock.simulate(weeks, open_weeks, harvest_rate)
        prices = (self.price.path(weeks) if price_path is None
                  else np.asarray(price_path, dtype=float).ravel())
        revenue = path.revenue(prices)
        summary = self.assessor.describe(revenue)
        return {"path": path, "prices": prices, "revenue": revenue,
                "summary": summary,
                "risk": self.assessor.classify_cv(summary["cv"])}

    def monte_carlo(self, weeks: int = 52, n_paths: int = 1_000,
                    open_weeks: ArrayLike | None = None,
                    harvest_rate: float | None = None,
                    seed: int | None = None) -> np.ndarray:
        """Annual revenue under ``n_paths`` independent price paths.

        The biomass path is deterministic given the harvest rate, so only the
        price is resampled: that is where the revenue uncertainty comes from.
        """
        path = self.stock.simulate(weeks, open_weeks, harvest_rate)
        prices = self.price.paths(weeks, n_paths, seed)
        return (path.harvest * KG_PER_TONNE) @ prices.T

    def sweep(self, rates: Sequence[float], weeks: int = 52,
              n_paths: int = 1_000, open_weeks: ArrayLike | None = None
              ) -> pd.DataFrame:
        """Compare several harvest rates on stock, catch, revenue and risk."""
        if not len(rates):
            raise ValueError("need at least one harvest rate")
        rows = []
        for rate in rates:
            outcome = self.run(weeks, open_weeks, harvest_rate=float(rate))
            annual = self.monte_carlo(weeks, n_paths, open_weeks,
                                      harvest_rate=float(rate))
            tail = self.assessor.tail_risk(annual)
            rows.append({
                "h": float(rate),
                "final_stock_t": outcome["path"].final_biomass,
                "equilibrium_t": self.stock.equilibrium_biomass(float(rate)),
                "total_catch_t": outcome["path"].total_harvest,
                "sustainable_yield_t": self.stock.equilibrium_yield(float(rate)),
                "revenue_ugx": float(outcome["revenue"].sum()),
                "cv": outcome["summary"]["cv"],
                "risk": str(outcome["risk"]),
                "mean_annual_ugx": tail["mean"],
                "var_5pct_ugx": tail["var"],
            })
        return pd.DataFrame(rows).set_index("h")


def closed_season_weeks(weeks: int = 52, closed: Sequence[int] = range(14, 22)
                        ) -> np.ndarray:
    """Boolean open/closed mask for a year with a closed season.

    Parameters
    ----------
    weeks : int
        Length of the year in weeks.
    closed : sequence of int
        Zero-based week numbers during which no harvesting happens. The
        default closes weeks 14-21, which is the eight-week window around the
        main long-rains spawning peak on Lake Victoria -- a closure is only
        worth its lost revenue if it protects fish that are about to breed.
    """
    n = positive_int(weeks, "weeks")
    mask = np.ones(n, dtype=bool)
    shut = [int(w) for w in closed]
    if any(w < 0 or w >= n for w in shut):
        raise ValueError(f"closed weeks must lie in [0, {n})")
    mask[shut] = False
    return mask
