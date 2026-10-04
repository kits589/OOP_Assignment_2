"""Mini-Project 2: Solar Micro-Grid Dispatch Planner.

A rural health centre in Kasese draws each day's energy from solar panels
(``x``) and batteries (``y``) so as to satisfy two demand constraints::

    3x + 2y = D1   (daytime load, kWh)
    4x +  y = D2   (critical-equipment load, kWh)

Contents
--------
:class:`MicroGrid`
    The coefficient matrix, its conditioning diagnostics and the solvers.
:class:`HybridMicroGrid`
    Adds a diesel generator and a third constraint (the extension).
:class:`DemandLog`
    The two input modes: a seeded 30-day CSV, and interactive entry.
:class:`TariffSchedule`
    Costs a dispatch at the brief's UGX 150/kWh solar, UGX 450/kWh battery.
:class:`DispatchReport`
    The outcome of dispatching a batch of days, including what was repaired.

Infeasibility strategy
----------------------
This build takes the first option the brief offers: **clip the negative
component to zero and report the energy the repair fails to deliver**. The
residual ``norm(A @ repaired - demand)`` is carried through to the report, so
a repaired day is never silently presented as if it had been met.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Iterator, Sequence

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy import linalg

from src.core.validation import positive_int

__all__ = ["Conditioning", "DemandLog", "DispatchReport", "DispatchStatus",
           "HybridMicroGrid", "MicroGrid", "TariffSchedule"]

#: Rule of thumb for reading a condition number.
WELL_CONDITIONED = 100.0
ILL_CONDITIONED = 1e8


class Conditioning(Enum):
    """How trustworthy a linear system is, judged by its condition number."""

    SINGULAR = "singular"
    ILL_CONDITIONED = "ill-conditioned"
    ACCEPTABLE = "acceptable"
    WELL_CONDITIONED = "well-conditioned"

    @classmethod
    def of(cls, determinant: float, condition: float) -> "Conditioning":
        """Band a system from its determinant and condition number."""
        if not np.isfinite(condition) or np.isclose(determinant, 0.0):
            return cls.SINGULAR
        if condition >= ILL_CONDITIONED:
            return cls.ILL_CONDITIONED
        return cls.WELL_CONDITIONED if condition < WELL_CONDITIONED else cls.ACCEPTABLE

    @property
    def usable(self) -> bool:
        """Whether a system in this state can be solved and believed."""
        return self in {Conditioning.WELL_CONDITIONED, Conditioning.ACCEPTABLE}

    def __str__(self) -> str:
        return self.value


class DispatchStatus(Enum):
    """What happened to one day of dispatch."""

    MET = "met"
    REPAIRED = "repaired"

    def __str__(self) -> str:
        return self.value


class MicroGrid:
    """A linear dispatch model ``A s = d`` over a fixed set of energy sources.

    Parameters
    ----------
    matrix : array-like, optional
        Square coefficient matrix. Defaults to :attr:`COEFFICIENTS`.
    sources : sequence of str
        Column names: the energy sources being dispatched.
    loads : sequence of str
        Row names: the demand constraints being satisfied.

    Raises
    ------
    ValueError
        If the matrix is not square, is not finite, or if the number of names
        does not match its size.
    """

    #: The system given in the brief: 3x + 2y = D1, 4x + y = D2.
    COEFFICIENTS: list[list[float]] = [[3.0, 2.0], [4.0, 1.0]]
    SOURCES: tuple[str, ...] = ("solar", "battery")
    LOADS: tuple[str, ...] = ("D1", "D2")

    def __init__(self, matrix: ArrayLike | None = None,
                 sources: Sequence[str] | None = None,
                 loads: Sequence[str] | None = None) -> None:
        A = np.asarray(self.COEFFICIENTS if matrix is None else matrix, dtype=float)
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError(f"the coefficient matrix must be square, got {A.shape}")
        if not np.all(np.isfinite(A)):
            raise ValueError("the coefficient matrix must be finite")
        self.sources = tuple(self.SOURCES if sources is None else sources)
        self.loads = tuple(self.LOADS if loads is None else loads)
        if len(self.sources) != A.shape[1] or len(self.loads) != A.shape[0]:
            raise ValueError(
                f"need {A.shape[1]} source names and {A.shape[0]} load names, "
                f"got {len(self.sources)} and {len(self.loads)}")
        self.A = A

    # ----- dunder methods ---------------------------------------------
    def __repr__(self) -> str:
        return (f"{type(self).__name__}(order={self.order}, "
                f"det={self.determinant:.4g}, cond={self.condition_number:.4g}, "
                f"{self.conditioning})")

    def __len__(self) -> int:
        """The order of the system: how many sources it dispatches."""
        return self.order

    # ----- task 1: is the system well posed? ----------------------------
    @property
    def order(self) -> int:
        return int(self.A.shape[0])

    @property
    def determinant(self) -> float:
        """Zero means the loads are linearly dependent and there is no unique mix."""
        return float(np.linalg.det(self.A))

    @property
    def condition_number(self) -> float:
        """How far a 1% error in demand can be amplified in the dispatch."""
        return float(np.linalg.cond(self.A))

    @property
    def rank(self) -> int:
        return int(np.linalg.matrix_rank(self.A))

    @property
    def conditioning(self) -> Conditioning:
        """The band this system falls into."""
        return Conditioning.of(self.determinant, self.condition_number)

    def diagnose(self) -> dict[str, float | int | str]:
        """Every conditioning figure in one mapping, ready for a table."""
        return {"order": self.order, "determinant": self.determinant,
                "condition_number": self.condition_number, "rank": self.rank,
                "full_rank": self.rank == self.order,
                "verdict": str(self.conditioning)}

    def _require_usable(self) -> None:
        if not self.conditioning.usable:
            raise ValueError(
                f"refusing to solve a {self.conditioning} system "
                f"(det={self.determinant:.4g}, cond={self.condition_number:.4g})")

    # ----- task 2/3: solving --------------------------------------------
    def _as_demands(self, demands: ArrayLike) -> np.ndarray:
        """Coerce demand input to shape (order, n_days)."""
        d = np.asarray(demands, dtype=float)
        if d.ndim == 1:
            d = d.reshape(self.order, 1) if d.size == self.order else d.reshape(-1, 1)
        if d.ndim != 2 or d.shape[0] != self.order:
            raise ValueError(
                f"demands must have {self.order} rows, got shape {d.shape}")
        if not np.all(np.isfinite(d)):
            raise ValueError("demands must be finite")
        if np.any(d < 0):
            raise ValueError("demands must be non-negative")
        return d

    def solve_day(self, *demands: float) -> np.ndarray:
        """Dispatch for a single day, one argument per load.

        >>> MicroGrid().solve_day(100, 120).round(2)
        array([28.,  8.])
        """
        if len(demands) != self.order:
            raise ValueError(f"expected {self.order} demands, got {len(demands)}")
        return self.solve(np.asarray(demands, dtype=float))[:, 0]

    def solve(self, demands: ArrayLike) -> np.ndarray:
        """Vectorised solve: every day in one ``scipy.linalg.solve`` call.

        Returns an ``(order, n_days)`` array of source allocations.
        """
        self._require_usable()
        return linalg.solve(self.A, self._as_demands(demands))

    def solve_sequentially(self, demands: ArrayLike) -> np.ndarray:
        """The same answer, one day at a time in a Python loop.

        Kept so the notebook can time it against :meth:`solve` and show what
        vectorising actually buys.
        """
        self._require_usable()
        d = self._as_demands(demands)
        out = np.empty_like(d)
        for day in range(d.shape[1]):
            out[:, day] = linalg.solve(self.A, d[:, day])
        return out

    # ----- task 4: infeasible days ---------------------------------------
    @staticmethod
    def infeasible(allocation: ArrayLike, tolerance: float = 1e-9) -> np.ndarray:
        """Boolean mask of days where any source was asked to supply < 0."""
        return np.any(np.asarray(allocation, dtype=float) < -abs(tolerance), axis=0)

    def repair(self, allocation: ArrayLike,
               demands: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        """Clip negative allocations to zero and measure what goes unmet.

        A negative allocation is physically meaningless -- a panel cannot
        absorb energy to balance the books -- so the day is clipped. Clipping
        breaks the equality, and the amount by which it breaks is exactly the
        energy the micro-grid cannot deliver, so it is returned rather than
        discarded.

        Returns
        -------
        repaired : ndarray
            The clipped allocation, same shape as the input.
        unmet : ndarray
            Per-day ``norm(A @ repaired - demand)``, in kWh.
        """
        s = np.asarray(allocation, dtype=float)
        d = self._as_demands(demands)
        if s.shape != d.shape:
            raise ValueError(f"allocation {s.shape} and demands {d.shape} must match")
        repaired = np.clip(s, 0.0, None)
        unmet = np.linalg.norm(self.A @ repaired - d, axis=0)
        return repaired, unmet

    def dispatch(self, demands: ArrayLike) -> "DispatchReport":
        """Solve, repair what cannot be met, and report the whole batch."""
        d = self._as_demands(demands)
        raw = self.solve(d)
        flagged = self.infeasible(raw)
        repaired, unmet = self.repair(raw, d)
        table = pd.DataFrame(repaired.T, columns=list(self.sources))
        for i, load in enumerate(self.loads):
            table[load] = d[i]
        table["status"] = [str(DispatchStatus.REPAIRED if bad else DispatchStatus.MET)
                           for bad in flagged]
        table["unmet_kwh"] = unmet
        table.index.name = "day"
        return DispatchReport(grid=self, table=table, raw=raw, flagged=flagged)

    # ----- task 5: volatility ---------------------------------------------
    def volatility(self, allocation: ArrayLike) -> pd.DataFrame:
        """Mean, variance, sd and CV per source, computed with ``statistics``.

        The coefficient of variation is what makes the two sources comparable:
        variance is in kWh-squared and scales with the size of the source, so a
        raw variance says more about how much energy a source supplies than
        about how erratic it is.
        """
        import statistics

        s = np.asarray(allocation, dtype=float)
        if s.ndim != 2 or s.shape[0] != self.order:
            raise ValueError(f"expected an ({self.order}, n_days) allocation")
        if s.shape[1] < 2:
            raise ValueError("need at least two days to measure volatility")
        rows = []
        for name, values in zip(self.sources, s):
            data = values.tolist()
            mean = statistics.mean(data)
            sd = statistics.stdev(data)
            rows.append({"source": name, "mean_kwh": mean,
                         "variance": statistics.variance(data), "stdev": sd,
                         "cv": sd / mean if mean else float("inf")})
        return pd.DataFrame(rows).set_index("source")


class HybridMicroGrid(MicroGrid):
    """Solar, battery and a diesel generator, under a third constraint.

    The third load chosen here is the **evening load**, served after the
    panels stop producing::

        3x + 2y      = D1   (daytime load, unchanged)
        4x +  y      = D2   (critical-equipment load, unchanged)
        2x + 3y + 4z = D3   (evening load; diesel carries most of it)

    The coefficients say that a unit of diesel contributes twice what a unit
    of solar does after dark, which is the point of installing it.
    """

    COEFFICIENTS = [[3.0, 2.0, 0.0], [4.0, 1.0, 0.0], [2.0, 3.0, 4.0]]
    SOURCES = ("solar", "battery", "diesel")
    LOADS = ("D1", "D2", "D3")

    def __init__(self, matrix: ArrayLike | None = None,
                 sources: Sequence[str] | None = None,
                 loads: Sequence[str] | None = None) -> None:
        super().__init__(matrix, sources, loads)
        if self.order != 3:
            raise ValueError("a HybridMicroGrid describes a 3x3 system")

    @classmethod
    def with_dependent_load(cls) -> "HybridMicroGrid":
        """A deliberately broken variant whose third row is D1 + D2.

        Used in the notebook to show what a singular system looks like: the
        third equation adds no information, the determinant collapses to zero
        and no unique dispatch exists.
        """
        A = np.array(cls.COEFFICIENTS, dtype=float)
        A[2] = A[0] + A[1]
        return cls(A)


@dataclass(frozen=True)
class TariffSchedule:
    """Cost of energy by source, in UGX per kWh.

    Frozen, so a costing cannot be edited halfway through an analysis.

    Parameters
    ----------
    rates : dict of str to float
        UGX per kWh for each source. The brief gives solar 150 and battery 450.
    """

    rates: dict[str, float] = field(
        default_factory=lambda: {"solar": 150.0, "battery": 450.0, "diesel": 900.0})

    def __post_init__(self) -> None:
        if not self.rates:
            raise ValueError("a tariff schedule needs at least one rate")
        if any(rate < 0 for rate in self.rates.values()):
            raise ValueError("tariff rates must be non-negative")

    def daily(self, dispatch: pd.DataFrame) -> pd.Series:
        """UGX cost for each day of a dispatch table."""
        priced = [name for name in self.rates if name in dispatch.columns]
        if not priced:
            raise ValueError(
                f"none of {sorted(self.rates)} appear in {list(dispatch.columns)}")
        cost = sum(dispatch[name] * self.rates[name] for name in priced)
        return pd.Series(cost, index=dispatch.index, name="cost_ugx")

    def total(self, dispatch: pd.DataFrame) -> float:
        """Cost of the whole period."""
        return float(self.daily(dispatch).sum())

    def monthly(self, dispatch: pd.DataFrame, days: int = 30) -> float:
        """Mean daily cost scaled to a ``days``-day month."""
        return float(self.daily(dispatch).mean()) * positive_int(days, "days")


@dataclass
class DispatchReport:
    """The result of dispatching a batch of days."""

    grid: MicroGrid
    table: pd.DataFrame
    raw: np.ndarray
    flagged: np.ndarray

    def __len__(self) -> int:
        return len(self.table)

    def __repr__(self) -> str:
        return (f"DispatchReport(days={len(self)}, repaired={self.n_repaired}, "
                f"unmet={self.total_unmet:,.1f} kWh)")

    @property
    def n_repaired(self) -> int:
        """How many days needed clipping."""
        return int(self.flagged.sum())

    @property
    def total_unmet(self) -> float:
        """Total energy the repaired dispatch fails to deliver, in kWh."""
        return float(self.table["unmet_kwh"].sum())

    def repaired_days(self) -> pd.DataFrame:
        """Just the days that had to be repaired."""
        return self.table.loc[self.flagged]

    def volatility(self) -> pd.DataFrame:
        """Per-source volatility of the repaired allocation."""
        return self.grid.volatility(self.table[list(self.grid.sources)].to_numpy().T)


class DemandLog:
    """The two input modes the brief asks for: a generated CSV, and typing.

    Parameters
    ----------
    columns : sequence of str
        Names of the demand series, one per load.
    """

    def __init__(self, columns: Sequence[str] = ("D1", "D2")) -> None:
        if not columns:
            raise ValueError("need at least one demand column")
        self.columns = tuple(columns)

    def __repr__(self) -> str:
        return f"DemandLog(columns={self.columns})"

    def __iter__(self) -> Iterator[str]:
        return iter(self.columns)

    # ----- mode (b): a seeded CSV ------------------------------------------
    def generate(self, path: str | Path, days: int = 30, seed: int = 7,
                 base: Sequence[float] = (100.0, 120.0),
                 weekly_swing: float = 0.12, noise: float = 0.05) -> pd.DataFrame:
        """Write ``days`` days of synthetic demand with a weekly rhythm.

        A health centre runs busier on weekdays, so the series carries a
        sinusoidal weekly component plus multiplicative noise. Everything goes
        through ``np.random.default_rng(seed)``, so the file regenerates
        identically every run.

        Parameters
        ----------
        path : str or Path
            Where to write the CSV.
        days : int
            How many days to generate.
        seed : int
            Seed for the generator.
        base : sequence of float
            Mean demand in kWh for each column.
        weekly_swing : float
            Amplitude of the weekly cycle, as a fraction of the mean.
        noise : float
            Standard deviation of the multiplicative noise.
        """
        n = positive_int(days, "days")
        if len(base) != len(self.columns):
            raise ValueError(f"need {len(self.columns)} base values, got {len(base)}")
        if not 0 <= weekly_swing < 1 or noise < 0:
            raise ValueError("weekly_swing must be in [0, 1) and noise non-negative")
        rng = np.random.default_rng(seed)
        day_index = np.arange(n)
        weekly = 1 + weekly_swing * np.sin(2 * np.pi * day_index / 7)
        frame = pd.DataFrame({"day": day_index + 1})
        for name, level in zip(self.columns, base):
            jitter = rng.normal(1.0, noise, size=n)
            frame[name] = np.round(level * weekly * jitter, 2)
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(target, index=False)
        return frame

    def load(self, path: str | Path) -> pd.DataFrame:
        """Read a demand CSV and check it has what the grid needs."""
        source = Path(path)
        if not source.exists():
            raise FileNotFoundError(f"no demand file at {source}")
        frame = pd.read_csv(source)
        missing = [c for c in self.columns if c not in frame.columns]
        if missing:
            raise ValueError(f"{source.name} is missing column(s) {missing}")
        values = frame[list(self.columns)]
        if values.isna().any().any():
            raise ValueError(f"{source.name} contains blanks")
        if (values < 0).any().any():
            raise ValueError(f"{source.name} contains negative demand")
        return frame

    def as_matrix(self, frame: pd.DataFrame) -> np.ndarray:
        """The demand columns as an ``(n_loads, n_days)`` array."""
        return frame[list(self.columns)].to_numpy(dtype=float).T

    # ----- mode (a): interactive entry ---------------------------------------
    @staticmethod
    def parse(text: str) -> float:
        """Turn one typed line into a demand figure, or raise ``ValueError``.

        Rejects blanks, non-numeric text, negatives and non-finite values,
        with a message that says which of those went wrong.
        """
        if text is None or not str(text).strip():
            raise ValueError("please enter a number (the entry was blank)")
        try:
            value = float(str(text).strip().replace(",", ""))
        except ValueError:
            raise ValueError(f"{text.strip()!r} is not a number") from None
        if not np.isfinite(value):
            raise ValueError("the demand must be a finite number")
        if value < 0:
            raise ValueError("the demand cannot be negative")
        return value

    @classmethod
    def ask(cls, prompt: str, reader: Callable[[str], str] = input,
            writer: Callable[[str], None] = print, attempts: int = 3) -> float:
        """Prompt until a usable number arrives, or give up after ``attempts``.

        ``reader`` and ``writer`` are injected so the notebook and the tests
        can drive this without a human at a keyboard.
        """
        limit = positive_int(attempts, "attempts")
        for remaining in range(limit, 0, -1):
            try:
                return cls.parse(reader(prompt))
            except ValueError as err:
                writer(f"  {err}. {remaining - 1} attempt(s) left.")
        raise ValueError(f"no valid entry after {limit} attempts")

    def ask_all(self, reader: Callable[[str], str] = input,
                writer: Callable[[str], None] = print) -> np.ndarray:
        """Collect one demand per load, interactively."""
        return np.array([self.ask(f"{name} (kWh): ", reader, writer)
                         for name in self.columns])


def monte_carlo_sensitivity(grid: MicroGrid, demands: ArrayLike,
                            spread: float = 0.05, draws: int = 1000,
                            seed: int = 7) -> dict[str, float | np.ndarray]:
    """Extension: how far does the dispatch move when demand moves +/- 5%?

    Perturbs each demand uniformly within ``+/- spread`` and re-solves. The
    result is compared with the condition number, which bounds the ratio of
    relative output error to relative input error.

    Parameters
    ----------
    grid : MicroGrid
        The system to perturb.
    demands : array-like
        One day's demands, the point being perturbed around.
    spread : float
        Relative perturbation, 0.05 for the brief's +/- 5%.
    draws : int
        Monte Carlo draws; the brief asks for 1,000.
    seed : int
        Seed for reproducibility.
    """
    if not 0 < spread < 1:
        raise ValueError("spread must be in (0, 1)")
    n = positive_int(draws, "draws")
    base = np.asarray(demands, dtype=float).ravel()
    if base.size != grid.order:
        raise ValueError(f"expected {grid.order} demands, got {base.size}")
    baseline = grid.solve_day(*base)
    rng = np.random.default_rng(seed)
    factors = rng.uniform(1 - spread, 1 + spread, size=(n, base.size))
    perturbed = grid.solve((base * factors).T)

    relative_out = np.linalg.norm(perturbed - baseline[:, None], axis=0) / np.linalg.norm(
        baseline)
    relative_in = np.linalg.norm((base * factors - base).T, axis=0) / np.linalg.norm(base)
    with np.errstate(divide="ignore", invalid="ignore"):
        amplification = np.where(relative_in > 0, relative_out / relative_in, np.nan)
    return {"baseline": baseline, "solutions": perturbed,
            "amplification": amplification,
            "max_amplification": float(np.nanmax(amplification)),
            "mean_amplification": float(np.nanmean(amplification)),
            "condition_number": grid.condition_number}
