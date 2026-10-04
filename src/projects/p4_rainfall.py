"""Mini-Project 4: Rainfall Pattern & Crop Suitability Analyser.

An agricultural extension officer wants to compare rainfall regimes across
regions and advise farmers on which months suit which crops.

Contents
--------
:class:`Region`
    A region's twelve monthly rainfall totals, with the summary statistics.
:class:`CropRule`
    A crop's suitable monthly rainfall band, with its source cited.
:class:`SuitabilityGrid`
    Composes regions and crop rules into the month-by-region classification.
:class:`SeasonProfile`
    Peak detection and the unimodal/bimodal verdict.
:class:`SimilarityStudy`
    The three pairwise measures, as matrices.

The previous version of this question asked for "cosine similarity" using
``math.cos``, which returns the cosine of an *angle* and has nothing to do
with the similarity of two vectors. :func:`cosine_similarity` implements it
properly and the notebook verifies it against SciPy.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy.signal import find_peaks, peak_prominences
from scipy.spatial.distance import cosine as scipy_cosine

from src.core.validation import as_series

__all__ = ["CROP_RULES", "MONTHS", "RAINFALL_MM", "CropRule", "Modality", "Region",
           "SeasonProfile", "SimilarityStudy", "Suitability", "SuitabilityGrid",
           "cosine_similarity", "euclidean_distance", "load_nasa_power",
           "load_regions", "pearson_correlation"]

MONTHS: tuple[str, ...] = ("Jan", "Feb", "Mar", "Apr", "May", "Jun",
                           "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")

#: Illustrative monthly rainfall in mm, as given in the brief.
RAINFALL_MM: dict[str, tuple[float, ...]] = {
    "Kampala": (120, 140, 180, 200, 220, 180, 90, 70, 60, 100, 110, 130),
    "Gulu": (8, 25, 75, 160, 190, 145, 170, 215, 175, 150, 60, 15),
    "Mbarara": (70, 85, 120, 140, 90, 25, 20, 55, 100, 125, 120, 90),
}

FAO_1986 = ("FAO (1986) Irrigation Water Management Training Manual No. 3: "
            "Irrigation Water Needs, Brouwer & Heibloem, Table 4")


class Suitability(Enum):
    """How a month's rainfall sits against one crop's requirement."""

    DROUGHT = "drought risk"
    GOOD = "good"
    WATERLOGGING = "waterlogging risk"

    @property
    def score(self) -> int:
        """-1 too dry, 0 suitable, +1 too wet -- convenient for a heatmap."""
        return {"drought risk": -1, "good": 0, "waterlogging risk": 1}[self.value]

    def __str__(self) -> str:
        return self.value


class Modality(Enum):
    """How many distinct rainy seasons a region has in a year."""

    UNIMODAL = "unimodal"
    BIMODAL = "bimodal"
    POLYMODAL = "polymodal"

    @classmethod
    def of(cls, n_peaks: int) -> "Modality":
        if n_peaks < 0:
            raise ValueError("peak count cannot be negative")
        if n_peaks <= 1:
            return cls.UNIMODAL
        return cls.BIMODAL if n_peaks == 2 else cls.POLYMODAL

    def __str__(self) -> str:
        return self.value


class Region:
    """One region's monthly rainfall, in millimetres.

    Parameters
    ----------
    name : str
        Region name.
    rainfall : array-like of float
        Twelve monthly totals, January first, all non-negative.

    Raises
    ------
    ValueError
        If the name is empty, the series is not twelve months long, or any
        value is negative or non-finite.
    """

    def __init__(self, name: str, rainfall: ArrayLike) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("name must be a non-empty string")
        values = as_series(rainfall, "rainfall", minimum=1)
        if values.size != len(MONTHS):
            raise ValueError(f"need {len(MONTHS)} monthly totals, got {values.size}")
        if np.any(values < 0):
            raise ValueError("rainfall cannot be negative")
        self.name = name.strip()
        self.rainfall = values

    # ----- dunder methods ---------------------------------------------
    def __repr__(self) -> str:
        return (f"Region({self.name!r}, annual={self.annual_total:,.0f}mm, "
                f"wettest={self.wettest_month}, driest={self.driest_month})")

    def __len__(self) -> int:
        return int(self.rainfall.size)

    def __getitem__(self, month: str | int) -> float:
        """``region["Mar"]`` or ``region[2]`` -- that month's rainfall."""
        if isinstance(month, str):
            if month not in MONTHS:
                raise KeyError(f"{month!r} is not a month name; expected one of {MONTHS}")
            return float(self.rainfall[MONTHS.index(month)])
        return float(self.rainfall[month])

    def __iter__(self) -> Iterator[tuple[str, float]]:
        return zip(MONTHS, self.rainfall.tolist())

    # ----- task 1: summary statistics ------------------------------------
    @property
    def annual_total(self) -> float:
        """Total rainfall across the year, in mm."""
        return float(self.rainfall.sum())

    @property
    def mean_monthly(self) -> float:
        return float(self.rainfall.mean())

    @property
    def wettest_month(self) -> str:
        return MONTHS[int(np.argmax(self.rainfall))]

    @property
    def driest_month(self) -> str:
        return MONTHS[int(np.argmin(self.rainfall))]

    @property
    def cv(self) -> float:
        """Coefficient of variation across the twelve months.

        High CV means the rain is concentrated into a short season; low CV
        means it is spread through the year.
        """
        mean = self.mean_monthly
        if mean == 0:
            raise ValueError("CV is undefined for a region with no rainfall")
        return float(self.rainfall.std(ddof=1) / mean)

    def summary(self) -> dict[str, float | str]:
        """Everything task 1 asks for, in one mapping."""
        return {"annual_mm": self.annual_total, "mean_monthly_mm": self.mean_monthly,
                "wettest": self.wettest_month, "driest": self.driest_month,
                "cv": self.cv}


@dataclass(frozen=True)
class CropRule:
    """A crop's suitable monthly rainfall band.

    The bands are derived from FAO crop water *needs* over a full growing
    period: the lower bound divides the lowest total need by the longest
    growing period, and the upper bound divides the highest total need by the
    shortest. That gives the monthly rate the crop must receive to meet its
    seasonal requirement.

    Parameters
    ----------
    crop : str
        Crop name.
    min_mm, max_mm : float
        Suitable monthly rainfall band, in mm.
    source : str
        Citation for the figures.

    Raises
    ------
    ValueError
        If the band is empty or negative.
    """

    crop: str
    min_mm: float
    max_mm: float
    source: str = ""

    def __post_init__(self) -> None:
        if not self.crop or not str(self.crop).strip():
            raise ValueError("crop must be named")
        if not 0 <= self.min_mm < self.max_mm:
            raise ValueError(
                f"{self.crop}: need 0 <= min_mm < max_mm, "
                f"got {self.min_mm} and {self.max_mm}")

    def classify(self, mm: float) -> Suitability:
        """Where a month's rainfall sits against this crop's band."""
        if mm < 0 or not np.isfinite(mm):
            raise ValueError("rainfall must be finite and non-negative")
        if mm < self.min_mm:
            return Suitability.DROUGHT
        return Suitability.WATERLOGGING if mm > self.max_mm else Suitability.GOOD

    def score(self, mm: float) -> int:
        """-1, 0 or +1, for the heatmap."""
        return self.classify(mm).score

    def __str__(self) -> str:
        return f"{self.crop} ({self.min_mm:.0f}-{self.max_mm:.0f} mm/month)"


#: The three crops studied here: the staple cereal, the main legume and the
#: principal cash crop of Uganda's northern farming system, which is where the
#: advisory note at the end is directed.
CROP_RULES: tuple[CropRule, ...] = (
    # 450-650 mm over 120-130 days (3.9-4.3 months)
    CropRule("Sorghum", 105, 167, FAO_1986 + ": 450-650 mm over 120-130 days"),
    # 500-700 mm over 90-140 days (3.0-4.6 months)
    CropRule("Groundnut", 110, 233, FAO_1986 + ": 500-700 mm over 90-140 days"),
    # 700-1300 mm over 180-195 days (5.9-6.4 months)
    CropRule("Cotton", 110, 220, FAO_1986 + ": 700-1300 mm over 180-195 days"),
)


# ----------------------------------------------------------------------
# Task 3-4: similarity measures
# ----------------------------------------------------------------------
def _pair(a: ArrayLike, b: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    left = as_series(a, "a", minimum=1)
    right = as_series(b, "b", minimum=1)
    if left.shape != right.shape:
        raise ValueError(f"vectors must match in length, got {left.size} and {right.size}")
    return left, right


def cosine_similarity(a: ArrayLike, b: ArrayLike) -> float:
    """Cosine of the angle *between two vectors*: ``a.b / (|a| |b|)``.

    This is what the previous version of the question got wrong. ``math.cos(x)``
    takes a single angle in radians and returns its cosine; it cannot express
    the relationship between two rainfall profiles at all. The quantity wanted
    is the normalised dot product, which is 1 when the two profiles have the
    same *shape* regardless of magnitude.
    """
    left, right = _pair(a, b)
    norms = np.linalg.norm(left) * np.linalg.norm(right)
    if norms == 0:
        raise ValueError("cosine similarity is undefined for a zero vector")
    return float(np.dot(left, right) / norms)


def pearson_correlation(a: ArrayLike, b: ArrayLike) -> float:
    """Pearson correlation: cosine similarity after removing each mean.

    Because it centres first, it measures whether two regions are wet and dry
    in the *same months*, which is the question cosine similarity cannot
    answer for strictly positive data.
    """
    left, right = _pair(a, b)
    if left.size < 2:
        raise ValueError("correlation needs at least two observations")
    if left.std() == 0 or right.std() == 0:
        raise ValueError("correlation is undefined for a constant series")
    return float(np.corrcoef(left, right)[0, 1])


def euclidean_distance(a: ArrayLike, b: ArrayLike) -> float:
    """Straight-line distance in mm: sensitive to magnitude, unlike cosine."""
    left, right = _pair(a, b)
    return float(np.linalg.norm(left - right))


class SimilarityStudy:
    """Pairwise comparison of several regions under three measures.

    Parameters
    ----------
    regions : sequence of Region
        At least two regions to compare.
    """

    MEASURES = {"cosine": cosine_similarity,
                "pearson": pearson_correlation,
                "euclidean": euclidean_distance}

    def __init__(self, regions: Sequence[Region]) -> None:
        if len(regions) < 2:
            raise ValueError("need at least two regions to compare")
        self.regions = list(regions)

    def __len__(self) -> int:
        return len(self.regions)

    def __repr__(self) -> str:
        return f"SimilarityStudy({[r.name for r in self.regions]})"

    def matrix(self, measure: str = "cosine") -> pd.DataFrame:
        """Square matrix of one measure over every pair of regions."""
        if measure not in self.MEASURES:
            raise ValueError(
                f"unknown measure {measure!r}; expected one of {sorted(self.MEASURES)}")
        fn = self.MEASURES[measure]
        names = [r.name for r in self.regions]
        grid = np.array([[fn(x.rainfall, y.rainfall) for y in self.regions]
                         for x in self.regions])
        return pd.DataFrame(grid, index=names, columns=names)

    def all_matrices(self) -> dict[str, pd.DataFrame]:
        return {name: self.matrix(name) for name in self.MEASURES}

    def verify_against_scipy(self) -> pd.DataFrame:
        """Check every cosine figure against ``scipy.spatial.distance.cosine``.

        SciPy returns the cosine *distance*, so the similarity is ``1 - d``.
        """
        rows = []
        for i, x in enumerate(self.regions):
            for y in self.regions[i + 1:]:
                ours = cosine_similarity(x.rainfall, y.rainfall)
                theirs = 1 - float(scipy_cosine(x.rainfall, y.rainfall))
                rows.append({"pair": f"{x.name} / {y.name}", "ours": ours,
                             "scipy": theirs, "abs_diff": abs(ours - theirs)})
        return pd.DataFrame(rows).set_index("pair")


# ----------------------------------------------------------------------
# Task 5: season detection
# ----------------------------------------------------------------------
class SeasonProfile:
    """Detects rainy seasons in a twelve-month profile.

    The calendar wraps, so a peak in December is adjacent to January. The
    series is therefore tiled three times and peaks are detected in the middle
    copy, which gives December and January real neighbours instead of the
    artificial cliff a plain twelve-point array would create.

    A peak counts as a season only if its prominence clears
    ``prominence_share`` of the year's range. Without that rule, every small
    wobble registers: the test is not "is this month wetter than its
    neighbours" but "does the rain genuinely stop in between".

    Parameters
    ----------
    region : Region
        The profile to analyse.
    prominence_share : float
        Fraction of the annual range a peak must stand above its surroundings.
    min_separation : int
        Minimum months between two peaks, so one broad season is not split in
        two by a single dry month.
    """

    def __init__(self, region: Region, prominence_share: float = 0.25,
                 min_separation: int = 2) -> None:
        if not 0 < prominence_share < 1:
            raise ValueError("prominence_share must be in (0, 1)")
        if min_separation < 1:
            raise ValueError("min_separation must be at least 1")
        self.region = region
        self.prominence_share = float(prominence_share)
        self.min_separation = int(min_separation)

    def __repr__(self) -> str:
        return (f"SeasonProfile({self.region.name!r}, peaks={self.peak_months()}, "
                f"{self.modality})")

    def _detect(self) -> tuple[np.ndarray, np.ndarray]:
        """Peak indices (0-11) and their prominences."""
        values = self.region.rainfall
        tiled = np.tile(values, 3)
        span = float(values.max() - values.min())
        if span == 0:
            return np.array([], dtype=int), np.array([])
        found, _ = find_peaks(tiled, prominence=self.prominence_share * span,
                              distance=self.min_separation)
        middle = found[(found >= len(values)) & (found < 2 * len(values))]
        prominences = peak_prominences(tiled, middle)[0] if middle.size else np.array([])
        return middle - len(values), prominences

    def peak_indices(self) -> np.ndarray:
        return self._detect()[0]

    def peak_months(self) -> list[str]:
        """Names of the months that start a rainy season."""
        return [MONTHS[i] for i in self.peak_indices()]

    def prominences(self) -> np.ndarray:
        return self._detect()[1]

    @property
    def modality(self) -> Modality:
        """Unimodal, bimodal or polymodal."""
        return Modality.of(len(self.peak_indices()))

    def summary(self) -> dict[str, object]:
        indices, prominences = self._detect()
        return {"region": self.region.name,
                "peaks": ", ".join(MONTHS[i] for i in indices) or "none",
                "n_peaks": int(indices.size),
                "max_prominence_mm": float(prominences.max()) if prominences.size else 0.0,
                "modality": str(self.modality)}


# ----------------------------------------------------------------------
# Tasks 2 and 6: the suitability grid
# ----------------------------------------------------------------------
class SuitabilityGrid:
    """Classifies every month of every region against every crop.

    Composition: the grid *has* regions and crop rules. Neither is a kind of
    the other, and both are useful on their own.

    Parameters
    ----------
    regions : sequence of Region
    crops : sequence of CropRule
    """

    def __init__(self, regions: Sequence[Region],
                 crops: Sequence[CropRule] = CROP_RULES) -> None:
        if not regions:
            raise ValueError("need at least one region")
        if not crops:
            raise ValueError("need at least one crop rule")
        self.regions = list(regions)
        self.crops = list(crops)

    def __repr__(self) -> str:
        return (f"SuitabilityGrid(regions={[r.name for r in self.regions]}, "
                f"crops={[c.crop for c in self.crops]})")

    def __len__(self) -> int:
        return len(self.regions) * len(self.crops)

    def labels(self, crop: str) -> pd.DataFrame:
        """Month-by-region table of verdicts for one crop."""
        rule = self._rule(crop)
        return pd.DataFrame(
            {r.name: [str(rule.classify(v)) for v in r.rainfall] for r in self.regions},
            index=list(MONTHS))

    def scores(self, crop: str) -> pd.DataFrame:
        """The same table as -1/0/+1, ready for a heatmap."""
        rule = self._rule(crop)
        return pd.DataFrame(
            {r.name: [rule.score(v) for v in r.rainfall] for r in self.regions},
            index=list(MONTHS))

    def suitable_months(self, crop: str) -> pd.Series:
        """How many months each region is suitable for one crop."""
        return (self.scores(crop) == 0).sum()

    def overview(self) -> pd.DataFrame:
        """Suitable-month counts for every crop and region."""
        return pd.DataFrame({c.crop: self.suitable_months(c.crop)
                             for c in self.crops})

    def _rule(self, crop: str) -> CropRule:
        for rule in self.crops:
            if rule.crop.lower() == crop.lower():
                return rule
        raise ValueError(
            f"unknown crop {crop!r}; have {[c.crop for c in self.crops]}")


def load_regions(names: Sequence[str] | None = None) -> list[Region]:
    """Build :class:`Region` objects from the illustrative data."""
    wanted = list(RAINFALL_MM) if names is None else list(names)
    missing = [n for n in wanted if n not in RAINFALL_MM]
    if missing:
        raise ValueError(f"unknown region(s) {missing}; have {sorted(RAINFALL_MM)}")
    return [Region(n, RAINFALL_MM[n]) for n in wanted]


def load_nasa_power(path: str | Path) -> pd.DataFrame:
    """Read a NASA POWER monthly CSV into a year x month table of mm.

    The POWER download carries a header block terminated by ``-END HEADER-``,
    then a row per year with one column per month. Used by the optional
    extension; the notebook falls back to the illustrative data if no file is
    present.
    """
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"no NASA POWER file at {source}")
    lines = source.read_text(encoding="utf-8").splitlines()
    start = next((i + 1 for i, line in enumerate(lines) if "-END HEADER-" in line), 0)
    frame = pd.read_csv(source, skiprows=start)
    frame.columns = [str(c).strip().upper() for c in frame.columns]
    month_cols = [c for c in frame.columns if c[:3].title() in MONTHS]
    if "YEAR" not in frame.columns or len(month_cols) != 12:
        raise ValueError(f"{source.name} does not look like a POWER monthly export")
    table = frame.set_index("YEAR")[month_cols]
    table.columns = list(MONTHS)
    return table.astype(float)
