# OOP with Python — Advent 2026 Mini-Projects

**Student:** _<your name>_ · **Reg. no.:** _<your registration number>_ · **Programme:** MSCS / MSDS

Five mini-projects set in Ugandan contexts. Each is modelled with classes, backed by reusable
modules under `src/`, driven from its own Jupyter notebook and covered by `pytest`.

| # | Mini-project | Notebook | Module |
|---|---|---|---|
| 1 | UBOS District Population Forecaster | `notebooks/project1_population.ipynb` | `src/projects/p1_population.py` |
| 2 | Solar Micro-Grid Dispatch Planner (Kasese) | `notebooks/project2_microgrid.ipynb` | `src/projects/p2_microgrid.py` |
| 3 | Lake Victoria Fish Stock & Export Risk | `notebooks/project3_fishery.ipynb` | `src/projects/p3_fishery.py` |
| 4 | Rainfall Pattern & Crop Suitability | `notebooks/project4_rainfall.ipynb` | `src/projects/p4_rainfall.py` |
| 5 | Taxi Route Revenue, Pricing & Fleet Planner | `notebooks/project5_taxi.ipynb` | `src/projects/p5_taxi.py` |

## Repository layout

```
├── README.md
├── requirements.txt
├── src/
│   ├── core/                  # shared across mini-projects
│   │   ├── forecaster.py      # abstract Forecaster + self-registering model catalogue
│   │   ├── metrics.py         # ErrorReport: MAE / RMSE / MAPE as one comparable object
│   │   ├── validation.py      # @validates_series, @forecasts, @requires_fit
│   │   └── plotting.py        # house chart style and figure paths
│   └── projects/
│       ├── p1_population.py   ├── p2_microgrid.py   ├── p3_fishery.py
│       ├── p4_rainfall.py     └── p5_taxi.py
├── notebooks/                 # one notebook per mini-project, committed with outputs
├── tests/                     # pytest suites, one file per mini-project (292 tests)
├── data/                      # the generated demand CSV; optional NASA POWER files
├── figures/                   # PNGs written by the notebooks
└── tools/
    ├── build_notebooks.py     # percent-format .py  ->  executed .ipynb
    └── nb_src/                # the notebooks' real source, in plain text
```

### Design decisions worth asking about

* **Validation by decorator, not by template method.** `@validates_series` coerces and
  stores a model's training data; `@forecasts` checks the horizon and that `fit` has run.
  A concrete model therefore contains only its own mathematics and no guard clauses, and
  no check is written twice.
* **Models register themselves.** A model declares `family="population"` or
  `family="taxi"` in its class header; `Forecaster.__init_subclass__` files it into a
  catalogue that `ModelSelector` and `WalkForward` read. Defining a model is enough to
  enter it into the comparison — there is no factory dictionary to drift out of step.
* **One forecasting hierarchy, two mini-projects.** P5's `RouteLinearTrend` **inherits**
  P1's `LinearTrendModel` unchanged; only the registered family differs, because fitting a
  line with `np.polyfit` is the same operation on a population or a passenger count.
* **Inheritance where it earns its place.** `HybridMicroGrid(MicroGrid)` reuses every
  solver, diagnostic and repair method and supplies only a larger matrix.
* **Composition where that is the honest relationship.** `ExportScenario` combines a
  `FishStock`, a `PriceModel` and a `RiskAssessor`; `SuitabilityGrid` combines `Region`
  and `CropRule` objects. None is a kind of the others, and each is useful alone.
* **Results are objects, not tuples.** `ErrorReport` computes MAE/RMSE/MAPE as
  `cached_property` and implements `__lt__`, so model selection is `min(reports)`.
* **Enums for every classification.** `Conditioning`, `DispatchStatus`, `RiskClass`,
  `Suitability` and `Modality` return enum members rather than bare strings, so a typo in
  a comparison fails loudly instead of quietly evaluating false.
* **Frozen value objects.** `ClassroomPlanner`, `TariffSchedule`, `RiskAssessor`,
  `CropRule`, `MarketModel` and `FleetPlanner` are frozen dataclasses: a set of
  assumptions cannot be edited halfway through an analysis.
* **Validation everywhere.** Every constructor rejects empty, negative, non-finite or
  mismatched input with a `ValueError`; type hints and docstrings on all public API.

## Setup

```bash
git clone <this-repo-url>
cd advent2026-a
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Developed on Python 3.13.7 with NumPy 2.4.4, SciPy 1.18.1, pandas 3.0.2 and
Matplotlib 3.11.2. Python 3.10 or newer should work.

## Running

```bash
pytest -q                                    # 292 tests, from the repository root

jupyter notebook notebooks/                  # then Kernel ▸ Restart & Run All

python tools/build_notebooks.py              # rebuild + execute every notebook
python tools/build_notebooks.py project3_fishery --no-exec   # assemble only
```

The notebooks locate `src/` whether Jupyter starts at the repository root or in
`notebooks/`. Their real source is the percent-format scripts in `tools/nb_src/`, so
diffs stay readable; `build_notebooks.py` regenerates the `.ipynb` files and runs them.
It refuses to write a notebook whose cells raised, so a committed notebook is one that
executed cleanly.

**Reproducibility.** All randomness goes through `np.random.default_rng(seed)` with the
seed fixed at the top of each notebook (`SEED = 7`) or passed into the function that
needs it. The P2 demand CSV is regenerated from its seed on every run.

## Summary of findings

**P1 — Population.** Every district grows close to exponentially, so on a 2015–21 train /
2022–24 test split the CAGR model wins outright (MAPE 0.16–2.03%). The linear trend comes
second (0.55–4.01%) but cannot curve; the Fibonacci-ratio model misses by 145–167% because
its ratios converge on φ, implying 61.8% annual growth — ten times the fastest rate
observed. Wakiso compounds fastest at 6.47% a year, Mbale slowest at 2.42%. By 2029 the
five districts need about **4,600 additional primary classrooms**: Wakiso 2,088, Kampala
1,545, Gulu 412, Mbale 279, Jinja 276. The 95% residual bootstrap adds only ~1%, because a
CAGR curve is pinned to its endpoints and cannot express model error.

**P2 — Micro-grid.** det A = −5 and κ ≈ 5.83, so the system is well posed and well
conditioned; Monte Carlo perturbation of ±5% produced a worst-case amplification just
under κ, as the bound promises. Solving 30 days in one vectorised call runs about **5×**
faster than looping, almost entirely on per-call overhead rather than arithmetic. Four of
30 days were infeasible (all with D2 > 4/3·D1, which drives the battery negative) and were
repaired by clipping, leaving ~18 kWh unserved. The battery is far more volatile (CV 0.69
against solar's 0.10) because it is the swing producer, and at 3× the tariff it is **42%
of the bill on 19% of the energy**: about UGX 7,300 a day, UGX 220,000 a month. Adding
diesel under an evening-load constraint keeps the system well conditioned (det = −20).

**P3 — Fishery.** The logistic model replaces unbounded Fibonacci growth and reproduces
the closed-form equilibrium N\* = K(1−h/r) to six decimal places. MSY is rK/4 = **1,000
t/week at h = 0.20**. The cooperative's current h = 0.10 is safe (stock settles at 7,500 t)
but lands **27% below** what the fishery could sustainably yield; pushing to h = 0.30 is
strictly worse than MSY on both counts — 17% less fish *and* half the standing stock. The
old "variance > 50,000" rule is not merely miscalibrated but meaningless: revenue variance
is in UGX², so re-expressing the same revenue in thousands changes it by 10⁶ while the
risk is unchanged. Risk is therefore classified on the dimensionless CV (0.13 at h = 0.10
→ low), and the 5% VaR of annual revenue sits ~16% below expectation. An eight-week closed
season over the spawning peak **costs** revenue at h ≤ 0.20 (−13.5% at h = 0.10) and only
pays once the fishery is over-exploited (+7.4% at h = 0.30, +41.5% at h = 0.35).

**P4 — Rainfall.** Cosine similarity, implemented properly and verified against SciPy to
machine precision, rates every pair of regions 0.75–0.89 — which looks like agreement and
is an artefact: rainfall vectors are non-negative, so they all sit in the positive orthant
where cosine cannot fall below zero. Pearson correlation, which centres first, shows the
seasonal timing is essentially unrelated (−0.17 to 0.21). Peak detection with circular
padding and a 25% relative-prominence rule classifies **Gulu unimodal and Mbarara
bimodal**, matching known climate zones, and is stable from 10% to 60%. Kampala is
misclassified as unimodal because the illustrative series lacks the June–July dip, not
because the rule fails. All three crop bands cluster near 105–110 mm/month despite
seasonal requirements differing threefold, because drought-tolerant crops also have
shorter growing periods — so the classification is driven by the upper bound.

**P5 — Taxis.** The Ntinda market clears at **P\* = UGX 2,200, Q\* = 76**, so the current
UGX 2,000 fare sustains a shortage of ~10 passengers per trip-hour. Revenue ranking
follows fare, not volume: Entebbe earns 3.6× Ntinda's revenue on 1.4× the passengers. In
walk-forward backtesting the drift model wins on two routes and SES on one, but the α grid
search lands on **α = 1.0** everywhere — SES collapsing into the naive forecast, meaning
ten observations hold no exploitable structure. Fleet size depends on a definition the
brief leaves open: **1 vehicle** per route read as daily totals, **6–8** read as hourly
stage counts over a 12-hour day. Over 60 simulated days with a weekly pattern, a
seasonal-naïve model cuts MAE from 8.4 to 4.0 passengers (**52%**), because the moving
average's errors are not noise but the weekly shape it cannot represent.

## Data sources and assumptions

* Population, rainfall, route and demand data are **illustrative**, as the brief states.
  **Jinja** and **Mbale** (P1) are the two additional districts required by task 1; the P2
  demand CSV is generated here from a fixed seed.
* Crop water needs: Brouwer, C. & Heibloem, M. (1986) *Irrigation Water Management
  Training Manual No. 3: Irrigation Water Needs*, FAO, Table 4.
  <https://www.fao.org/4/s2022e/s2022e02.htm>
* **Optional P4 extension (real data).** Download monthly precipitation from the
  [NASA POWER Data Access Viewer](https://power.larc.nasa.gov/data-access-viewer/):
  community *Agroclimatology*, temporal *Monthly & Annual*, parameter *Precipitation
  Corrected (PRECTOTCORR)*, 2010–2023, CSV. Points: Kampala (0.35, 32.58), Gulu
  (2.78, 32.30), Mbarara (−0.61, 30.65). Save as `data/nasa_power_Kampala.csv` and re-run
  notebook 4; it will draw monthly box plots. The notebook runs fine without them.

## AI-use declaration

_Edit this section so that it describes accurately how **you** used AI._

This repository was developed with the help of an AI coding assistant. It was used to
scaffold the repository structure, draft the class designs, implementations, tests and
notebook text, and to check results. I have reviewed, run and tested all of the code:
`pytest` passes and every notebook runs top to bottom with Restart & Run All. I can
explain each method, formula and design decision. Key results are checked against hand
calculations, which appear as `assert` statements in the notebooks.
