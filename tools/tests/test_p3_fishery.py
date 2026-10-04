"""Tests for Mini-Project 3."""
import numpy as np
import pytest

from src.projects.p3_fishery import (
    KG_PER_TONNE, ExportScenario, FishStock, PriceModel, RiskAssessor, RiskClass,
    closed_season_weeks, fibonacci_stock)


@pytest.fixture
def stock() -> FishStock:
    return FishStock()


@pytest.fixture
def scenario() -> ExportScenario:
    return ExportScenario(stock=FishStock(), price=PriceModel(seed=7))


# ----- task 1: the Fibonacci baseline ---------------------------------
def test_fibonacci_baseline_matches_the_sequence():
    assert fibonacci_stock(15).astype(int).tolist()[:7] == [1, 1, 2, 3, 5, 8, 13]
    assert fibonacci_stock(15).size == 15


def test_fibonacci_growth_is_unbounded():
    short, long = fibonacci_stock(15), fibonacci_stock(40)
    # it keeps accelerating; a real stock would flatten at carrying capacity
    assert long[-1] / long[-2] > 1.6
    assert long[-1] > 1_000 * short[-1]


@pytest.mark.parametrize("count", [0, -3, 2.5])
def test_fibonacci_rejects_a_bad_count(count):
    with pytest.raises(ValueError):
        fibonacci_stock(count)


# ----- task 2: the logistic stock --------------------------------------
def test_defaults_match_the_brief(stock):
    assert (stock.r, stock.K, stock.N0) == (0.4, 10_000.0, 4_000.0)


@pytest.mark.parametrize("kwargs", [
    {"r": 0}, {"r": -0.1}, {"K": 0}, {"N0": -5}, {"N0": 20_000},
    {"harvest_rate": 1.0}, {"harvest_rate": -0.1}, {"r": np.inf},
])
def test_impossible_parameters_are_rejected(kwargs):
    with pytest.raises(ValueError):
        FishStock(**kwargs)


def test_msy_matches_the_closed_form(stock):
    # MSY = rK/4 at h = r/2
    assert stock.msy == pytest.approx(0.4 * 10_000 / 4)
    assert stock.h_msy == pytest.approx(0.2)
    assert stock.equilibrium_yield(stock.h_msy) == pytest.approx(stock.msy)


@pytest.mark.parametrize("h,expected", [(0.0, 10_000), (0.1, 7_500),
                                        (0.2, 5_000), (0.3, 2_500), (0.4, 0.0)])
def test_equilibrium_biomass_matches_k_times_one_minus_h_over_r(stock, h, expected):
    assert stock.equilibrium_biomass(h) == pytest.approx(expected)


def test_yield_peaks_at_h_msy(stock):
    rates = np.linspace(0.01, 0.39, 200)
    yields = [stock.equilibrium_yield(h) for h in rates]
    assert rates[int(np.argmax(yields))] == pytest.approx(0.2, abs=0.01)


def test_simulation_converges_on_the_equilibrium(stock):
    path = stock.simulate(weeks=200, harvest_rate=0.1)
    assert path.final_biomass == pytest.approx(stock.equilibrium_biomass(0.1), rel=1e-6)


def test_simulation_shapes_and_accounting(stock):
    path = stock.simulate(weeks=52)
    assert len(path) == 52
    assert path.biomass.size == 53            # one more: the starting stock
    assert path.harvest == pytest.approx(0.10 * path.biomass[:-1])
    assert path.total_harvest == pytest.approx(path.harvest.sum())


def test_stock_never_goes_negative():
    heavy = FishStock(harvest_rate=0.39)
    assert heavy.simulate(weeks=200).biomass.min() >= 0


def test_overfishing_catches_less_than_msy(stock):
    at_msy = stock.simulate(52, harvest_rate=0.20)
    overfished = stock.simulate(52, harvest_rate=0.30)
    assert overfished.total_harvest < at_msy.total_harvest
    assert overfished.final_biomass < at_msy.final_biomass


@pytest.mark.parametrize("kwargs", [{"weeks": 0}, {"weeks": -4},
                                    {"harvest_rate": 1.5}])
def test_simulate_rejects_bad_arguments(stock, kwargs):
    with pytest.raises(ValueError):
        stock.simulate(**kwargs)


def test_open_weeks_must_match_the_horizon(stock):
    with pytest.raises(ValueError):
        stock.simulate(52, open_weeks=np.ones(10, dtype=bool))


# ----- task 3: the price model -----------------------------------------
def test_price_path_stays_inside_the_bounds():
    model = PriceModel(step_sd=3_000, seed=7)      # large steps, to stress it
    path = model.paths(weeks=500, n_paths=20)
    assert path.min() >= model.lower - 1e-9
    assert path.max() <= model.upper + 1e-9


def test_price_is_reproducible_for_a_fixed_seed():
    assert PriceModel(seed=7).path(52) == pytest.approx(PriceModel(seed=7).path(52))
    assert not np.allclose(PriceModel(seed=7).path(52), PriceModel(seed=8).path(52))


def test_price_paths_have_the_requested_shape():
    assert PriceModel(seed=7).paths(weeks=52, n_paths=100).shape == (100, 52)


@pytest.mark.parametrize("kwargs", [
    {"lower": 16_000, "upper": 9_000}, {"start": 20_000},
    {"start": 1_000}, {"step_sd": 0}, {"step_sd": -5},
])
def test_price_model_rejects_impossible_settings(kwargs):
    with pytest.raises(ValueError):
        PriceModel(**kwargs)


def test_revenue_converts_tonnes_to_kilograms(stock):
    path = stock.simulate(weeks=3)
    revenue = path.revenue([10_000.0, 10_000.0, 10_000.0])
    assert revenue == pytest.approx(path.harvest * KG_PER_TONNE * 10_000.0)


def test_revenue_needs_one_price_per_week(stock):
    with pytest.raises(ValueError):
        stock.simulate(weeks=5).revenue([12_000.0, 12_000.0])


# ----- tasks 4-5: risk --------------------------------------------------
def test_describe_reports_the_five_statistics():
    summary = RiskAssessor.describe([100.0, 110.0, 90.0, 105.0])
    assert set(summary) == {"mean", "median", "variance", "stdev", "cv"}
    assert summary["cv"] == pytest.approx(summary["stdev"] / summary["mean"])


def test_describe_needs_two_observations():
    with pytest.raises(ValueError):
        RiskAssessor.describe([100.0])


@pytest.mark.parametrize("cv,expected", [
    (0.05, RiskClass.LOW), (0.149, RiskClass.LOW),
    (0.15, RiskClass.MODERATE), (0.299, RiskClass.MODERATE),
    (0.30, RiskClass.HIGH), (1.2, RiskClass.HIGH),
])
def test_cv_bands(cv, expected):
    assert RiskAssessor().classify_cv(cv) is expected


def test_cv_is_invariant_to_a_change_of_units():
    """The point of using CV rather than variance: scale does not matter."""
    ugx = np.array([1e9, 1.2e9, 0.9e9, 1.1e9])
    thousands = ugx / 1_000
    assessor = RiskAssessor()
    assert (assessor.describe(ugx)["cv"]
            == pytest.approx(assessor.describe(thousands)["cv"]))
    # whereas the variance changes by a factor of a million
    assert (assessor.describe(ugx)["variance"]
            == pytest.approx(assessor.describe(thousands)["variance"] * 1e6))


@pytest.mark.parametrize("kwargs", [
    {"moderate_above": 0}, {"moderate_above": 0.4, "high_above": 0.2},
    {"moderate_above": -0.1},
])
def test_assessor_rejects_impossible_thresholds(kwargs):
    with pytest.raises(ValueError):
        RiskAssessor(**kwargs)


def test_assessor_is_frozen():
    with pytest.raises(Exception):
        RiskAssessor().moderate_above = 0.5


def test_tail_risk_orders_correctly():
    draws = np.random.default_rng(7).normal(1_000, 100, size=10_000)
    tail = RiskAssessor.tail_risk(draws, alpha=0.05)
    assert tail["expected_shortfall"] <= tail["var"] <= tail["mean"]
    assert tail["var_shortfall"] == pytest.approx(tail["mean"] - tail["var"])


def test_tail_risk_var_is_the_right_quantile():
    draws = np.arange(1, 1001, dtype=float)
    assert RiskAssessor.tail_risk(draws, alpha=0.10)["var"] == pytest.approx(
        np.percentile(draws, 10))


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, 2.0])
def test_tail_risk_rejects_a_bad_alpha(alpha):
    with pytest.raises(ValueError):
        RiskAssessor.tail_risk([1.0, 2.0, 3.0], alpha=alpha)


# ----- task 6: the scenario sweep ----------------------------------------
def test_scenario_composes_its_three_parts(scenario):
    assert isinstance(scenario.stock, FishStock)
    assert isinstance(scenario.price, PriceModel)
    assert isinstance(scenario.assessor, RiskAssessor)


def test_run_returns_a_consistent_year(scenario):
    outcome = scenario.run(weeks=52)
    assert outcome["revenue"].size == 52
    assert outcome["prices"].size == 52
    assert isinstance(outcome["risk"], RiskClass)


def test_monte_carlo_shape_and_reproducibility(scenario):
    first = scenario.monte_carlo(weeks=52, n_paths=200, seed=7)
    second = scenario.monte_carlo(weeks=52, n_paths=200, seed=7)
    assert first.shape == (200,)
    assert first == pytest.approx(second)


def test_sweep_reproduces_the_msy_result(scenario):
    table = scenario.sweep([0.05, 0.10, 0.20, 0.30], n_paths=200)
    assert list(table.index) == [0.05, 0.10, 0.20, 0.30]
    # h = 0.20 = r/2 maximises both the catch and the sustainable yield
    assert table["total_catch_t"].idxmax() == 0.20
    assert table["sustainable_yield_t"].idxmax() == 0.20
    # and over-fishing at 0.30 lands less than at 0.20 while leaving less stock
    assert table.loc[0.30, "total_catch_t"] < table.loc[0.20, "total_catch_t"]
    assert table.loc[0.30, "final_stock_t"] < table.loc[0.20, "final_stock_t"]


def test_sweep_final_stock_tracks_the_equilibrium(scenario):
    table = scenario.sweep([0.10, 0.20], n_paths=50)
    assert table["final_stock_t"].to_numpy() == pytest.approx(
        table["equilibrium_t"].to_numpy(), rel=1e-3)


def test_sweep_needs_at_least_one_rate(scenario):
    with pytest.raises(ValueError):
        scenario.sweep([])


# ----- extension: the closed season ---------------------------------------
def test_closed_season_mask_shuts_the_right_weeks():
    mask = closed_season_weeks(52, range(14, 22))
    assert mask.size == 52
    assert (~mask).sum() == 8
    assert not mask[14] and not mask[21] and mask[13] and mask[22]


@pytest.mark.parametrize("closed", [range(50, 60), [-1], [52]])
def test_closed_season_rejects_weeks_outside_the_year(closed):
    with pytest.raises(ValueError):
        closed_season_weeks(52, closed)


def test_no_harvest_during_the_closed_weeks(stock):
    mask = closed_season_weeks(52, range(14, 22))
    path = stock.simulate(52, open_weeks=mask)
    assert path.harvest[14:22] == pytest.approx(np.zeros(8))
    assert path.closed_weeks == 8
    assert (path.harvest[:14] > 0).all()


def test_closing_the_season_lets_the_stock_recover(stock):
    overfished = FishStock(harvest_rate=0.35)
    mask = closed_season_weeks(52, range(14, 22))
    assert (overfished.simulate(52, open_weeks=mask).final_biomass
            > overfished.simulate(52).final_biomass)
