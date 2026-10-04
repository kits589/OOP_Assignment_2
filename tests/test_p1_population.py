"""Tests for Mini-Project 1.

Style note: this build uses plain test functions and leans on
``@pytest.mark.parametrize`` to cover the rejection cases in one place.
"""
import math
import statistics

import numpy as np
import pytest

from src.core.forecaster import Forecaster
from src.core.metrics import ErrorReport, comparison_table
from src.projects.p1_population import (
    FAMILY, CAGRModel, ClassroomPlanner, DistrictPopulation, FibonacciRatioModel,
    LinearTrendModel, ModelSelector, Phase, bootstrap_intervals, fibonacci,
    load_districts, variance_comparison)


@pytest.fixture
def kampala() -> DistrictPopulation:
    """The brief's Kampala series, 2015-2024."""
    return load_districts(["Kampala"])[0]


@pytest.fixture
def straight_line() -> DistrictPopulation:
    """A perfectly linear series, where the linear model must be exact."""
    return DistrictPopulation("Linear", list(range(2015, 2025)),
                              [100 + 10 * i for i in range(10)])


# ----- the container ---------------------------------------------------
def test_repr_and_len_report_the_series(kampala):
    assert len(kampala) == 10
    assert "Kampala" in repr(kampala)
    assert "2015-2024" in repr(kampala)


def test_membership_checks_years(kampala):
    assert 2015 in kampala and 2024 in kampala
    assert 2030 not in kampala


@pytest.mark.parametrize("years,people", [
    ([2020, 2021], [1, 2, 3]),          # lengths differ
    ([2020, 2021], [100, -5]),          # negative population
    ([], []),                           # empty, the edge case the brief names
    ([2021, 2020], [100, 110]),         # years not increasing
    ([2020, 2021], [100, math.nan]),    # non-finite
])
def test_bad_input_is_rejected(years, people):
    with pytest.raises(ValueError):
        DistrictPopulation("Test", years, people)


@pytest.mark.parametrize("name", ["", "   ", None, 42])
def test_name_must_be_a_non_empty_string(name):
    with pytest.raises(ValueError):
        DistrictPopulation(name, [2020, 2021], [100, 110])


# ----- task 2: the ddof difference -------------------------------------
def test_stdlib_variance_matches_numpy_with_ddof_one(kampala):
    stdlib = kampala.describe_stdlib()
    assert stdlib["variance"] == pytest.approx(kampala.describe_numpy(ddof=1)["variance"])
    assert stdlib["mean"] == pytest.approx(statistics.mean(kampala.people.tolist()))


def test_numpy_default_is_the_population_variance(kampala):
    n = len(kampala)
    sample = kampala.describe_stdlib()["variance"]
    population = kampala.describe_numpy(ddof=0)["variance"]
    # dividing by n instead of n-1 scales the result by (n-1)/n
    assert population == pytest.approx(sample * (n - 1) / n)


# ----- task 3: growth --------------------------------------------------
def test_growth_rates_have_one_fewer_element(kampala):
    assert kampala.yoy_growth().size == len(kampala) - 1
    assert np.all(kampala.yoy_growth() > 0)


def test_compound_growth_reproduces_the_last_value(kampala):
    rate = kampala.compound_growth()
    span = int(kampala.years[-1] - kampala.years[0])
    assert kampala.people[0] * (1 + rate) ** span == pytest.approx(kampala.people[-1])


def test_growth_undefined_after_a_zero():
    zeroed = DistrictPopulation("Zero", [2020, 2021, 2022], [0, 100, 110])
    with pytest.raises(ValueError):
        zeroed.yoy_growth()


# ----- task 4: the models ----------------------------------------------
def test_every_model_is_registered_in_the_family():
    catalogue = Forecaster.catalogue(FAMILY)
    assert set(catalogue) == {"linear-trend", "cagr", "fibonacci-ratio"}
    assert catalogue["cagr"] is CAGRModel


def test_linear_model_is_exact_on_a_straight_line(straight_line):
    model = LinearTrendModel().fit(straight_line.people, straight_line.years)
    assert model.predict(2) == pytest.approx([200.0, 210.0])
    assert model.fitted() == pytest.approx(straight_line.people)


def test_cagr_model_reproduces_constant_growth():
    y = [100 * 1.05 ** i for i in range(8)]
    model = CAGRModel().fit(y)
    assert model.rate_ == pytest.approx(0.05)
    assert model.predict(1)[0] == pytest.approx(y[-1] * 1.05)


def test_fibonacci_ratios_converge_on_the_golden_ratio():
    seq = fibonacci(30)
    assert seq[:6].tolist() == [1, 1, 2, 3, 5, 8]
    assert seq[-1] / seq[-2] == pytest.approx((1 + math.sqrt(5)) / 2, abs=1e-9)


def test_fibonacci_model_grows_at_roughly_the_golden_ratio(kampala):
    model = FibonacciRatioModel().fit(kampala.people)
    forecast = model.predict(3)
    assert forecast[0] / kampala.people[-1] == pytest.approx(1.618, abs=0.01)


@pytest.mark.parametrize("model_cls", [LinearTrendModel, CAGRModel, FibonacciRatioModel])
def test_predict_before_fit_raises(model_cls):
    with pytest.raises(RuntimeError):
        model_cls().predict(3)


@pytest.mark.parametrize("horizon", [0, -1, 2.5, "3", True])
def test_horizon_must_be_a_positive_integer(kampala, horizon):
    model = CAGRModel().fit(kampala.people)
    with pytest.raises(ValueError):
        model.predict(horizon)


def test_fit_rejects_a_series_shorter_than_min_obs():
    with pytest.raises(ValueError):
        LinearTrendModel().fit([100.0])


# ----- tasks 5-6: selection --------------------------------------------
def test_split_keeps_the_boundary_year_in_train(kampala):
    train, test = kampala.split_at(2021)
    assert train.years[-1] == 2021 and test.years[0] == 2022
    assert len(train) == 7 and len(test) == 3
    assert kampala.phase_of(2021) is Phase.TRAIN
    assert kampala.phase_of(2022) is Phase.TEST


@pytest.mark.parametrize("year", [2014, 2024, 2030])
def test_split_outside_the_series_raises(kampala, year):
    with pytest.raises(ValueError):
        kampala.split_at(year)


def test_cagr_beats_the_others_on_every_district():
    selector = ModelSelector()
    for district in load_districts():
        assert selector.best(district)[0] == "cagr"


def test_comparison_table_is_sorted_best_first(kampala):
    table = ModelSelector().table(kampala)
    assert list(table.columns) == ["MAE", "RMSE", "MAPE_%"]
    assert table["MAPE_%"].is_monotonic_increasing
    assert table.index[0] == "cagr"


def test_error_report_orders_by_mape_and_formats():
    good = ErrorReport.score("good", [100, 200], [101, 201])
    bad = ErrorReport.score("bad", [100, 200], [150, 250])
    assert good < bad
    assert min(bad, good) is good
    assert "MAE" in format(good, ".2f")
    with pytest.raises(ValueError):
        ErrorReport.score("zero", [0, 100], [1, 101]).mape


def test_error_report_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        ErrorReport.score("x", [1, 2, 3], [1, 2])
    with pytest.raises(ValueError):
        comparison_table([])


def test_forecast_variance_is_lower_than_actual(kampala):
    model = ModelSelector().fit_best(kampala)
    result = variance_comparison(kampala, model.predict(5))
    assert 0 < result["ratio"] < 1


# ----- extension and planning ------------------------------------------
def test_bootstrap_interval_brackets_the_point_forecast(kampala):
    model = ModelSelector().fit_best(kampala)
    interval = bootstrap_intervals(model, horizon=5, n_boot=1000, seed=7)
    assert np.all(interval["lower"] <= interval["upper"])
    assert interval["point"].size == 5


def test_bootstrap_is_reproducible_for_a_fixed_seed(kampala):
    model = ModelSelector().fit_best(kampala)
    first = bootstrap_intervals(model, n_boot=500, seed=7)["lower"]
    second = bootstrap_intervals(model, n_boot=500, seed=7)["lower"]
    assert first == pytest.approx(second)


@pytest.mark.parametrize("kwargs", [
    {"age_share": 0.0}, {"age_share": 1.5}, {"class_size": 0}, {"class_size": -3},
])
def test_planner_rejects_impossible_parameters(kwargs):
    with pytest.raises(ValueError):
        ClassroomPlanner(**kwargs)


def test_planner_matches_a_hand_calculation():
    planner = ClassroomPlanner()
    # 100k extra people * 18% = 18,000 pupils; 18,000 / 53 = 339.6 -> 340 rooms
    assert planner.additional(1_000, 1_100) == 340
    assert planner.pupils(100) == pytest.approx(18_000)


def test_planner_returns_zero_when_population_shrinks():
    assert ClassroomPlanner().additional(1_200, 1_100) == 0
