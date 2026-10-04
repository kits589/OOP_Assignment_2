"""Tests for Mini-Project 5."""
import numpy as np
import pytest

from src.core.forecaster import Forecaster
from src.projects.p1_population import LinearTrendModel
from src.projects.p5_taxi import (
    FAMILY, DriftModel, ExponentialSmoothingModel, FleetPlanner, MarketModel,
    MovingAverageModel, Route, RouteLinearTrend, SeasonalNaiveModel, WalkForward,
    load_routes, simulate_weekly_demand)


@pytest.fixture
def ntinda() -> Route:
    return load_routes(["Kampala-Ntinda"])[0]


@pytest.fixture
def market() -> MarketModel:
    return MarketModel()


# ----- task 1: the Route class ----------------------------------------
def test_route_revenue_matches_a_hand_calculation(ntinda):
    # 35+40+42+50+55+60+48+52+47+45 = 474 passengers at UGX 2,000
    assert ntinda.passengers.sum() == 474
    assert ntinda.total_revenue == pytest.approx(474 * 2_000)
    assert ntinda.daily_revenue[0] == pytest.approx(35 * 2_000)


def test_route_statistics(ntinda):
    stats = ntinda.stats()
    assert stats["mean"] == pytest.approx(47.4)
    assert set(stats) == {"mean", "median", "variance", "stdev", "cv"}
    assert stats["cv"] == pytest.approx(stats["stdev"] / stats["mean"])


def test_route_dunders(ntinda):
    assert len(ntinda) == 10
    assert "Ntinda" in repr(ntinda)
    assert list(ntinda)[0] == (1, 35.0)


@pytest.mark.parametrize("kwargs", [
    {"passengers": []}, {"passengers": [10, -5]}, {"fare": 0},
    {"fare": -100}, {"fare": np.inf}, {"name": ""},
])
def test_route_rejects_bad_input(kwargs):
    defaults = {"name": "Test", "passengers": [10, 20], "fare": 1_000.0}
    with pytest.raises(ValueError):
        Route(**{**defaults, **kwargs})


def test_stats_need_two_days():
    with pytest.raises(ValueError):
        Route("One", [40], 2_000).stats()


# ----- task 2: the equilibrium ------------------------------------------
def test_equilibrium_matches_the_closed_form(market):
    # 120 - 0.02P = 10 + 0.03P  ->  110 = 0.05P  ->  P* = 2,200, Q* = 76
    price, quantity = market.equilibrium()
    assert price == pytest.approx(2_200.0)
    assert quantity == pytest.approx(76.0)


def test_equilibrium_clears_both_curves(market):
    price, quantity = market.equilibrium()
    assert market.quantity_demanded(price) == pytest.approx(quantity)
    assert market.quantity_supplied(price) == pytest.approx(quantity)


def test_the_current_fare_is_below_equilibrium(market):
    gap = market.gap_at(2_000)
    assert gap["demanded"] == pytest.approx(80.0)
    assert gap["supplied"] == pytest.approx(70.0)
    assert gap["excess_demand"] == pytest.approx(10.0)       # a shortage
    assert gap["distance_from_equilibrium"] == pytest.approx(-200.0)


def test_above_equilibrium_the_sign_flips(market):
    assert market.gap_at(2_500)["excess_demand"] < 0


def test_system_is_well_conditioned(market):
    assert market.determinant == pytest.approx(0.05)
    assert market.condition_number < 100


@pytest.mark.parametrize("kwargs", [
    {"demand_slope": 0}, {"supply_slope": -0.01},
    {"demand_intercept": 5, "supply_intercept": 10},
])
def test_market_rejects_impossible_curves(kwargs):
    with pytest.raises(ValueError):
        MarketModel(**kwargs)


def test_market_rejects_a_negative_price(market):
    with pytest.raises(ValueError):
        market.gap_at(-100)


# ----- task 3: the models ------------------------------------------------
def test_every_model_registers_in_the_taxi_family():
    catalogue = Forecaster.catalogue(FAMILY)
    assert set(catalogue) == {"moving-average", "ses", "linear-trend", "drift",
                              "seasonal-naive"}


def test_linear_trend_is_inherited_from_mini_project_one():
    """The point of RouteLinearTrend: reuse, not reimplementation."""
    assert issubclass(RouteLinearTrend, LinearTrendModel)
    assert RouteLinearTrend.fit is LinearTrendModel.fit
    assert RouteLinearTrend.predict is LinearTrendModel.predict
    assert RouteLinearTrend.family == FAMILY and LinearTrendModel.family == "population"


def test_moving_average_is_the_mean_of_the_window(ntinda):
    model = MovingAverageModel(3).fit(ntinda.passengers)
    assert model.predict(1)[0] == pytest.approx(np.mean([52, 47, 45]))
    assert model.predict(4) == pytest.approx([model.level_] * 4)


def test_moving_average_needs_a_full_window():
    with pytest.raises(ValueError):
        MovingAverageModel(3).fit([40.0, 42.0])


def test_ses_recursion_matches_a_hand_calculation():
    model = ExponentialSmoothingModel(alpha=0.5).fit([10.0, 20.0, 30.0])
    # l0 = 10; l1 = .5*20 + .5*10 = 15; l2 = .5*30 + .5*15 = 22.5
    assert model.level_ == pytest.approx(22.5)
    assert model.predict(2) == pytest.approx([22.5, 22.5])


def test_ses_at_alpha_one_is_the_naive_forecast(ntinda):
    model = ExponentialSmoothingModel(alpha=1.0).fit(ntinda.passengers)
    assert model.predict(1)[0] == pytest.approx(ntinda.passengers[-1])


@pytest.mark.parametrize("alpha", [0.0, -0.1, 1.5, 2.0])
def test_ses_rejects_an_alpha_outside_the_unit_interval(alpha):
    with pytest.raises(ValueError):
        ExponentialSmoothingModel(alpha)


def test_drift_extends_the_line_through_the_endpoints():
    model = DriftModel().fit([10.0, 20.0, 30.0, 40.0])
    assert model.drift_ == pytest.approx(10.0)
    assert model.predict(2) == pytest.approx([50.0, 60.0])


def test_seasonal_naive_repeats_the_previous_week():
    series = np.arange(1.0, 15.0)              # two clean weeks
    model = SeasonalNaiveModel(period=7).fit(series)
    assert model.predict(7) == pytest.approx(series[-7:])
    assert model.predict(9)[7] == pytest.approx(series[-7])


def test_seasonal_naive_needs_a_full_period():
    with pytest.raises(ValueError):
        SeasonalNaiveModel(period=7).fit(np.arange(5.0))


@pytest.mark.parametrize("build", [
    lambda: MovingAverageModel(3), lambda: ExponentialSmoothingModel(0.5),
    RouteLinearTrend, DriftModel, lambda: SeasonalNaiveModel(7),
])
def test_predicting_before_fitting_raises(build):
    with pytest.raises(RuntimeError):
        build().predict(1)


# ----- task 4: backtesting ------------------------------------------------
def test_backtest_covers_days_four_to_ten(ntinda):
    backtest = WalkForward(ntinda.passengers, first_day=4)
    assert len(backtest) == 7
    table = backtest.forecasts(lambda: MovingAverageModel(3))
    assert list(table.index) == list(range(4, 11))
    assert table["actual"].to_numpy() == pytest.approx(ntinda.passengers[3:])


def test_backtest_refits_at_every_origin(ntinda):
    """A later origin must see more data, so its forecast generally differs."""
    table = WalkForward(ntinda.passengers, 4).forecasts(lambda: MovingAverageModel(3))
    assert table["forecast"].nunique() > 1


@pytest.mark.parametrize("first_day", [0, 1, 11, 99, -2])
def test_backtest_rejects_an_impossible_start(ntinda, first_day):
    with pytest.raises(ValueError):
        WalkForward(ntinda.passengers, first_day)


def test_comparison_table_is_sorted_best_first(ntinda):
    table = WalkForward(ntinda.passengers, 4).compare({
        "moving-average": lambda: MovingAverageModel(3),
        "drift": DriftModel,
    })
    assert table["MAPE_%"].is_monotonic_increasing
    assert set(table.columns) == {"MAE", "RMSE", "MAPE_%"}


def test_comparison_needs_at_least_one_model(ntinda):
    with pytest.raises(ValueError):
        WalkForward(ntinda.passengers, 4).compare({})


def test_alpha_tuning_returns_the_minimising_value(ntinda):
    best, scores = WalkForward(ntinda.passengers, 4).tune_alpha()
    assert best in scores.index
    assert scores[best] == pytest.approx(scores.min())
    assert 0 < best <= 1


def test_alpha_grid_is_validated(ntinda):
    with pytest.raises(ValueError):
        WalkForward(ntinda.passengers, 4).tune_alpha(grid=[0.0, 0.5])
    with pytest.raises(ValueError):
        WalkForward(ntinda.passengers, 4).tune_alpha(grid=[0.5, 1.5])


# ----- task 6: fleet planning ---------------------------------------------
def test_capacity_matches_the_brief():
    planner = FleetPlanner()
    assert planner.capacity == 14 * 8 == 112


def test_vehicle_count_applies_the_buffer_and_rounds_up():
    planner = FleetPlanner()
    # 50 passengers * 1.15 = 57.5, / 112 = 0.51 -> 1 vehicle
    assert planner.vehicles_for(50) == 1
    # 200 * 1.15 = 230, / 112 = 2.05 -> 3 vehicles
    assert planner.vehicles_for(200) == 3
    # exactly at capacity before the buffer still needs two
    assert planner.vehicles_for(112) == 2


def test_an_empty_route_needs_no_vehicle_but_a_busy_one_needs_at_least_one():
    planner = FleetPlanner()
    assert planner.vehicles_for(0) == 0
    assert planner.vehicles_for(1) == 1


def test_plan_covers_every_route():
    plan = FleetPlanner().plan({"A": 45.0, "B": 64.0})
    assert list(plan.index) == ["A", "B"]
    assert (plan["vehicles"] >= 1).all()


@pytest.mark.parametrize("kwargs", [
    {"seats": 0}, {"trips_per_day": 0}, {"buffer": -0.1}, {"buffer": 10},
])
def test_planner_rejects_impossible_settings(kwargs):
    with pytest.raises(ValueError):
        FleetPlanner(**kwargs)


def test_planner_rejects_negative_passengers():
    with pytest.raises(ValueError):
        FleetPlanner().vehicles_for(-5)


# ----- extension: the weekly pattern ---------------------------------------
def test_simulation_is_reproducible_and_weekly():
    first = simulate_weekly_demand(60, seed=7)
    assert first == pytest.approx(simulate_weekly_demand(60, seed=7))
    assert first.size == 60
    by_weekday = [first[i::7].mean() for i in range(7)]
    # Friday (index 4) busiest, Sunday (index 6) quietest
    assert int(np.argmax(by_weekday)) == 4
    assert int(np.argmin(by_weekday)) == 6


@pytest.mark.parametrize("kwargs", [
    {"days": 0}, {"base": 0}, {"noise": -0.1},
    {"weekday_effect": [1.0] * 5},
])
def test_simulation_rejects_bad_settings(kwargs):
    with pytest.raises(ValueError):
        simulate_weekly_demand(**{"days": 60, **kwargs})


def test_seasonal_naive_beats_the_moving_average_on_weekly_data():
    """The whole point of the extension."""
    series = simulate_weekly_demand(60, seed=7)
    backtest = WalkForward(series, first_day=15)
    seasonal = backtest.score(lambda: SeasonalNaiveModel(7))
    moving = backtest.score(lambda: MovingAverageModel(3))
    assert seasonal.mae < moving.mae
    assert seasonal.mae < moving.mae * 0.75      # and by a wide margin
