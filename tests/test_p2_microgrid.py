"""Tests for Mini-Project 2."""
import numpy as np
import pandas as pd
import pytest

from src.projects.p2_microgrid import (
    Conditioning, DemandLog, DispatchStatus, HybridMicroGrid, MicroGrid,
    TariffSchedule, monte_carlo_sensitivity)


@pytest.fixture
def grid() -> MicroGrid:
    return MicroGrid()


@pytest.fixture
def demand_csv(tmp_path):
    """Thirty seeded days written to a temporary file."""
    path = tmp_path / "demand.csv"
    log = DemandLog()
    log.generate(path, days=30, seed=7)
    return log, path


# ----- task 1: conditioning ------------------------------------------
def test_default_system_matches_the_brief(grid):
    assert grid.A.tolist() == [[3.0, 2.0], [4.0, 1.0]]
    assert grid.order == 2 and len(grid) == 2
    assert grid.sources == ("solar", "battery")


def test_determinant_and_condition_number(grid):
    # det = 3*1 - 2*4 = -5, computed by hand
    assert grid.determinant == pytest.approx(-5.0)
    assert grid.condition_number == pytest.approx(5.828427, abs=1e-5)
    assert grid.rank == 2
    assert grid.conditioning is Conditioning.WELL_CONDITIONED
    assert grid.conditioning.usable


def test_diagnose_reports_every_figure(grid):
    report = grid.diagnose()
    assert report["full_rank"] is True
    assert report["verdict"] == "well-conditioned"


def test_singular_system_is_detected_and_refused():
    singular = MicroGrid([[1.0, 2.0], [2.0, 4.0]])
    assert singular.determinant == pytest.approx(0.0)
    assert singular.conditioning is Conditioning.SINGULAR
    assert not singular.conditioning.usable
    with pytest.raises(ValueError, match="singular"):
        singular.solve_day(10, 20)


@pytest.mark.parametrize("matrix", [
    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],      # not square
    [[1.0, np.nan], [2.0, 3.0]],             # not finite
    [1.0, 2.0],                              # not 2-D
])
def test_bad_matrices_are_rejected(matrix):
    with pytest.raises(ValueError):
        MicroGrid(matrix)


def test_name_counts_must_match_the_matrix():
    with pytest.raises(ValueError):
        MicroGrid([[3.0, 2.0], [4.0, 1.0]], sources=("solar",))


# ----- tasks 2-3: solving ---------------------------------------------
def test_solve_day_matches_a_hand_calculation(grid):
    # x = (2*D2 - D1)/5, y = (4*D1 - 3*D2)/5
    assert grid.solve_day(100, 120) == pytest.approx([28.0, 8.0])
    assert grid.solve_day(90, 110) == pytest.approx([26.0, 6.0])


def test_solve_day_checks_the_argument_count(grid):
    with pytest.raises(ValueError):
        grid.solve_day(100)


def test_solution_satisfies_the_system(grid):
    demands = np.array([[100.0, 110.0, 95.0], [120.0, 128.0, 118.0]])
    allocation = grid.solve(demands)
    assert grid.A @ allocation == pytest.approx(demands)


def test_loop_and_vectorised_solvers_agree(grid):
    demands = np.array([[100.0, 110.0, 95.0], [120.0, 128.0, 118.0]])
    assert grid.solve(demands) == pytest.approx(grid.solve_sequentially(demands))


@pytest.mark.parametrize("demands", [
    [[100.0], [120.0], [90.0]],      # too many rows
    [[-10.0], [120.0]],              # negative demand
    [[np.inf], [120.0]],             # non-finite
])
def test_bad_demands_are_rejected(grid, demands):
    with pytest.raises(ValueError):
        grid.solve(demands)


# ----- task 4: infeasibility -------------------------------------------
def test_infeasible_day_is_detected(grid):
    # y < 0 whenever D2 > 4*D1/3; 100 and 140 breaches that
    allocation = grid.solve([[100.0], [140.0]])
    assert allocation[1, 0] < 0
    assert grid.infeasible(allocation).tolist() == [True]


def test_feasible_day_is_not_flagged(grid):
    assert grid.infeasible(grid.solve([[100.0], [120.0]])).tolist() == [False]


def test_repair_clips_and_reports_what_is_unmet(grid):
    demands = np.array([[100.0], [140.0]])
    repaired, unmet = grid.repair(grid.solve(demands), demands)
    assert np.all(repaired >= 0)
    assert unmet[0] > 0                       # the clip really does break the equality
    assert unmet[0] == pytest.approx(
        float(np.linalg.norm(grid.A @ repaired - demands)))


def test_repair_leaves_a_feasible_day_alone(grid):
    demands = np.array([[100.0], [120.0]])
    repaired, unmet = grid.repair(grid.solve(demands), demands)
    assert repaired.ravel() == pytest.approx([28.0, 8.0])
    assert unmet[0] == pytest.approx(0.0, abs=1e-9)


def test_dispatch_reports_statuses_and_unmet_energy(grid, demand_csv):
    log, path = demand_csv
    report = grid.dispatch(log.as_matrix(log.load(path)))
    assert len(report) == 30
    assert set(report.table["status"]) <= {str(DispatchStatus.MET),
                                           str(DispatchStatus.REPAIRED)}
    assert report.n_repaired == len(report.repaired_days())
    assert (report.table[["solar", "battery"]] >= 0).all().all()
    if report.n_repaired:
        assert report.total_unmet > 0


# ----- task 5: volatility -----------------------------------------------
def test_battery_is_the_more_volatile_source(grid, demand_csv):
    log, path = demand_csv
    report = grid.dispatch(log.as_matrix(log.load(path)))
    volatility = report.volatility()
    assert volatility.loc["battery", "cv"] > volatility.loc["solar", "cv"]
    assert set(volatility.columns) == {"mean_kwh", "variance", "stdev", "cv"}


def test_volatility_needs_more_than_one_day(grid):
    with pytest.raises(ValueError):
        grid.volatility(np.array([[28.0], [8.0]]))


# ----- task 6: cost ------------------------------------------------------
def test_tariff_costs_a_dispatch_by_hand():
    tariff = TariffSchedule()
    table = pd.DataFrame({"solar": [10.0, 20.0], "battery": [1.0, 2.0]})
    # 10*150 + 1*450 = 1,950 and 20*150 + 2*450 = 3,900
    assert tariff.daily(table).tolist() == pytest.approx([1950.0, 3900.0])
    assert tariff.total(table) == pytest.approx(5850.0)
    assert tariff.monthly(table, days=30) == pytest.approx(2925.0 * 30)


def test_tariff_rejects_negative_rates():
    with pytest.raises(ValueError):
        TariffSchedule(rates={"solar": -1.0})
    with pytest.raises(ValueError):
        TariffSchedule(rates={})


def test_tariff_needs_a_column_it_recognises():
    with pytest.raises(ValueError):
        TariffSchedule().daily(pd.DataFrame({"wind": [1.0]}))


# ----- the two input modes ------------------------------------------------
def test_generated_csv_is_reproducible(tmp_path):
    log = DemandLog()
    first = log.generate(tmp_path / "a.csv", days=30, seed=7)
    second = log.generate(tmp_path / "b.csv", days=30, seed=7)
    assert first.equals(second)
    assert len(first) == 30
    assert (first[["D1", "D2"]] > 0).all().all()


def test_generated_csv_carries_a_weekly_pattern(tmp_path):
    frame = DemandLog().generate(tmp_path / "c.csv", days=70, seed=7,
                                 weekly_swing=0.3, noise=0.0)
    by_weekday = frame.groupby((frame["day"] - 1) % 7)["D1"].mean()
    assert by_weekday.max() / by_weekday.min() > 1.3


def test_loading_rejects_a_missing_or_broken_file(tmp_path):
    log = DemandLog()
    with pytest.raises(FileNotFoundError):
        log.load(tmp_path / "absent.csv")
    (tmp_path / "wrong.csv").write_text("day,X\n1,100\n", encoding="utf-8")
    with pytest.raises(ValueError):
        log.load(tmp_path / "wrong.csv")
    (tmp_path / "negative.csv").write_text("day,D1,D2\n1,100,-5\n", encoding="utf-8")
    with pytest.raises(ValueError):
        log.load(tmp_path / "negative.csv")


@pytest.mark.parametrize("text", ["", "   ", "abc", "-5", "nan", "inf", None])
def test_interactive_parse_rejects_bad_entries(text):
    with pytest.raises(ValueError):
        DemandLog.parse(text)


@pytest.mark.parametrize("text,expected", [
    ("150", 150.0), (" 99.5 ", 99.5), ("1,250.5", 1250.5), ("0", 0.0),
])
def test_interactive_parse_accepts_good_entries(text, expected):
    assert DemandLog.parse(text) == pytest.approx(expected)


def test_prompt_reprompts_then_succeeds():
    answers = iter(["oops", "-3", "150"])
    said = []
    value = DemandLog.ask("D1: ", reader=lambda _: next(answers), writer=said.append)
    assert value == pytest.approx(150.0)
    assert len(said) == 2                      # it complained twice before accepting


def test_prompt_gives_up_after_the_attempt_limit():
    with pytest.raises(ValueError, match="no valid entry"):
        DemandLog.ask("D1: ", reader=lambda _: "nonsense",
                      writer=lambda _: None, attempts=2)


def test_ask_all_collects_one_value_per_load():
    answers = iter(["100", "120"])
    collected = DemandLog().ask_all(reader=lambda _: next(answers),
                                    writer=lambda _: None)
    assert collected == pytest.approx([100.0, 120.0])


# ----- extension: the 3x3 system and sensitivity ---------------------------
def test_hybrid_inherits_and_solves_a_3x3_system():
    hybrid = HybridMicroGrid()
    assert isinstance(hybrid, MicroGrid)
    assert hybrid.order == 3
    assert hybrid.determinant == pytest.approx(-20.0)
    allocation = hybrid.solve_day(120, 140, 200)
    assert hybrid.A @ allocation == pytest.approx([120.0, 140.0, 200.0])


def test_hybrid_reuses_the_parent_dispatch_machinery():
    hybrid = HybridMicroGrid()
    demands = np.array([[120.0, 118.0], [140.0, 139.0], [200.0, 205.0]])
    report = hybrid.dispatch(demands)
    assert list(report.table.columns)[:3] == ["solar", "battery", "diesel"]
    assert len(report) == 2


def test_hybrid_rejects_a_system_of_the_wrong_size():
    with pytest.raises(ValueError):
        HybridMicroGrid([[3.0, 2.0], [4.0, 1.0]], sources=("a", "b"),
                        loads=("D1", "D2"))


def test_a_dependent_third_load_makes_the_system_singular():
    dependent = HybridMicroGrid.with_dependent_load()
    assert dependent.determinant == pytest.approx(0.0)
    assert dependent.rank == 2                 # one equation short of full rank
    assert dependent.conditioning is Conditioning.SINGULAR
    with pytest.raises(ValueError):
        dependent.solve_day(100, 120, 220)


def test_sensitivity_amplification_respects_the_condition_number(grid):
    result = monte_carlo_sensitivity(grid, [100.0, 120.0], draws=1000, seed=7)
    assert result["solutions"].shape == (2, 1000)
    assert result["max_amplification"] <= result["condition_number"] + 1e-9
    assert result["mean_amplification"] > 0


def test_sensitivity_is_reproducible(grid):
    first = monte_carlo_sensitivity(grid, [100.0, 120.0], draws=200, seed=7)
    second = monte_carlo_sensitivity(grid, [100.0, 120.0], draws=200, seed=7)
    assert first["solutions"] == pytest.approx(second["solutions"])


@pytest.mark.parametrize("kwargs", [
    {"spread": 0.0}, {"spread": 1.0}, {"draws": 0},
])
def test_sensitivity_rejects_impossible_settings(grid, kwargs):
    with pytest.raises(ValueError):
        monte_carlo_sensitivity(grid, [100.0, 120.0], **kwargs)
