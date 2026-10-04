"""Tests for Mini-Project 4."""
import math

import numpy as np
import pytest
from scipy.spatial.distance import cosine as scipy_cosine

from src.projects.p4_rainfall import (
    CROP_RULES, MONTHS, CropRule, Modality, Region, SeasonProfile, SimilarityStudy,
    Suitability, SuitabilityGrid, cosine_similarity, euclidean_distance,
    load_nasa_power, load_regions, pearson_correlation)


@pytest.fixture
def regions() -> list[Region]:
    return load_regions()


@pytest.fixture
def kampala(regions) -> Region:
    return regions[0]


# ----- task 1: the Region class ---------------------------------------
def test_region_summary_statistics(kampala):
    assert kampala.annual_total == pytest.approx(1_600.0)
    assert kampala.mean_monthly == pytest.approx(1_600.0 / 12)
    assert kampala.wettest_month == "May"
    assert kampala.driest_month == "Sep"
    assert 0 < kampala.cv < 1


def test_region_indexing_and_iteration(kampala):
    assert kampala["May"] == 220.0
    assert kampala[4] == 220.0
    assert len(kampala) == 12
    assert list(kampala)[0] == ("Jan", 120.0)


def test_region_rejects_an_unknown_month(kampala):
    with pytest.raises(KeyError):
        kampala["Smarch"]


@pytest.mark.parametrize("rainfall", [
    [100] * 11,                       # too few months
    [100] * 13,                       # too many
    [100] * 11 + [-5],                # negative
    [100] * 11 + [math.nan],          # non-finite
    [],                               # empty
])
def test_region_rejects_bad_rainfall(rainfall):
    with pytest.raises(ValueError):
        Region("Test", rainfall)


@pytest.mark.parametrize("name", ["", "   ", None, 12])
def test_region_requires_a_name(name):
    with pytest.raises(ValueError):
        Region(name, [100] * 12)


def test_cv_separates_seasonal_from_even_regions(regions):
    by_cv = {r.name: r.cv for r in regions}
    # Gulu's rain is concentrated into one long season, so it varies most
    assert max(by_cv, key=by_cv.__getitem__) == "Gulu"


# ----- task 2: crop rules -----------------------------------------------
@pytest.mark.parametrize("mm,expected", [
    (50, Suitability.DROUGHT), (104, Suitability.DROUGHT),
    (105, Suitability.GOOD), (167, Suitability.GOOD),
    (168, Suitability.WATERLOGGING), (400, Suitability.WATERLOGGING),
])
def test_sorghum_band_boundaries(mm, expected):
    sorghum = CROP_RULES[0]
    assert sorghum.crop == "Sorghum"
    assert sorghum.classify(mm) is expected


def test_suitability_scores_map_to_minus_one_zero_one():
    assert Suitability.DROUGHT.score == -1
    assert Suitability.GOOD.score == 0
    assert Suitability.WATERLOGGING.score == 1


@pytest.mark.parametrize("kwargs", [
    {"min_mm": 200, "max_mm": 100},    # inverted
    {"min_mm": -10, "max_mm": 100},    # negative
    {"min_mm": 100, "max_mm": 100},    # empty band
    {"crop": ""},                      # unnamed
])
def test_crop_rule_rejects_impossible_bands(kwargs):
    defaults = {"crop": "Test", "min_mm": 100.0, "max_mm": 200.0}
    with pytest.raises(ValueError):
        CropRule(**{**defaults, **kwargs})


def test_crop_rule_is_frozen():
    with pytest.raises(Exception):
        CROP_RULES[0].min_mm = 0


def test_every_crop_rule_cites_a_source():
    assert all(rule.source for rule in CROP_RULES)
    assert all("FAO" in rule.source for rule in CROP_RULES)


def test_classify_rejects_impossible_rainfall():
    with pytest.raises(ValueError):
        CROP_RULES[0].classify(-5)


# ----- task 3: cosine similarity done properly ---------------------------
def test_cosine_matches_scipy_on_every_pair(regions):
    for i, a in enumerate(regions):
        for b in regions[i + 1:]:
            ours = cosine_similarity(a.rainfall, b.rainfall)
            theirs = 1 - float(scipy_cosine(a.rainfall, b.rainfall))
            assert ours == pytest.approx(theirs, abs=1e-12)


def test_cosine_is_one_for_identical_shapes():
    assert cosine_similarity([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)
    # and still 1 when one is a scaled copy: cosine ignores magnitude entirely
    assert cosine_similarity([1, 2, 3], [10, 20, 30]) == pytest.approx(1.0)


def test_cosine_is_zero_for_orthogonal_vectors():
    assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)


def test_cosine_rejects_a_zero_vector():
    with pytest.raises(ValueError):
        cosine_similarity([0, 0, 0], [1, 2, 3])


@pytest.mark.parametrize("fn", [cosine_similarity, pearson_correlation,
                                euclidean_distance])
def test_measures_reject_mismatched_lengths(fn):
    with pytest.raises(ValueError):
        fn([1, 2, 3], [1, 2])


# ----- task 4: the three matrices -----------------------------------------
def test_pearson_separates_magnitude_from_timing():
    """Cosine says 'identical', Pearson says 'identical' -- for a scaled copy."""
    base = [10, 20, 30, 40]
    assert cosine_similarity(base, np.array(base) * 3) == pytest.approx(1.0)
    assert pearson_correlation(base, np.array(base) * 3) == pytest.approx(1.0)
    # but add a constant and only cosine stays high
    shifted = np.array(base) + 500
    assert cosine_similarity(base, shifted) > 0.8
    assert pearson_correlation(base, shifted) == pytest.approx(1.0)


def test_cosine_rates_positive_vectors_as_similar(regions):
    """The point of task 4: all-positive data forces cosine high."""
    study = SimilarityStudy(regions)
    cosine = study.matrix("cosine")
    off_diagonal = cosine.to_numpy()[~np.eye(len(regions), dtype=bool)]
    assert off_diagonal.min() > 0.7          # everything looks "similar"
    pearson = study.matrix("pearson").to_numpy()[~np.eye(len(regions), dtype=bool)]
    assert abs(pearson).max() < 0.5          # yet the timing is unrelated


def test_matrices_are_the_right_shape_and_symmetric(regions):
    study = SimilarityStudy(regions)
    for name in ("cosine", "pearson", "euclidean"):
        matrix = study.matrix(name)
        assert matrix.shape == (3, 3)
        assert matrix.to_numpy() == pytest.approx(matrix.to_numpy().T)


def test_diagonals_take_their_expected_values(regions):
    study = SimilarityStudy(regions)
    assert np.diag(study.matrix("cosine")) == pytest.approx([1.0] * 3)
    assert np.diag(study.matrix("euclidean")) == pytest.approx([0.0] * 3)


def test_verification_table_reports_zero_difference(regions):
    checks = SimilarityStudy(regions).verify_against_scipy()
    assert len(checks) == 3
    assert checks["abs_diff"].max() < 1e-12


def test_study_rejects_an_unknown_measure_or_too_few_regions(regions):
    with pytest.raises(ValueError):
        SimilarityStudy(regions).matrix("manhattan")
    with pytest.raises(ValueError):
        SimilarityStudy(regions[:1])


# ----- task 5: season detection --------------------------------------------
def test_detected_modality_matches_known_climatology(regions):
    verdicts = {r.name: SeasonProfile(r).modality for r in regions}
    # Gulu has one long season; Mbarara has two
    assert verdicts["Gulu"] is Modality.UNIMODAL
    assert verdicts["Mbarara"] is Modality.BIMODAL


def test_mbarara_peaks_fall_in_the_right_months(regions):
    mbarara = next(r for r in regions if r.name == "Mbarara")
    assert SeasonProfile(mbarara).peak_months() == ["Apr", "Oct"]


def test_gulu_peaks_once_in_august(regions):
    gulu = next(r for r in regions if r.name == "Gulu")
    assert SeasonProfile(gulu).peak_months() == ["Aug"]


def test_detection_wraps_around_the_calendar():
    """A December peak must be found even though the array ends there."""
    profile = SeasonProfile(Region("Wrap", [200, 60, 20, 20, 20, 20,
                                            20, 20, 20, 20, 60, 190]))
    assert profile.peak_months() == ["Jan"]
    assert profile.modality is Modality.UNIMODAL


def test_a_flat_profile_has_no_peaks():
    profile = SeasonProfile(Region("Flat", [100] * 12))
    assert profile.peak_indices().size == 0
    assert profile.modality is Modality.UNIMODAL


def test_a_low_prominence_wobble_is_not_a_season():
    """One long season with a shallow notch in it is still one season.

    May (200) and Jul (195) are both local maxima, but the Jun dip only falls
    to 185 -- a prominence of 10 mm against an annual range of 180 mm. A rule
    that counted it would report two rainy seasons where the rain never
    actually stopped.
    """
    notched = Region("Notched", [20, 40, 90, 150, 200, 185,
                                 195, 160, 100, 50, 25, 20])
    profile = SeasonProfile(notched, prominence_share=0.25)
    assert profile.peak_months() == ["May"]
    assert profile.modality is Modality.UNIMODAL
    # drop the threshold far enough and the notch does register
    assert SeasonProfile(notched, prominence_share=0.05).modality is Modality.BIMODAL


@pytest.mark.parametrize("kwargs", [{"prominence_share": 0.0},
                                    {"prominence_share": 1.0},
                                    {"min_separation": 0}])
def test_profile_rejects_impossible_settings(kampala, kwargs):
    with pytest.raises(ValueError):
        SeasonProfile(kampala, **kwargs)


@pytest.mark.parametrize("count,expected", [
    (0, Modality.UNIMODAL), (1, Modality.UNIMODAL),
    (2, Modality.BIMODAL), (3, Modality.POLYMODAL),
])
def test_modality_bands(count, expected):
    assert Modality.of(count) is expected


# ----- tasks 2 and 6: the grid ----------------------------------------------
def test_grid_labels_every_month_for_every_region(regions):
    grid = SuitabilityGrid(regions)
    labels = grid.labels("Sorghum")
    assert labels.shape == (12, 3)
    assert list(labels.index) == list(MONTHS)
    assert set(labels.to_numpy().ravel()) <= {str(s) for s in Suitability}


def test_grid_scores_agree_with_labels(regions):
    grid = SuitabilityGrid(regions)
    labels, scores = grid.labels("Cotton"), grid.scores("Cotton")
    assert ((labels == "good") == (scores == 0)).all().all()


def test_overview_counts_suitable_months(regions):
    overview = SuitabilityGrid(regions).overview()
    assert list(overview.columns) == ["Sorghum", "Groundnut", "Cotton"]
    assert (overview >= 0).all().all() and (overview <= 12).all().all()


def test_grid_rejects_an_unknown_crop_or_empty_inputs(regions):
    with pytest.raises(ValueError):
        SuitabilityGrid(regions).labels("Quinoa")
    with pytest.raises(ValueError):
        SuitabilityGrid([])
    with pytest.raises(ValueError):
        SuitabilityGrid(regions, crops=[])


def test_crop_lookup_is_case_insensitive(regions):
    grid = SuitabilityGrid(regions)
    assert grid.labels("sorghum").equals(grid.labels("Sorghum"))


# ----- loading --------------------------------------------------------------
def test_load_regions_rejects_an_unknown_name():
    with pytest.raises(ValueError):
        load_regions(["Narnia"])


def test_nasa_loader_reads_a_power_style_export(tmp_path):
    path = tmp_path / "power.csv"
    header = "\n".join(["-BEGIN HEADER-", "some metadata", "-END HEADER-"])
    body = ("YEAR,JAN,FEB,MAR,APR,MAY,JUN,JUL,AUG,SEP,OCT,NOV,DEC\n"
            "2020,10,20,30,40,50,60,70,80,90,100,110,120\n"
            "2021,15,25,35,45,55,65,75,85,95,105,115,125\n")
    path.write_text(f"{header}\n{body}", encoding="utf-8")
    table = load_nasa_power(path)
    assert list(table.columns) == list(MONTHS)
    assert table.loc[2020, "Jan"] == pytest.approx(10.0)
    assert len(table) == 2


def test_nasa_loader_rejects_a_file_that_is_not_one(tmp_path):
    path = tmp_path / "wrong.csv"
    path.write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_nasa_power(path)
    with pytest.raises(FileNotFoundError):
        load_nasa_power(tmp_path / "absent.csv")
