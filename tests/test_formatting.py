"""Tests for how feature values are shown to people (tables and letter fact sheets)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from secondlook.explain import format_duration, format_value  # noqa: E402


@pytest.mark.parametrize("years,expected", [
    (3 / 365.25, "less than a week"),
    (7 / 365.25, "about 1 week"),
    (0.1, "about 5 weeks"),          # 36.5 days: the case that read as "0.1 years" in a live letter
    (0.25, "about 3 months"),
    (1 / 12, "about 4 weeks"),       # 30 days: still weeks (under 8 weeks)
    (0.2, "about 2 months"),
    (0.97, "1.0 years"),             # rounds to 12 months: shown in years, never "about 12 months"
    (1.2, "1.2 years"),
    (13.14, "13.1 years"),
])
def test_format_duration(years, expected):
    assert format_duration(years) == expected


def test_time_features_use_durations():
    assert format_value("BUR_YEARS_SINCE_LAST_LOAN", 0.1) == "about 5 weeks"
    assert format_value("YEARS_EMPLOYED", 0.5) == "about 6 months"


def test_normalised_building_scores_are_not_durations():
    assert format_value("YEARS_BUILD_AVG", 0.62) == "0.62"


@pytest.mark.parametrize("value,expected", [(-13, "never late"), (0, "never late"), (1, "1 day"), (12, "12 days")])
def test_max_days_late_never_negative(value, expected):
    assert format_value("INST_MAX_DAYS_LATE", value) == expected


def test_car_age():
    assert format_value("OWN_CAR_AGE", 3.0) == "3 years old"
    assert format_value("OWN_CAR_AGE", 1.0) == "1 year old"


def test_missing_value():
    assert format_value("YEARS_EMPLOYED", float("nan")) == "not available"
