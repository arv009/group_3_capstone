"""Smoke tests: every Streamlit page runs without errors on the demo bundle (no API calls). Run: pytest tests/"""
import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP))          # pages import the shared `lib` module from app/

PAGES = ["Home.py", "pages/1_Portfolio_Dashboard.py", "pages/2_Applicant_Review.py", "pages/3_Model_Card.py",
         "pages/4_AI_Analyst.py"]


@pytest.mark.parametrize("page", PAGES)
def test_page_runs(page):
    at = AppTest.from_file(str(APP / page), default_timeout=180).run()
    assert not at.exception, [e.value for e in at.exception]


def test_dashboard_cutoff_slider_changes_kpis():
    at = AppTest.from_file(str(APP / "pages/1_Portfolio_Dashboard.py"), default_timeout=180).run()
    before = at.metric[0].value
    at.sidebar.slider[0].set_value(0.30).run()
    assert not at.exception
    assert at.metric[0].value != before          # approval rate moves with the cutoff


@pytest.mark.parametrize("group", ["Under review", "Decline", "Approve"])
def test_applicant_review_shows_cached_letter(group):
    at = AppTest.from_file(str(APP / "pages/2_Applicant_Review.py"), default_timeout=180).run()
    at.sidebar.radio[0].set_value(group).run()
    assert not at.exception
    assert any("Subject:" in m.value for m in at.markdown), "cached letter not shown"
    assert at.text_area[0].value.strip().endswith("The Lending Team")
    badges = [a.value for a in list(at.success) + list(at.warning)]
    assert any("claims supported" in b for b in badges), "letter audit badge not shown"


def test_dashboard_has_personas_tab():
    at = AppTest.from_file(str(APP / "pages/1_Portfolio_Dashboard.py"), default_timeout=180).run()
    assert [t.label for t in at.tabs] == ["Portfolio", "Personas"]
    assert any("Newer worker, ordinary loan" in m.value for m in at.markdown)
