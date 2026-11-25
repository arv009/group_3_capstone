"""Smoke tests: every Streamlit page runs without errors on the demo bundle (no API calls). Run: pytest tests/"""
import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP))          # pages import the shared `lib` module from app/

PAGES = ["Home.py", "pages/1_Portfolio_Dashboard.py", "pages/2_Applicant_Review.py", "pages/3_Model_Card.py", "pages/4_AI_Analyst.py"]


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
    labels = [m.label for m in at.metric]
    assert "Claims supported" in labels, "independent audit not shown next to the letter"


def test_model_card_shows_fairness_audit_and_decision():
    at = AppTest.from_file(str(APP / "pages/3_Model_Card.py"), default_timeout=180).run()
    assert not at.exception
    assert len(at.tabs) == 3                                         # gender, age band, credit file
    assert any("Decision: we ship the model with all features" in s.value for s in at.success)
    assert any("bootstrap resamples" in c.value for c in at.caption)
    phase2 = next(m.value for m in at.markdown if "**Path to approval**" in m.value)
    assert "**90.1%** of under-review and **31.3%** of declined" in phase2       # notebook 06
    assert "**91.6%** of the 178 claims in the 30 demo letters" in phase2        # notebook 07


def test_dashboard_personas_tab():
    at = AppTest.from_file(str(APP / "pages/1_Portfolio_Dashboard.py"), default_timeout=180).run()
    assert not at.exception
    assert [t.label for t in at.tabs] == ["Overview", "Personas"]
    assert any(m.label == "Have a path to approval" for m in at.metric)
    assert any("**Thin file**" in m.value for m in at.markdown)


def test_ai_analyst_page_without_questions():
    at = AppTest.from_file(str(APP / "pages/4_AI_Analyst.py"), default_timeout=180).run()
    assert not at.exception
    assert any(m.label == "Questions left" and m.value == "10 of 10" for m in at.sidebar.metric)


def test_ai_analyst_session_limit():
    at = AppTest.from_file(str(APP / "pages/4_AI_Analyst.py"), default_timeout=180)
    at.session_state["analyst_asked"] = 10
    at.run()
    assert not at.exception
    assert len(at.chat_input) == 0, "chat box still open after 10 questions"
    if at.info:                                   # no API key in this environment
        return
    assert any("used all 10 questions" in w.value for w in at.warning)


@pytest.mark.parametrize("off_topic,locked", [(1, False), (2, False), (3, True)])
def test_ai_analyst_off_topic_limit(off_topic, locked):
    at = AppTest.from_file(str(APP / "pages/4_AI_Analyst.py"), default_timeout=180)
    at.session_state["analyst_off_topic"] = off_topic
    at.run()
    assert not at.exception
    if at.info:                                   # no API key in this environment
        return
    assert (len(at.chat_input) == 0) == locked
    if locked:
        assert any("chat is closed" in e.value for e in at.error)
    else:
        assert any(f"Off-topic message {off_topic} of 2" in w.value for w in at.warning)
