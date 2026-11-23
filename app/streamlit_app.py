"""SecondLook Streamlit app: entry point.

Run from the project root:  streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # import secondlook from src/

import streamlit as st

st.set_page_config(page_title="SecondLook", page_icon=":material/account_balance:", layout="wide")

page = st.navigation(
    [
        st.Page(
            "app_pages/home.py", title="Overview", icon=":material/home:", default=True,
            help="The problem SecondLook solves and the portfolio at a glance.",
        ),
        st.Page(
            "app_pages/applicant_review.py", title="Applicant review", icon=":material/person_search:",
            help=("Pick a held-out applicant to see their decision (approve, refer or decline), the reasons, "
                  "a path to approval and a decision letter."),
        ),
    ],
    position="top",
)
page.run()
