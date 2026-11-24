# SecondLook

**An AI-assisted second review for borderline loan declines.**

A Group 3 Capstone Project for Uplift Code Camp, Python for Data and AI (Batch 3).

## Problem

About half of Filipino adults have no formal financial account, and thin credit files get many of them declined with no explanation.

**Business question:** Among applicants a lender would decline, which ones are actually safe to approve, and what realistic changes would get the rest approved?

## What SecondLook does

1. Scores each applicant's default risk with a trained model.
2. Flags borderline declines near the approval cutoff.
3. Explains each score with SHAP.
4. Suggests a realistic path to approval (never age, gender or past defaults).
5. Writes a plain-language decision letter with Claude, then checks it against the model.
6. Answers portfolio questions in plain English through an AI agent.

## Data

[Home Credit Default Risk (Kaggle)](https://www.kaggle.com/competitions/home-credit-default-risk/data): 307,511 labelled applicants (8.07% default rate) plus 6 credit-history tables.

The data is not stored in this repo. To download it:

```bash
pip install kaggle
kaggle auth login            # or put your token in ~/.kaggle/
kaggle competitions download home-credit-default-risk -p data/raw
```

Keep the zip as is. The code reads CSVs straight from it (`src/secondlook/data.py`). If you extracted the CSVs into `data/raw/` instead (for example with kagglehub), that works too.

## Project structure

```
data/raw/          Kaggle zip (not committed)
data/interim/      cleaned tables, SQLite DB (not committed)
data/processed/    model-ready features (not committed)
notebooks/         01 EDA, 02 features, 03 modelling, 04 explainability, 05 letters,
                   06 path to approval, 07 letter audit, 08 fairness audit, 09 personas,
                   10 AI analyst
src/secondlook/    reusable Python package: data.py (load from zip), cleaning.py,
                   sql_features.py (SQLite + SQL aggregations), features.py (pipeline),
                   modeling.py, explain.py (SHAP), path.py (path to approval),
                   letters.py (Claude), letter_audit.py (faithfulness and tone),
                   fairness.py (fairness audit), agent.py + analyst_db.py (AI Analyst),
                   demo_bundle.py
app/               Streamlit app (Home.py + pages/), demo_data/ bundle, requirements.txt
tests/             pytest tests (letter guardrails and audit, formatting, path to approval, fairness,
                   AI Analyst safety and loop, app smoke tests)
models/            trained model files (not committed)
reports/figures/   charts used in the README and pitch
docs/              proposal, feature dictionary, model card
```

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env          # then add your ANTHROPIC_API_KEY
```

Build the model-ready feature table (about 10 minutes the first time; creates a 2.3 GB SQLite database in `data/interim/` and `data/processed/features.parquet`):

```bash
cd src && python -m secondlook.features
```

Or run `notebooks/02_features.ipynb`, which does the same step by step with explanations.

## Run the app

The app runs on a small committed bundle in `app/demo_data/` (model, 61,503 test predictions, 300 applicants, pre-drafted letters and their audits, personas, fairness results and the AI Analyst's 4.6 MB SQLite file), so **it works straight after cloning**, without the Kaggle data:

```bash
pip install -r app/requirements.txt
streamlit run app/Home.py
```

| Page | What it shows |
|---|---|
| **Home** | Problem, headline results, how SecondLook works |
| **Portfolio Dashboard** | Portfolio tab: risk distribution, live cutoff and profit simulator, calibration, approvals vs score-only, thin-file results. Personas tab: the four borrower types among applicants the model turns away |
| **Applicant Review** | One applicant: PD, decision, SHAP reasons, re-scored path to approval (borrow less or a longer term), a short Claude letter that offers those terms (pre-drafted, or drafted live with an API key) with its audit badges (claims supported, tone), officer decision |
| **Model Card** | Model details, metrics, SHAP importance, fairness audit with 95% intervals, the retraining experiment and the decision, intended use and limitations |
| **AI Analyst** | Ask a portfolio question in plain English; Claude writes the SQL, calls SecondLook's tools and shows every query, with its cost. 10 questions per session |

Live letter drafting, the live letter audit and the AI Analyst need `ANTHROPIC_API_KEY` in `.env` (locally) or in the app's secrets (cloud). Without a key, everything else works and pre-drafted letters and audits are shown. Every Claude call uses **Claude Haiku 5.5** (`claude-haiku-5-5`).

To rebuild the bundle after retraining (needs the full pipeline): `cd src && python -m secondlook.demo_bundle`, then re-run notebooks 07 to 09 and `python -m secondlook.analyst_db`.

### Deploy to Streamlit Community Cloud

1. Sign in at [share.streamlit.io](https://share.streamlit.io) with GitHub.
2. **Create app** → repository `Funkchini/secondlook`, branch `main`, main file `app/Home.py`.
3. **Advanced settings** → Python 3.12, and under **Secrets** add `ANTHROPIC_API_KEY = "..."` (optional, for live letters).
4. Deploy. The app installs `app/requirements.txt`.

## Phase 2 results

| Step | Result | Where |
|---|---|---|
| 8. Path to approval | 90.1% of borderline and 31.3% of declined test applicants have a realistic re-scored path | notebook 06, `path.py` |
| Letters (prompt v7) | Short and human: median 98 words (v6: 176), plain words instead of score values; 30 of 30 demo letters pass every check | `letters.py`, notebook 05 |
| 9. Letter audit | 158 of 160 claims in the demo letters supported; all 4 purpose-built bad letters flagged; about $0.0004 per letter on top of $0.0007 to draft it. The audit also found decline letters offering "different terms" with no path, now a guardrail | notebook 07, `letter_audit.py` |
| 10. Fairness audit | Equal-opportunity gaps with 95% bootstrap intervals: women +4.2 pts vs men (3.6 to 4.7), 60s +14.1 vs 20s (13.2 to 15.0), bureau history +5.3 vs thin file (4.6 to 6.1). Retraining without age and family features cut AUC (0.7864 to 0.7850) and **widened** the age gap (to 16.0), so the shipped model stays | notebook 08, `fairness.py` |
| 11. Personas | Four K-Means personas among the 7,263 applicants not approved (established borrower, newer worker, thin file, stretched loan); every one repays about 71-72% of the time | notebook 09 |
| 12. AI Analyst | 10 of 10 test questions correct against pandas-computed answers, about $0.0006 per question; unsafe SQL refused | notebook 10, `agent.py` |

## Tests

```bash
pytest tests/
```

79 tests, no API calls: letter guardrails (24), letter audit with four bad letters (7, Claude answers cached in `tests/fixtures/`), value formatting (17), path to approval (6), fairness helpers (3), AI Analyst SQL safety and agent loop (12) and app smoke tests that run every page, the cutoff slider, the personas tab and the cached letters with their audit badges (10).

## Roadmap

- **Phase 1 (done):** EDA, features, models, SHAP, Claude letters, Streamlit MVP, deployed at [secondlook.streamlit.app](https://secondlook.streamlit.app).
- **Phase 2 (done):** path to approval, faithfulness and tone checks, fairness audit, personas, AI Analyst agent.
- **Step 13, by Dec 1:** re-run notebooks 01 to 10 top to bottom, written summary (`docs/SecondLook - Summary.pdf`), demo rehearsal.

## Team

Group 3. Roles to be filled in.
