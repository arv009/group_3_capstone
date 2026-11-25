"""Build the small, committed data bundle the Streamlit app runs on (app/demo_data/).

The full dataset, database and feature table are too large for GitHub, so the deployed app
uses a compact bundle:
  - model.txt            the trained LightGBM model in LightGBM's own text format
                          (version-robust, no pickle)
  - model_meta.json      features, categories, cutoff, metrics, assumptions
  - portfolio.parquet    test-set predictions for the dashboard (61,503 applicants, few columns)
  - applicants.parquet   full model features for 300 test applicants the officer can open
  - letters.json         Claude letters already drafted for the demo applicants

Run from the project root after notebooks 02-05:   cd src && python -m secondlook.demo_bundle
"""
from __future__ import annotations

import json
import shutil

import numpy as np
import pandas as pd

from . import modeling as M
from .config import ID_COL, PROCESSED_DIR, ROOT, TARGET_COL
from .features import load_features
from .letters import CACHE_PATH, MODEL as LETTER_MODEL, PROMPT_VERSION

BUNDLE_DIR = ROOT / "app" / "demo_data"
N_EXTRA = 270            # applicants beyond the 30 with ready-made letters
SEED = 7


def build(verbose: bool = True) -> dict:
    BUNDLE_DIR.mkdir(parents=True, exist_ok=True)
    model, meta = M.load_model()

    # 1. Model in LightGBM's native text format + metadata
    model.booster_.save_model(str(BUNDLE_DIR / "model.txt"))
    shutil.copy(M.META_PATH, BUNDLE_DIR / "model_meta.json")

    # 2. Portfolio predictions (test set) + the external score, for the "score-only" comparison
    _, test = M.split(load_features())
    preds = pd.read_parquet(PROCESSED_DIR / "test_predictions.parquet")
    assert (preds[ID_COL].values == test[ID_COL].values).all()
    preds["EXT_SOURCE_MEAN"] = test["EXT_SOURCE_MEAN"].values
    preds.to_parquet(BUNDLE_DIR / "portfolio.parquet", index=False)

    # 3. Applicants the officer can open: 30 demo applicants + a random sample of other test applicants
    demo_ids = json.loads((PROCESSED_DIR / "demo_ids.json").read_text())
    rest = test[~test[ID_COL].isin(demo_ids)].sample(N_EXTRA, random_state=SEED)[ID_COL].tolist()
    keep_ids = demo_ids + rest
    cols = [ID_COL, TARGET_COL] + meta["features"]
    applicants = test.set_index(ID_COL).loc[keep_ids].reset_index()[cols]
    applicants.to_parquet(BUNDLE_DIR / "applicants.parquet", index=False)

    # 4. Letters already drafted for these applicants (current prompt version only)
    letters = write_letters(keep_ids)

    # 5. Check: the text-format model gives the same predictions as the trained model
    import lightgbm as lgb
    booster = lgb.Booster(model_file=str(BUNDLE_DIR / "model.txt"))
    X = M.prepare_X(applicants, meta["features"], meta["categorical"], meta["categories"])
    max_diff = float(np.abs(booster.predict(X) - model.predict_proba(X)[:, 1]).max())

    report = {
        "files": {p.name: round(p.stat().st_size / 1e6, 2) for p in sorted(BUNDLE_DIR.iterdir())},
        "portfolio_rows": len(preds), "applicants": len(applicants), "demo_with_letters": len(demo_ids),
        "letters": len(letters), "max_prediction_difference": max_diff,
    }
    if verbose:
        print(json.dumps(report, indent=2))
    return report


def write_letters(keep_ids=None) -> dict:
    """Copy the cached letters for the bundle's applicants (current prompt version and model) into letters.json."""
    if keep_ids is None:
        keep_ids = pd.read_parquet(BUNDLE_DIR / "applicants.parquet", columns=[ID_COL])[ID_COL].tolist()
    cache = json.loads(CACHE_PATH.read_text(encoding="utf-8")) if CACHE_PATH.exists() else {}
    letters = {k: v for k, v in cache.items()
               if v["applicant_id"] in set(keep_ids) and v.get("prompt_version") == PROMPT_VERSION and v.get("model") == LETTER_MODEL}
    (BUNDLE_DIR / "letters.json").write_text(json.dumps(letters, indent=2, ensure_ascii=False), encoding="utf-8")
    return letters


def draft_demo_letters(verbose: bool = True) -> dict:
    """Draft (or reuse from cache) the demo applicants' letters with the current prompt, exactly as the app would:
    bundle model, SHAP reasons and path to approval. Then refresh letters.json. Calls the Claude API for new ones."""
    import lightgbm as lgb
    from .explain import Explainer
    from .letters import build_letter_input, generate_letter
    from .path import path_to_approval

    booster = lgb.Booster(model_str=(BUNDLE_DIR / "model.txt").read_bytes().decode("utf-8").replace("\r\n", "\n"))
    meta = json.loads((BUNDLE_DIR / "model_meta.json").read_text())
    apps = pd.read_parquet(BUNDLE_DIR / "applicants.parquet")
    explainer = Explainer(booster)
    results = []
    for sid in json.loads((PROCESSED_DIR / "demo_ids.json").read_text()):
        row = apps[apps[ID_COL] == sid]
        exp = explainer.explain(M.prepare_X(row, meta["features"], meta["categorical"], meta["categories"]))
        path = path_to_approval(row, booster, meta, meta["cutoff"])
        li = build_letter_input(sid, exp, row.iloc[0], meta["cutoff"], path=path)
        results.append(generate_letter(li))
    letters = write_letters()
    report = {"letters": len(results), "passed": sum(r["checks_passed"] for r in results),
              "new": sum(not r["from_cache"] for r in results),
              "with_adjusted_terms": sum(bool(r["fact_sheet"].get("adjusted_terms")) for r in results),
              "cost_usd_new": round(sum(r["cost_usd"] for r in results if not r["from_cache"]), 4),
              "in_bundle": len(letters)}
    if verbose:
        print(json.dumps(report, indent=2))
    return report


def audit_demo_letters(verbose: bool = True) -> dict:
    """Audit every letter in letters.json (faithfulness + tone, cached) and save the results to audits.json."""
    from .letter_audit import audit_letter
    letters = json.loads((BUNDLE_DIR / "letters.json").read_text(encoding="utf-8"))
    audits = {}
    for key, letter in letters.items():
        a = audit_letter(letter)
        audits[key] = {k: v for k, v in a.items() if k != "raw"}      # keep the bundle small: no raw Claude output
    (BUNDLE_DIR / "audits.json").write_text(json.dumps(audits, indent=2, ensure_ascii=False), encoding="utf-8")
    report = {"audited": len(audits), "passed": sum(a["passed"] for a in audits.values()),
              "claims": sum(a["n_claims"] for a in audits.values()),
              "supported": sum(a["n_supported"] for a in audits.values())}
    if verbose:
        print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    build()
