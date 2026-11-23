"""Everything the applicant review page needs, kept out of the Streamlit script so it can be tested.

The decision follows notebook 08: each score band has a confidence interval for its default rate
(fit on the "calib" split); the band is approved when the whole interval is below the break-even
default rate, declined when it is above, and referred to an underwriter when it straddles it.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import ID_COL, MODELS_DIR, TARGET_COL
from .conformal import assign_zones, bin_edges, binned_intervals
from .explain import path_to_approval, shap_values, top_reasons
from .features import load_features
from .modeling import ProfitAssumptions, load_bundle, prepare

N_BINS = 20
DECISION_TEXT = {"approve": "Approve", "refer": "Refer to an underwriter", "decline": "Decline"}


@dataclass
class ReviewContext:
    bundle: dict
    applicants: pd.DataFrame       # test-split applicants (never used in training), with scores and decisions
    zones: pd.DataFrame            # score bands: raw-score edges, default-rate interval, decision
    breakeven_pd: float
    approve_cutoff: float          # calibrated probability of default where the refer zone starts
    decline_cutoff: float          # calibrated probability of default where the decline zone starts


def load_context() -> ReviewContext:
    """Load the model and score the held-out applicants (a few seconds; cache the result)."""
    bundle = load_bundle("secondlook_lgbm")
    meta = json.loads((MODELS_DIR / "secondlook_lgbm_meta.json").read_text())
    econ = ProfitAssumptions(**meta["profit_assumptions"])
    df = prepare(load_features())
    df = df[df["split"].isin(["valid", "calib", "test"])].copy()
    df["raw"] = bundle["model"].predict_proba(df[bundle["columns"]])[:, 1]
    df["p"] = bundle["calibrator"].predict(df["raw"])

    edges = bin_edges(df.loc[df["split"] == "valid", "raw"], N_BINS)
    calib = df[df["split"] == "calib"]
    zones = assign_zones(binned_intervals(calib[TARGET_COL], calib["raw"], edges), econ.breakeven_pd)

    test = df[df["split"] == "test"].copy()
    test["band"] = np.clip(np.searchsorted(edges, test["raw"], side="right") - 1, 0, N_BINS - 1)
    test["decision"] = zones["zone"].to_numpy()[test["band"]]

    def calibrated_start(zone: str) -> float:
        edge = zones.loc[zones["zone"] == zone, "lower_edge"].min()
        return float(bundle["calibrator"].predict([edge])[0]) if np.isfinite(edge) else 1.0

    return ReviewContext(bundle, test.set_index(ID_COL, drop=False), zones, econ.breakeven_pd,
                         calibrated_start("refer"), calibrated_start("decline"))


def review(ctx: ReviewContext, applicant_id: int, n: int = 5, with_path: bool = True) -> dict:
    """Decision, interval, reasons and (if not approved) a path to automatic approval for one applicant."""
    row = ctx.applicants.loc[[applicant_id]]
    x = row.iloc[0]
    contrib, base = shap_values(ctx.bundle, row)
    band = ctx.zones.loc[int(x["band"])]
    out = {
        "applicant_id": int(applicant_id),
        "p_default": float(x["p"]),
        "decision": x["decision"],
        "band_low": float(band["low"]), "band_high": float(band["high"]),
        "band_rate": float(band["default_rate"]),
        "thin_file": bool(x["THIN_FILE"]),
        "credit": float(x["AMT_CREDIT"]),
        "outcome": int(x[TARGET_COL]),
        "reasons": top_reasons(contrib.iloc[0], x, n=n, direction="risk"),
        "strengths": top_reasons(contrib.iloc[0], x, n=n, direction="strength"),
        "contributions": contrib.iloc[0],
        "options": None,
        "path_target": None,
    }
    if with_path and x["decision"] != "approve":
        # Calibrated scores move in steps and a step can straddle two zones, so a change must reach a
        # strictly lower step (margin ~0) to land in the better zone. Try automatic approval first; for a
        # declined applicant with no such path, show what would get them to an underwriter instead.
        targets = [("approve", ctx.approve_cutoff)]
        if x["decision"] == "decline":
            targets.append(("refer", ctx.decline_cutoff))
        for target, cutoff in targets:
            options = path_to_approval(ctx.bundle, row, cutoff=cutoff, margin=1e-6)
            if len(options):
                out["options"], out["path_target"] = options, target
                break
    return out
