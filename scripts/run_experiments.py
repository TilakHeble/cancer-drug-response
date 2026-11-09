"""Run the experiments. Each experiment can be split across processes with --part i --of n.

    python scripts/run_experiments.py main  --part 0 --of 2   # all eligible drugs, main models
    python scripts/run_experiments.py panel --part 0 --of 2   # focus panel: slower models, robustness
    python scripts/run_experiments.py merge                   # combine parts

Outputs (outputs/):
    preds_<exp>.parquet    out-of-fold predictions: drug, line, fold, model, pred, actual, design
    coefs_<exp>.pkl        per drug x fold x model: selected / largest features
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from drugresponse import data  # noqa: E402
from drugresponse.config import OUTPUTS  # noqa: E402
from drugresponse.experiment import Store, run_drug  # noqa: E402
from drugresponse.models import (MAIN_MODELS, PANEL_MODELS, LassoModel, RidgeModel,  # noqa: E402
                                 TissueModel)
from drugresponse.panel import focus_panel, holdout_tissues  # noqa: E402


def _save(name, part, preds, coefs):
    OUTPUTS.mkdir(exist_ok=True)
    pd.concat(preds, ignore_index=True).to_parquet(OUTPUTS / f"preds_{name}.part{part}.parquet")
    with open(OUTPUTS / f"coefs_{name}.part{part}.pkl", "wb") as f:
        pickle.dump(coefs, f)


def run_main(S, drugs, part):
    preds, coefs = [], []
    t0 = time.time()
    for i, d in enumerate(drugs):
        p, c = run_drug(S, d, MAIN_MODELS)
        p["design"] = "main"
        preds.append(p)
        coefs += c
        if (i + 1) % 10 == 0:
            print(f"main part{part}: {i + 1}/{len(drugs)} drugs, {time.time() - t0:.0f}s", flush=True)
            _save("main", part, preds, coefs)
    _save("main", part, preds, coefs)


def run_panel(S, drugs, part):
    """Slower models and robustness designs on the focus panel."""
    g1 = data.load_response("gdsc1").reindex(S.E.index)
    ic50 = data.load_response("gdsc2", "LN_IC50").reindex(S.E.index)
    cross = focus_panel(S)["gdsc1_id"].to_dict()
    tissues = holdout_tissues(S)
    preds, coefs = [], []
    # resume: keep a previous partial run of this part and skip its drugs
    prev = OUTPUTS / f"preds_panel.part{part}.parquet"
    if prev.exists():
        old = pd.read_parquet(prev)
        with open(OUTPUTS / f"coefs_panel.part{part}.pkl", "rb") as f:
            coefs = pickle.load(f)
        done = set(old.drug)
        preds = [old]
        coefs = [c for c in coefs if c["drug"] in done]
        drugs = [d for d in drugs if d not in done]
        print(f"panel part{part}: resuming, {len(done)} drugs already done", flush=True)
    t0 = time.time()
    for i, d in enumerate(drugs):
        # 1. adaptive lasso and group lasso
        p, c = run_drug(S, d, PANEL_MODELS)
        p["design"] = "main"
        preds.append(p)
        coefs += c
        # 2. gene-filter sensitivity
        for ng in (1000, 5000):
            p, _ = run_drug(S, d, [RidgeModel, LassoModel], n_genes=ng, keep_coefs=False)
            p["design"] = f"genes_{ng}"
            preds.append(p)
        # 3. ln(IC50) as the response
        p, _ = run_drug(S, d, [TissueModel, RidgeModel, LassoModel], y_override=ic50[d], keep_coefs=False)
        p["design"] = "target_ln_ic50"
        preds.append(p)
        # 4. leave one cancer type out: train without tissue t, predict tissue t
        for t in tissues:
            mask = S.tissue != t
            p, _ = run_drug(S, d, [RidgeModel, LassoModel], train_mask=mask, keep_coefs=False)
            p = p[S.M.tissue.reindex(p.line).values == t]
            p["design"] = f"leave_out:{t}"
            preds.append(p)
        # 5. cross-assay: GDSC2-trained predictions scored against GDSC1 for the same drug
        if isinstance(cross.get(d), str):
            p, _ = run_drug(S, d, [TissueModel, RidgeModel, LassoModel], keep_coefs=False)
            p["actual_gdsc1"] = g1[cross[d]].reindex(p.line).values
            p["design"] = "cross_assay"
            preds.append(p)
        print(f"panel part{part}: {i + 1}/{len(drugs)} drugs, {time.time() - t0:.0f}s", flush=True)
        _save("panel", part, preds, coefs)


def merge(name):
    parts = sorted(OUTPUTS.glob(f"preds_{name}.part*.parquet"))
    pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True).to_parquet(
        OUTPUTS / f"preds_{name}.parquet")
    coefs = []
    for p in sorted(OUTPUTS.glob(f"coefs_{name}.part*.pkl")):
        with open(p, "rb") as f:
            coefs += pickle.load(f)
    with open(OUTPUTS / f"coefs_{name}.pkl", "wb") as f:
        pickle.dump(coefs, f)
    for p in parts + sorted(OUTPUTS.glob(f"coefs_{name}.part*.pkl")):
        p.unlink()
    print(f"merged {name}: {len(parts)} parts")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment", choices=["main", "panel", "merge"])
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--drugs", nargs="*", help="explicit drug IDs for this part (overrides --of)")
    a = ap.parse_args()
    if a.experiment == "merge":
        for n in ("main", "panel"):
            if list(OUTPUTS.glob(f"preds_{n}.part*.parquet")):
                merge(n)
        sys.exit()
    S = Store()
    drugs = S.eligible_drugs() if a.experiment == "main" else list(focus_panel(S).index)
    drugs = a.drugs if a.drugs else drugs[a.part::a.of]
    (run_main if a.experiment == "main" else run_panel)(S, drugs, a.part)
