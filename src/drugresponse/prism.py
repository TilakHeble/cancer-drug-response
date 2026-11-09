"""Independent-lab test: Broad Institute PRISM Repurposing secondary screen (Corsello et al. 2020).

PRISM measures viability of pooled, DNA-barcoded cell lines: a different lab, assay, plate
format and concentration range from GDSC. The GDSC2 models are *not* retrained: each cell
line already has an out-of-fold prediction from a model that never saw it, and those
predictions are scored against PRISM's AUC for the same drug and line. Agreement between the
GDSC2 and PRISM measurements themselves is the ceiling.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .config import PROCESSED, RAW
from .data import SANGER, _download

PRISM_URL = "https://ndownloader.figshare.com/files/20237739"   # secondary-screen-dose-response-curve-parameters.csv
PRISM_FILE = RAW / "prism" / "secondary-screen-dose-response-curve-parameters.csv"


def _key(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def download():
    return _download(PRISM_URL, PRISM_FILE)


def build() -> pd.DataFrame | None:
    """Long table: drug (GDSC2 id), line (Sanger id), prism_auc. None if PRISM is unavailable."""
    if not PRISM_FILE.exists():
        return None
    p = pd.read_csv(PRISM_FILE, usecols=["depmap_id", "name", "auc", "r2", "screen_id",
                                         "passed_str_profiling"])
    p = p[p.passed_str_profiling.astype(str).str.upper() == "TRUE"]   # cell identity verified
    p = p[p.auc.notna() & (p.r2.fillna(0) >= 0)]              # drop curves that fit worse than flat
    models = pd.read_csv(SANGER / "model_list_latest.csv.gz", usecols=["model_id", "BROAD_ID"])
    broad = models.dropna().set_index("BROAD_ID").model_id
    p["line"] = p.depmap_id.map(broad)
    drugs = pd.read_csv(PROCESSED / "drugs.csv", dtype={"DRUG_ID": str}, index_col=0)
    by_name = {}
    for d, n in drugs.sort_values("n_lines", ascending=False).drug_name.items():
        by_name.setdefault(_key(n), d)
    p["drug"] = p.name.map(lambda n: by_name.get(_key(n)))
    p = p.dropna(subset=["line", "drug"])
    out = p.groupby(["drug", "line"]).auc.median().rename("prism_auc").reset_index()
    out.to_parquet(PROCESSED / "prism_auc.parquet")
    return out


def score(preds: pd.DataFrame, prism: pd.DataFrame, min_lines: int = 50) -> pd.DataFrame:
    """Per drug x model: r(prediction, PRISM), r(GDSC2 measured, PRISM), r(prediction, GDSC2)."""
    m = preds.merge(prism, on=["drug", "line"])
    rows = []
    for (d, mod), g in m.groupby(["drug", "model"]):
        if len(g) < min_lines or g.pred.std() == 0:
            continue
        rows.append(dict(drug=d, model=mod, n=len(g),
                         r_prism=float(np.corrcoef(g.pred, g.prism_auc)[0, 1]),
                         r_gdsc2=float(np.corrcoef(g.pred, g.actual)[0, 1]),
                         r_assays=float(np.corrcoef(g.actual, g.prism_auc)[0, 1])))
    return pd.DataFrame(rows)
