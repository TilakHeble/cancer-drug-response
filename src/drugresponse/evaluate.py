"""Evaluation of out-of-fold predictions.

Why several metrics: much of the variation in drug response is explained by cancer type
(blood-cancer lines are sensitive to many drugs). A model can score a high correlation by
learning tissue alone, so every model is compared with the Tissue baseline, and accuracy is
also measured *within* cancer types.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
from scipy import stats

MIN_TISSUE_N = 15


def _r(a, b):
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def _rho(a, b):
    if len(a) < 3:
        return np.nan
    return float(stats.spearmanr(a, b).statistic)


def within_tissue_rho(g: pd.DataFrame, tissue: pd.Series) -> float:
    """Line-count-weighted mean Spearman correlation within tissues with >= MIN_TISSUE_N lines."""
    t = tissue.reindex(g.line).values
    vals, w = [], []
    for tt in np.unique(t):
        m = t == tt
        if m.sum() >= MIN_TISSUE_N and np.std(g.pred.values[m]) > 0:
            vals.append(_rho(g.pred.values[m], g.actual.values[m]))
            w.append(m.sum())
    return float(np.average(vals, weights=w)) if vals else np.nan


def drug_metrics(preds: pd.DataFrame, tissue: pd.Series) -> pd.DataFrame:
    rows = []
    for (d, m), g in preds.groupby(["drug", "model"], sort=False):
        sst = np.sum((g.actual - g.actual.mean()) ** 2)
        rows.append(dict(drug=d, model=m, n=len(g), r=_r(g.pred, g.actual), rho=_rho(g.pred, g.actual),
                         rmse=float(np.sqrt(np.mean((g.pred - g.actual) ** 2))),
                         r2=float(1 - np.sum((g.pred - g.actual) ** 2) / sst),
                         # Mean and Tissue predictions are constant within a tissue (up to
                         # fold-to-fold noise), so within-tissue ranking is undefined for them
                         rho_within=(np.nan if m in ("Mean", "Tissue")
                                     else within_tissue_rho(g, tissue))))
    return pd.DataFrame(rows)


def summary(met: pd.DataFrame, baseline="Tissue") -> pd.DataFrame:
    """Median metrics across drugs, and the paired comparison with the baseline."""
    wide = met.pivot(index="drug", columns="model", values="r")
    wr = met.pivot(index="drug", columns="model", values="rho_within")
    r2 = met.pivot(index="drug", columns="model", values="r2")
    out = []
    for m in wide.columns:
        d = (wide[m] - wide[baseline]).dropna()
        p = stats.wilcoxon(d).pvalue if m != baseline and len(d) > 10 and d.abs().sum() > 0 else np.nan
        out.append(dict(model=m, n_drugs=int(wide[m].notna().sum()),
                        median_r=wide[m].median(), median_r2=r2[m].median(),
                        median_rho_within=wr[m].median(),
                        share_r_above_0_3=(wide[m] > 0.3).mean(),
                        median_gain_vs_tissue=d.median() if m != baseline else np.nan,
                        share_beats_tissue=(d > 0).mean() if m != baseline else np.nan,
                        wilcoxon_p_vs_tissue=p))
    return pd.DataFrame(out).set_index("model")


def pairwise_wins(met: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    """Share of drugs on which the row model has a higher r than the column model."""
    w = met.pivot(index="drug", columns="model", values="r")[models]
    return pd.DataFrame({b: {a: float((w[a] > w[b]).mean()) for a in models} for b in models})


# --------------------------------------------------------------------------- selection
def selection_stability(coefs: list[dict], model: str) -> pd.DataFrame:
    """Per drug: features selected in every fold, in >= 3 of 5, and mean pairwise Jaccard."""
    by = {}
    for c in coefs:
        if c["model"] == model:
            by.setdefault(c["drug"], []).append(set(f for f in c["features"] if not f.startswith("tissue:")))
    rows = []
    for d, sets in by.items():
        cnt = Counter(g for s in sets for g in s)
        jac = [len(a & b) / max(1, len(a | b)) for i, a in enumerate(sets) for b in sets[i + 1:]]
        rows.append(dict(drug=d, model=model, median_selected=np.median([len(s) for s in sets]),
                         n_all_folds=sum(v == len(sets) for v in cnt.values()),
                         n_3plus=sum(v >= 3 for v in cnt.values()),
                         mean_jaccard=np.mean(jac) if jac else np.nan))
    return pd.DataFrame(rows)


def gene_frequency(coefs: list[dict], model: str, drug: str) -> pd.DataFrame:
    """Selection frequency and mean coefficient of each gene for one drug and model."""
    rows = [(f, v) for c in coefs if c["model"] == model and c["drug"] == drug
            for f, v in zip(c["features"], c["values"])]
    n_folds = len({c["fold"] for c in coefs if c["model"] == model and c["drug"] == drug})
    if not rows:
        return pd.DataFrame(columns=["feature", "folds", "mean_coef"])
    df = pd.DataFrame(rows, columns=["feature", "coef"])
    g = df.groupby("feature").coef.agg(folds="size", mean_coef="mean").reset_index()
    g["share"] = g.folds / n_folds
    return g.sort_values(["folds", "mean_coef"], key=lambda s: -s.abs() if s.name == "mean_coef" else -s)
