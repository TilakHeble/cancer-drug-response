"""Independent validation checks on the real data (outputs/tables/validation_checks.csv).

    python scripts/validate.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.linear_model import LogisticRegression, Ridge  # noqa: E402
from sklearn.model_selection import cross_val_score  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from drugresponse import data  # noqa: E402
from drugresponse.config import TABLES  # noqa: E402
from drugresponse.experiment import Store, run_drug  # noqa: E402
from drugresponse.models import (RidgeModel, TissueModel, coexpression_modules,  # noqa: E402
                                 group_lambda_max, group_lasso_path, inner_folds)

DRUGS = ["Nutlin-3a (-)", "Cytarabine"]


def did(S, name):
    ids = S.drugs.index[S.drugs.drug_name == name]
    return max(ids, key=lambda i: S.drugs.loc[i, "n_lines"])


def check_no_leakage(S):
    worst = 0.0
    for name in DRUGS:
        d = did(S, name)
        p1, _ = run_drug(S, d, [TissueModel, RidgeModel], keep_coefs=False)
        y = S.R[d].copy()
        y[(S.fold == 2).values & y.notna().values] = 5.0          # corrupt fold-2 responses
        p2, _ = run_drug(S, d, [TissueModel, RidgeModel], y_override=y, keep_coefs=False)
        a = p1[p1.fold == 2].sort_values(["model", "line"]).pred.values
        b = p2[p2.fold == 2].sort_values(["model", "line"]).pred.values
        worst = max(worst, float(np.abs(a - b).max()))
    return dict(check="test responses never used (corrupted fold-2 AUC)", passed=worst < 1e-10,
                detail=f"{len(DRUGS)} drugs, max change in fold-2 predictions {worst:.1e}")


def check_folds(S):
    f = S.fold
    ok = f.between(0, 4).all() and f.index.is_unique
    share = pd.crosstab(S.M.tissue, f)
    big = share[share.sum(1) >= 25]
    spread = float((big.max(1) - big.min(1)).max())
    return dict(check="outer folds: one per line, tissue-stratified", passed=bool(ok and spread <= 1),
                detail=f"{len(f)} lines; max difference in a tissue's count across folds: {spread:.0f}")


def check_response_values(S):
    raw = data.read_gdsc("gdsc2")
    R = data.load_response("gdsc2")
    rng = np.random.default_rng(0)
    sample = raw.sample(500, random_state=1)
    dup = raw.groupby(["SANGER_MODEL_ID", "DRUG_ID"]).size()
    sample = sample[[dup.get((m, d), 1) == 1 for m, d in zip(sample.SANGER_MODEL_ID, sample.DRUG_ID)]]
    sample = sample[sample.SANGER_MODEL_ID.isin(R.index)]
    got = np.array([R.loc[m, str(d)] for m, d in zip(sample.SANGER_MODEL_ID, sample.DRUG_ID)])
    diff = float(np.nanmax(np.abs(got - sample.AUC.values)))
    _ = rng
    return dict(check="processed AUC equals Sanger release file", passed=diff < 1e-9,
                detail=f"{len(sample)} random (line, drug) pairs, max abs difference {diff:.1e}")


def check_batch_within_fold(S):
    tr = S.fold.values != 0
    te = ~tr
    des, X = S.design(0, tr)
    G = X[:, ~des.is_tissue()]
    P = PCA(20, random_state=0).fit(G[tr]).transform(G[te])
    src = S.sanger[te]
    auc = cross_val_score(LogisticRegression(max_iter=5000), P, src, cv=5, scoring="roc_auc").mean()
    Zr = S.Ev[:, des.genes_]
    Zr = (Zr - Zr[tr].mean(0)) / Zr[tr].std(0)
    Pr = PCA(20, random_state=0).fit(Zr[tr]).transform(Zr[te])
    auc_raw = cross_val_score(LogisticRegression(max_iter=5000), Pr, src, cv=5, scoring="roc_auc").mean()
    return dict(check="batch correction (fitted on folds 1-4) on fold-0 lines", passed=auc < 0.65,
                detail=f"source predictable from 20 PCs: AUC {auc_raw:.3f} raw -> {auc:.3f} corrected")


def check_ridge_vs_sklearn(S):
    d = did(S, "Nutlin-3a (-)")
    y = S.R[d].values
    ok = np.isfinite(y)
    tr = ok & (S.fold.values != 0)
    des, X = S.design(0, S.fold.values != 0)
    m = RidgeModel().fit(X[tr], y[tr], inner_folds(tr.sum(), 5, 1), des)
    sk = Ridge(alpha=m.alpha_).fit(X[tr], y[tr])
    diff = float(np.abs(m.coef_ - sk.coef_).max())
    return dict(check="ridge (SVD path) equals scikit-learn Ridge", passed=diff < 1e-6,
                detail=f"Nutlin-3a fold 0, alpha {m.alpha_:.3g}: max coefficient difference {diff:.1e}")


def check_group_lasso_kkt(S):
    d = did(S, "Nutlin-3a (-)")
    y = S.R[d].values
    ok = np.isfinite(y)
    tr = ok & (S.fold.values != 0)
    des, X = S.design(0, S.fold.values != 0)
    Xt, yt = X[tr], y[tr]
    t = des.is_tissue()
    groups = np.r_[coexpression_modules(Xt[:, ~t], 100), 100 + np.arange(t.sum())]
    Xc, yc = Xt - Xt.mean(0), yt - yt.mean()
    lmax = group_lambda_max(Xc, yc, groups)
    lam = 0.2 * lmax
    b = group_lasso_path(Xc, yc, groups, lmax * np.logspace(0, np.log10(0.2), 10), max_iter=5000, tol=1e-10)[-1]
    grad = Xc.T @ (yc - Xc @ b) / len(yc)
    gap_a, gap_z = 0.0, 0.0
    for g in np.unique(groups):
        i = groups == g
        w = np.sqrt(i.sum())
        nb = np.linalg.norm(b[i])
        if nb > 0:
            gap_a = max(gap_a, np.linalg.norm(grad[i] - lam * w * b[i] / nb) / (lam * w))
        else:
            gap_z = max(gap_z, np.linalg.norm(grad[i]) / (lam * w) - 1)
    return dict(check="group lasso KKT conditions (real data)", passed=gap_a < 1e-3 and gap_z < 1e-3,
                detail=f"Nutlin-3a fold 0, lambda = 0.2 x max, {len(np.unique(groups[np.abs(b) > 0]))} active groups:"
                       f" stationarity gap {gap_a:.1e}, zero-group excess {max(gap_z, 0):.1e}")


def main():
    S = Store()
    res = [check_no_leakage(S), check_folds(S), check_response_values(S), check_batch_within_fold(S),
           check_ridge_vs_sklearn(S), check_group_lasso_kkt(S)]
    out = pd.DataFrame(res)
    TABLES.mkdir(parents=True, exist_ok=True)
    out.to_csv(TABLES / "validation_checks.csv", index=False)
    pd.set_option("display.max_colwidth", 200)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
