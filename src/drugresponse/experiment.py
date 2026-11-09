"""Nested cross-validation for every drug.

Outer loop: each cell line is assigned once to one of N_OUTER folds, stratified by tissue
(round-robin within each tissue after a seeded shuffle). The same assignment is used for
every drug and model, so differences between models are paired.

Inner loop: penalties tuned by K-fold CV on the outer-training lines only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import data
from .config import MIN_LINES_PER_DRUG, N_INNER, N_OUTER, N_TOP_VARIABLE, SEED
from .models import FAST_MODELS, Design, inner_folds


def outer_folds(models: pd.DataFrame, k: int = N_OUTER, seed: int = SEED) -> pd.Series:
    rng = np.random.default_rng(seed)
    fold = pd.Series(-1, index=models.index, dtype=int)
    offset = 0
    for _, idx in models.groupby("tissue").groups.items():
        idx = np.array(idx)
        rng.shuffle(idx)
        fold.loc[idx] = (np.arange(len(idx)) + offset) % k
        offset += len(idx)          # rotate the start so small tissues do not all land in fold 0
    return fold


class Store:
    """Everything an experiment needs, loaded once."""

    def __init__(self):
        self.E = data.load_expression()
        self.M = data.load_models().reindex(self.E.index)
        self.R = data.load_response("gdsc2").reindex(self.E.index)
        self.drugs = data.load_drugs("gdsc2")
        self.fold = outer_folds(self.M)
        self.genes = np.array(self.E.columns)
        self.Ev = self.E.values.astype(np.float64)
        self.tissue = self.M.tissue.values
        self.sanger = (self.M.expression_source == "Sanger").values

    def design(self, k: int, train: np.ndarray, n_genes: int = N_TOP_VARIABLE):
        """Fold-k design fitted on the given training lines (cached for the default case:
        all expression-profiled lines outside fold k). Uses no drug response."""
        key = (k, n_genes, hash(train.tobytes()))
        cache = self.__dict__.setdefault("_designs", {})
        if key not in cache:
            des = Design(n_genes).fit(self.Ev[train], self.tissue[train], self.sanger[train])
            X = des.transform(self.Ev, self.tissue, self.sanger)    # every line, fold-k scaling
            if len(cache) > 12:
                cache.clear()
            cache[key] = (des, X)
        return cache[key]

    def eligible_drugs(self):
        n = self.R.notna().sum()
        return list(n[n >= MIN_LINES_PER_DRUG].index)


def run_drug(S: Store, drug: str, model_classes=None, y_override: pd.Series | None = None,
             train_mask: np.ndarray | None = None, n_genes: int = N_TOP_VARIABLE,
             keep_coefs: bool = True):
    """Outer CV for one drug. Returns (predictions DataFrame, coefficient records)."""
    y_all = (y_override if y_override is not None else S.R[drug]).values
    ok = np.isfinite(y_all)
    model_classes = model_classes or FAST_MODELS
    preds, coefs = [], []
    for k in range(N_OUTER):
        te = ok & (S.fold.values == k)
        tr = ok & (S.fold.values != k)
        base = S.fold.values != k
        if train_mask is not None:
            tr &= train_mask
            base &= train_mask
        if te.sum() == 0 or tr.sum() < 30:
            continue
        # The design (batch correction, gene filter, scaling) uses every profiled line outside
        # fold k, whether or not this drug was screened on it: it never sees a response value.
        des, X = S.design(k, base, n_genes)
        Xtr, Xte = X[tr], X[te]
        ytr = y_all[tr]
        folds = inner_folds(len(ytr), N_INNER, SEED + k)
        names = np.r_[S.genes[des.genes_], ["tissue:" + t for t in des.tissues_]]
        for cls in model_classes:
            m = cls().fit(Xtr, ytr, folds, des)
            p = m.predict(Xte)
            preds.append(pd.DataFrame({"drug": drug, "line": S.E.index[te], "fold": k,
                                       "model": m.name, "pred": p, "actual": y_all[te]}))
            if keep_coefs and m.name not in ("Mean",):
                nz = np.flatnonzero(np.abs(m.coef_) > 1e-10)
                if not m.sparse:      # dense models: keep the 50 largest standardised coefficients
                    nz = nz[np.argsort(-np.abs(m.coef_[nz]))][:50]
                coefs.append(dict(drug=drug, fold=k, model=m.name, n_nonzero=int(np.sum(np.abs(m.coef_) > 1e-10)),
                                  alpha=getattr(m, "alpha_", np.nan),
                                  l1_ratio=getattr(m, "l1_ratio_", np.nan), k=getattr(m, "k_", np.nan),
                                  features=list(names[nz]), values=list(np.round(m.coef_[nz], 5))))
    return pd.concat(preds, ignore_index=True), coefs
