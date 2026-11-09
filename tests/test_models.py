import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso

from drugresponse.evaluate import selection_stability, within_tissue_rho
from drugresponse.experiment import outer_folds, run_drug
from drugresponse.models import (Design, GroupLassoModel, MeanModel, RidgeModel, TissueModel,
                                 _ridge_path, _thin_svd, group_lasso_path)

rng = np.random.default_rng(0)


def test_thin_svd_and_ridge_path_match_direct_solution():
    X = rng.normal(size=(40, 120))
    X -= X.mean(0)
    y = rng.normal(size=40)
    y -= y.mean()
    U, s, Vt = _thin_svd(X)
    assert np.allclose((U * s) @ Vt, X)
    for a in (0.5, 10.0):
        b = _ridge_path(X, y, np.array([a]))[0]
        direct = np.linalg.solve(X.T @ X + a * np.eye(120), X.T @ y)
        assert np.allclose(b, direct)


def test_batch_correction_removes_source_shift_but_keeps_tissue_effect():
    n, p = 300, 50
    tissue = rng.choice(["A", "B", "C"], n)
    sanger = rng.random(n) < 0.5
    E = rng.normal(size=(n, p))
    E[:, 0] += 3.0 * (tissue == "A")           # a real tissue effect
    E[:, 1] += 2.0 * sanger                     # a pure batch effect
    E[:, 2] += 2.0 * sanger + 3.0 * (tissue == "B")
    d = Design(p).fit(E, tissue, sanger)
    assert abs(d.batch_[1] - 2.0) < 0.3 and abs(d.batch_[2] - 2.0) < 0.3
    assert abs(d.batch_[0]) < 0.3                # tissue effect not mistaken for batch
    Ec = E - np.outer(sanger, d.batch_)
    assert abs(Ec[sanger, 1].mean() - Ec[~sanger, 1].mean()) < 0.3
    assert Ec[tissue == "A", 0].mean() - Ec[tissue != "A", 0].mean() > 2.5


def test_batch_correction_not_fooled_by_tissue_confounded_with_source():
    # source and tissue strongly associated: the tissue signal must survive
    n = 400
    tissue = np.where(rng.random(n) < 0.5, "A", "B")
    sanger = np.where(tissue == "A", rng.random(n) < 0.8, rng.random(n) < 0.2)
    E = rng.normal(size=(n, 3))
    E[:, 0] += 4.0 * (tissue == "A")
    d = Design(3).fit(E, tissue, sanger)
    assert abs(d.batch_[0]) < 0.4


def test_design_parameters_come_from_training_lines_only():
    E = rng.normal(size=(100, 30))
    t = rng.choice(["A", "B"], 100)
    s = rng.random(100) < 0.5
    d1 = Design(10).fit(E[:70], t[:70], s[:70])
    E2 = E.copy()
    E2[70:] += 100.0                             # change only the "test" lines
    d2 = Design(10).fit(E2[:70], t[:70], s[:70])
    assert np.array_equal(d1.genes_, d2.genes_) and np.allclose(d1.mu_, d2.mu_)


def test_group_lasso_singletons_equal_lasso():
    X = rng.normal(size=(80, 10))
    X = (X - X.mean(0)) / X.std(0)
    y = X[:, 0] * 2 - X[:, 3] + rng.normal(size=80)
    y -= y.mean()
    b = group_lasso_path(X, y, np.arange(10), [0.1], max_iter=5000, tol=1e-10)[-1]
    ref = Lasso(alpha=0.1, fit_intercept=False, tol=1e-10, max_iter=100000).fit(X, y).coef_
    assert np.allclose(b, ref, atol=1e-4)


def test_group_lasso_zeroes_whole_groups():
    X = rng.normal(size=(100, 12))
    y = X[:, :3] @ np.array([1.0, -1.0, 0.5]) + 0.1 * rng.normal(size=100)
    X -= X.mean(0)
    y -= y.mean()
    groups = np.repeat(np.arange(4), 3)
    b = group_lasso_path(X, y, groups, [0.15], max_iter=3000, tol=1e-9)[-1]
    norms = [np.linalg.norm(b[groups == g]) for g in range(4)]
    assert norms[0] > 0.1
    assert all(nm == 0 or nm > 1e-6 for nm in norms)
    assert sum(nm == 0 for nm in norms[1:]) >= 2


def test_outer_folds_are_stratified_and_deterministic():
    m = pd.DataFrame({"tissue": ["A"] * 50 + ["B"] * 20 + ["C"] * 3},
                     index=[f"L{i}" for i in range(73)])
    f1, f2 = outer_folds(m, 5, 1), outer_folds(m, 5, 1)
    assert f1.equals(f2) and set(f1) == set(range(5))
    assert (f1[m.tissue == "A"].value_counts() == 10).all()
    assert (f1[m.tissue == "B"].value_counts() == 4).all()


class FakeStore:
    """Minimal stand-in for experiment.Store with a planted gene signal."""

    def __init__(self, n=200, p=60):
        from drugresponse.experiment import Store
        self.E = pd.DataFrame(rng.normal(size=(n, p)), index=[f"L{i}" for i in range(n)],
                              columns=[f"G{j}" for j in range(p)])
        self.M = pd.DataFrame({"tissue": rng.choice(["A", "B", "C"], n),
                               "expression_source": np.where(rng.random(n) < 0.5, "Sanger", "Broad")},
                              index=self.E.index)
        y = 0.5 + 0.1 * self.E["G0"] + 0.02 * rng.normal(size=n)
        self.R = pd.DataFrame({"d1": y})
        self.fold = outer_folds(self.M, 5, 0)
        self.genes = np.array(self.E.columns)
        self.Ev = self.E.values
        self.tissue = self.M.tissue.values
        self.sanger = (self.M.expression_source == "Sanger").values
        self.design = Store.design.__get__(self)


def test_run_drug_predictions_do_not_depend_on_test_responses():
    S = FakeStore()
    p1, _ = run_drug(S, "d1", [MeanModel, TissueModel, RidgeModel], n_genes=40)
    S.R = S.R.copy()
    S.R.loc[S.fold.values == 0, "d1"] = 99.0      # corrupt fold-0 responses
    p2, _ = run_drug(S, "d1", [MeanModel, TissueModel, RidgeModel], n_genes=40)
    a = p1[p1.fold == 0].sort_values(["model", "line"]).pred.values
    b = p2[p2.fold == 0].sort_values(["model", "line"]).pred.values
    assert np.allclose(a, b)                      # fold-0 predictions unchanged
    r = np.corrcoef(p1[p1.model == "Ridge"].pred, p1[p1.model == "Ridge"].actual)[0, 1]
    assert r > 0.8                                # and the planted signal is recovered


def test_within_tissue_rho_ignores_between_tissue_signal():
    lines = [f"L{i}" for i in range(60)]
    tissue = pd.Series(["A"] * 30 + ["B"] * 30, index=lines)
    actual = np.r_[np.zeros(30), np.ones(30)] + rng.normal(scale=0.01, size=60)
    pred = np.r_[np.zeros(30), np.ones(30)] + rng.normal(scale=0.01, size=60)   # tissue only
    g = pd.DataFrame({"line": lines, "pred": pred, "actual": actual})
    assert np.corrcoef(pred, actual)[0, 1] > 0.99
    assert abs(within_tissue_rho(g, tissue)) < 0.5


def test_selection_stability_counts():
    coefs = [dict(drug="d", model="Lasso", fold=k, features=["A", "B"] if k < 3 else ["A"], values=[1, 1])
             for k in range(5)]
    s = selection_stability(coefs, "Lasso").iloc[0]
    assert s.n_all_folds == 1 and s.n_3plus == 2


def test_prism_score_uses_matching_lines_and_reports_ceiling():
    from drugresponse.prism import _key, score
    lines = [f"L{i}" for i in range(80)]
    truth = rng.normal(size=80)
    preds = pd.DataFrame({"drug": "d1", "line": lines, "model": "Ridge",
                          "pred": truth + rng.normal(scale=0.5, size=80),
                          "actual": truth + rng.normal(scale=0.3, size=80)})
    prism = pd.DataFrame({"drug": "d1", "line": lines[:60], "prism_auc": truth[:60] + rng.normal(scale=0.3, size=60)})
    s = score(preds, prism, min_lines=50).iloc[0]
    assert s.n == 60 and 0.5 < s.r_prism < 1 and 0.7 < s.r_assays < 1
    assert _key("Nutlin-3a (-)") == _key("nutlin3a") and _key("AZD-7762") == "azd7762"
