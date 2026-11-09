"""Models for drug response: one regression per drug, p >> n.

Every model receives the same training-fold design:
  * the N_TOP_VARIABLE most variable genes *in the training lines* (log2 TPM + 1),
  * one-hot tissue indicators,
standardised with training-fold statistics. Penalties are tuned by inner K-fold CV
on the training lines only (nested cross-validation).

Models
  Mean           training mean (no information)
  Tissue         tissue means (what you know from the cancer type alone)
  Ridge          L2, all genes + tissue; closed form via the SVD for all lambdas
  Lasso          L1
  Elastic net    L1 + L2, mixing chosen by CV
  Adaptive lasso lasso with weights 1/|ridge coefficient|
  PCR            principal components regression (dense benchmark)
  Group lasso    L2,1 over co-expression modules (gene clusters), FISTA from scratch
"""
from __future__ import annotations

import warnings

import numpy as np
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNetCV, LassoCV
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore", category=ConvergenceWarning)

RIDGE_ALPHAS = np.logspace(-1, 6, 40)
EN_L1 = [0.2, 0.8]


# --------------------------------------------------------------------------- design
class Design:
    """Fold-specific preprocessing, estimated on training lines only:

    1. Batch correction. Expression comes from two pipelines (Sanger, Broad) and the source
       is almost perfectly predictable from raw profiles. Each gene's source effect is
       estimated by regressing it on [source, tissue dummies] (so tissue differences are not
       removed) and subtracted.
    2. Gene filter: the n_genes most variable corrected genes.
    3. One-hot tissue indicators.
    4. Standardisation with training means and standard deviations.
    """

    def __init__(self, n_genes: int):
        self.n_genes = n_genes

    def fit(self, E_tr: np.ndarray, tissue_tr: np.ndarray, sanger_tr: np.ndarray):
        self.tissues_ = np.array(sorted(set(tissue_tr)))
        s = sanger_tr.astype(np.float64)
        T = self._tissue(tissue_tr)
        A = np.c_[np.ones(len(s)), T]
        sr = s - A @ np.linalg.lstsq(A, s, rcond=None)[0]       # source, residualised on tissue
        denom = sr @ sr
        self.batch_ = (sr @ E_tr) / denom if denom > 1e-9 else np.zeros(E_tr.shape[1])
        Ec = E_tr - np.outer(s, self.batch_)
        self.genes_ = np.sort(np.argsort(Ec.var(axis=0))[::-1][: self.n_genes])
        X = np.hstack([Ec[:, self.genes_], T])
        self.mu_ = X.mean(0)
        sd = X.std(0)
        self.sd_ = np.where(sd > 0, sd, 1.0)
        return self

    def _tissue(self, tissue):
        return (np.asarray(tissue)[:, None] == self.tissues_[None, :]).astype(np.float64)

    def transform(self, E, tissue, sanger):
        Ec = E[:, self.genes_] - np.outer(sanger.astype(np.float64), self.batch_[self.genes_])
        X = np.hstack([Ec, self._tissue(tissue)])
        return (X - self.mu_) / self.sd_

    @property
    def n_features(self):
        return len(self.genes_) + len(self.tissues_)

    def is_tissue(self):
        return np.r_[np.zeros(len(self.genes_), bool), np.ones(len(self.tissues_), bool)]


def inner_folds(n, k, seed):
    return list(KFold(k, shuffle=True, random_state=seed).split(np.arange(n)))


# --------------------------------------------------------------------------- base
class Model:
    name = "base"
    sparse = False

    def fit(self, X, y, folds, design: Design):
        raise NotImplementedError

    def predict(self, X):
        return self.intercept_ + X @ self.coef_


class MeanModel(Model):
    name = "Mean"

    def fit(self, X, y, folds, design):
        self.intercept_, self.coef_ = float(y.mean()), np.zeros(X.shape[1])
        return self


class TissueModel(Model):
    """OLS on the tissue indicators only (= tissue means), lightly ridged for rare tissues."""
    name = "Tissue"

    def fit(self, X, y, folds, design):
        t = design.is_tissue()
        Z = X[:, t]
        ym = y.mean()
        b = np.linalg.solve(Z.T @ Z + 1e-3 * np.eye(Z.shape[1]), Z.T @ (y - ym))
        self.coef_ = np.zeros(X.shape[1])
        self.coef_[t] = b
        self.intercept_ = ym
        return self


def _thin_svd(X):
    """SVD of X (n x p, n < p) via the n x n Gram matrix: much cheaper than LAPACK on X."""
    K = X @ X.T
    w, U = np.linalg.eigh(K)
    keep = w > w.max() * 1e-10
    w, U = w[keep][::-1], U[:, keep][:, ::-1]
    s = np.sqrt(w)
    Vt = (U.T @ X) / s[:, None]
    return U, s, Vt


def _ridge_path(X, y, alphas):
    """Ridge coefficients for every alpha from one SVD (X, y centred)."""
    U, s, Vt = _thin_svd(X)
    Uty = U.T @ y
    d = s[None, :] / (s[None, :] ** 2 + alphas[:, None])          # (n_alpha, r)
    return (d * Uty[None, :]) @ Vt                                   # (n_alpha, p)


class RidgeModel(Model):
    name = "Ridge"

    def fit(self, X, y, folds, design):
        errs = np.zeros(len(RIDGE_ALPHAS))
        for tr, va in folds:
            xm, ym = X[tr].mean(0), y[tr].mean()
            B = _ridge_path(X[tr] - xm, y[tr] - ym, RIDGE_ALPHAS)
            pred = ym + (X[va] - xm) @ B.T
            errs += ((y[va][:, None] - pred) ** 2).sum(0)
        self.alpha_ = RIDGE_ALPHAS[int(np.argmin(errs))]
        xm, ym = X.mean(0), y.mean()
        self.coef_ = _ridge_path(X - xm, y - ym, np.array([self.alpha_]))[0]
        self.intercept_ = ym - xm @ self.coef_
        return self


class LassoModel(Model):
    name = "Lasso"
    sparse = True

    def fit(self, X, y, folds, design):
        m = LassoCV(cv=folds, alphas=30, eps=1e-2, max_iter=3000, tol=1e-3,
                    precompute=True, selection="random", random_state=0).fit(X, y)
        self.alpha_, self.intercept_, self.coef_ = m.alpha_, m.intercept_, m.coef_
        return self


class ElasticNetModel(Model):
    name = "Elastic net"
    sparse = True

    def fit(self, X, y, folds, design):
        m = ElasticNetCV(l1_ratio=EN_L1, cv=folds, alphas=30, eps=1e-2, max_iter=3000,
                         tol=1e-3, precompute=True, selection="random", random_state=0).fit(X, y)
        self.alpha_, self.l1_ratio_ = m.alpha_, m.l1_ratio_
        self.intercept_, self.coef_ = m.intercept_, m.coef_
        return self


class AdaptiveLassoModel(Model):
    """Zou (2006). Weights from the CV-tuned ridge fit; solved by rescaling the columns."""
    name = "Adaptive lasso"
    sparse = True

    def fit(self, X, y, folds, design):
        r = RidgeModel().fit(X, y, folds, design)
        w = np.abs(r.coef_) / np.abs(r.coef_).max() + 1e-6
        m = LassoCV(cv=folds, alphas=30, eps=1e-3, max_iter=3000, tol=1e-3,
                    precompute=True, selection="random", random_state=0).fit(X * w, y)
        self.alpha_, self.intercept_, self.coef_ = m.alpha_, m.intercept_, m.coef_ * w
        return self


class PCRModel(Model):
    """Principal components regression; number of components by inner CV."""
    name = "PCR"
    ks = (1, 2, 5, 10, 20, 50, 100, 200)

    def fit(self, X, y, folds, design):
        errs = np.zeros(len(self.ks))
        for tr, va in folds:
            xm, ym = X[tr].mean(0), y[tr].mean()
            U, s, Vt = _thin_svd(X[tr] - xm)
            Zva = (X[va] - xm) @ Vt.T
            uy = U.T @ (y[tr] - ym)
            for j, k in enumerate(self.ks):
                k = min(k, len(s))
                pred = ym + Zva[:, :k] @ (uy[:k] / s[:k])
                errs[j] += ((y[va] - pred) ** 2).sum()
        self.k_ = min(self.ks[int(np.argmin(errs))], min(X.shape))
        xm, ym = X.mean(0), y.mean()
        U, s, Vt = _thin_svd(X - xm)
        k = self.k_
        self.coef_ = Vt[:k].T @ ((U[:, :k].T @ (y - ym)) / s[:k])
        self.intercept_ = ym - xm @ self.coef_
        return self


# --------------------------------------------------------------------------- group lasso
def coexpression_modules(X_genes: np.ndarray, n_modules: int, seed: int = 0) -> np.ndarray:
    """Cluster standardised genes by their profiles across training lines (k-means on
    the gene vectors = clustering on correlation, since columns are standardised)."""
    G = X_genes.T / np.sqrt(X_genes.shape[0])
    return KMeans(n_modules, n_init=2, random_state=seed).fit_predict(G)


def group_lasso_path(X, y, groups, lambdas, max_iter=400, tol=1e-4):
    """FISTA for (1/2n)||y - Xb||^2 + lam * sum_g sqrt(p_g) ||b_g||_2 (X, y centred)."""
    n, p = X.shape
    order = np.argsort(groups, kind="stable")
    gs = groups[order]
    bounds = np.r_[0, np.flatnonzero(np.diff(gs)) + 1, p]
    wts = np.sqrt(np.diff(bounds))
    Xo = X[:, order]
    L = np.linalg.norm(Xo, 2) ** 2 / n
    step = 1.0 / L
    b = np.zeros(p)
    out = []
    for lam in lambdas:
        x_old, v, t = b.copy(), b.copy(), 1.0
        for _ in range(max_iter):
            u = v - step * (Xo.T @ (Xo @ v - y)) / n
            x = np.zeros(p)
            for k in range(len(wts)):
                a, e = bounds[k], bounds[k + 1]
                nrm = np.linalg.norm(u[a:e])
                thr = step * lam * wts[k]
                if nrm > thr:
                    x[a:e] = (1 - thr / nrm) * u[a:e]
            t_new = (1 + np.sqrt(1 + 4 * t * t)) / 2
            v = x + ((t - 1) / t_new) * (x - x_old)
            done = np.linalg.norm(x - x_old) <= tol * max(1.0, np.linalg.norm(x_old))
            x_old, t = x, t_new
            if done:
                break
        b = x_old
        bb = np.empty(p)
        bb[order] = b
        out.append(bb)
    return np.array(out)


def group_lambda_max(X, y, groups):
    n = len(y)
    return max(np.linalg.norm(X[:, groups == g].T @ y) / (n * np.sqrt((groups == g).sum()))
               for g in np.unique(groups))


class GroupLassoModel(Model):
    """Group lasso over co-expression modules; each tissue indicator is its own group."""
    name = "Group lasso"
    sparse = True
    n_lambdas = 12

    def __init__(self, n_modules=100):
        self.n_modules = n_modules

    def fit(self, X, y, folds, design):
        t = design.is_tissue()
        mod = coexpression_modules(X[:, ~t], self.n_modules)
        groups = np.r_[mod, self.n_modules + np.arange(t.sum())]
        ym = y.mean()
        lmax = group_lambda_max(X - X.mean(0), y - ym, groups)
        lambdas = lmax * np.logspace(0, -2, self.n_lambdas)
        errs = np.zeros(len(lambdas))
        for tr, va in folds:
            xm, ymt = X[tr].mean(0), y[tr].mean()
            P = group_lasso_path(X[tr] - xm, y[tr] - ymt, groups, lambdas)
            pred = ymt + (X[va] - xm) @ P.T
            errs += ((y[va][:, None] - pred) ** 2).sum(0)
        k = int(np.argmin(errs))
        self.alpha_ = lambdas[k]
        xm = X.mean(0)
        self.coef_ = group_lasso_path(X - xm, y - ym, groups, lambdas[: k + 1])[-1]
        self.intercept_ = ym - xm @ self.coef_
        self.groups_ = groups
        return self


# every eligible drug
MAIN_MODELS = [MeanModel, TissueModel, RidgeModel, LassoModel, ElasticNetModel, PCRModel]
# the focus panel only (slower)
PANEL_MODELS = [AdaptiveLassoModel, GroupLassoModel]
FAST_MODELS = MAIN_MODELS
