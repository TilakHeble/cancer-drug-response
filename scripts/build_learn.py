"""Build the learn/ notebooks: regularised regression when features vastly outnumber samples.

    python scripts/build_learn.py            # all
    python scripts/build_learn.py 03 05      # selected
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nbtools import code, md, write_and_run  # noqa: E402

LEARN = Path(__file__).resolve().parents[1] / "learn"

SETUP = code("""
import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent / "src"))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (8, 4.2), "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.3})
rng = np.random.default_rng(7)
""")

REAL = code("""
# The project's data: one drug, its screened cell lines, 2,000 most variable genes
from drugresponse.experiment import Store
from drugresponse.models import Design
S = Store()
drug_name = "Nutlin-3a (-)"
d = S.drugs.index[S.drugs.drug_name == drug_name][0]
ok = S.R[d].notna().values
des = Design(2000).fit(S.Ev[ok], S.tissue[ok], S.sanger[ok])
X = des.transform(S.Ev[ok], S.tissue[ok], S.sanger[ok])[:, ~des.is_tissue()]   # genes only
y = S.R[d].values[ok]
genes = S.genes[des.genes_]
print(f"{drug_name}: {X.shape[0]} cell lines x {X.shape[1]} genes")
""")

# ----------------------------------------------------------------------------- 01
L01 = [
    md("""
# 1. When features outnumber samples (p >> n)

**Learning goals**
* See exactly how ordinary least squares breaks when p > n.
* Understand the minimum-norm solution and why perfect training fit means nothing.
* Meet this project's problem: about 900 cell lines, 18,000 genes.

In genomics, p >> n is the normal case: a few hundred samples, tens of thousands of genes.
With p > n the matrix $X^\\top X$ ($p \\times p$) has rank at most $n$, so it cannot be inverted.
The equations $X\\beta = y$ have infinitely many exact solutions: any $\\beta$ plus any vector in
the null space of $X$ (dimension at least $p - n$) fits the training data perfectly.
"""),
    SETUP,
    md("## Infinitely many perfect fits"),
    code("""
n, p = 30, 100
X = rng.normal(size=(n, p))
beta_true = np.zeros(p); beta_true[:3] = [2, -1.5, 1]
y = X @ beta_true + rng.normal(size=n)

b_min = np.linalg.pinv(X) @ y                    # minimum-norm least squares
null = np.linalg.svd(X)[2][n:].T                 # basis of the null space (p - n columns)
b_other = b_min + null @ rng.normal(size=p - n) * 2
for name, b in [("minimum norm", b_min), ("min norm + null-space vector", b_other)]:
    print(f"{name:<30} train RSS = {np.sum((y - X @ b) ** 2):.1e}   ||b|| = {np.linalg.norm(b):.1f}")
Xt = rng.normal(size=(1000, p)); yt = Xt @ beta_true + rng.normal(size=1000)
for name, b in [("minimum norm", b_min), ("min norm + null-space vector", b_other), ("truth", beta_true)]:
    print(f"{name:<30} test RMSE = {np.sqrt(np.mean((yt - Xt @ b) ** 2)):.2f}")
"""),
    md("""
Both fit the training data exactly, and both predict new data badly. The data cannot tell them
apart. **A penalty is a way to choose among the perfect fits**: ridge picks small $\\lVert\\beta\\rVert_2$,
the lasso picks small $\\lVert\\beta\\rVert_1$ (which favours sparse solutions). Minimum-norm OLS is
the limit of ridge as $\\lambda \\to 0$.

## The same thing on the real data
"""),
    REAL,
    code("""
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_predict, KFold
cv = KFold(5, shuffle=True, random_state=0)
rows = []
for lam in [1e-6, 1e0, 1e2, 1e3, 1e4, 1e5]:
    pred = cross_val_predict(Ridge(alpha=lam), X, y, cv=cv)
    fit = Ridge(alpha=lam).fit(X, y)
    rows.append((lam, np.corrcoef(fit.predict(X), y)[0, 1], np.corrcoef(pred, y)[0, 1]))
pd.DataFrame(rows, columns=["lambda", "training r", "cross-validated r"]).round(3)
"""),
    md("""
With $\\lambda \\approx 0$ the training correlation is 1 (memorisation) and the cross-validated
correlation is much lower. As $\\lambda$ grows the two meet. Somewhere in between is the best
out-of-sample accuracy, and choosing that $\\lambda$ honestly is the subject of notebook 5.

**Key takeaways**
* With p > n, OLS has infinitely many perfect fits; the training error is always zero.
* Regularisation chooses one of them using a prior preference (small, or sparse, coefficients).
* Never judge a p >> n model by its training fit.
"""),
]

# ----------------------------------------------------------------------------- 02
L02 = [
    md("""
# 2. Ridge in p >> n: the kernel trick and the SVD

**Learning goals**: compute ridge cheaply when p is huge, see what it does to the data's principal
directions, and connect it to principal components regression (PCR).

$$\\hat\\beta = (X^\\top X + \\lambda I_p)^{-1} X^\\top y = X^\\top (X X^\\top + \\lambda I_n)^{-1} y$$

The second form (the **kernel** or dual form) inverts an $n \\times n$ matrix instead of a $p \\times p$ one.
With 900 lines and 18,000 genes that is the difference between a 900² and an 18,000² problem.
Through the SVD $X = UDV^\\top$ (only $n$ non-zero singular values), every $\\lambda$ comes almost
free once $U$, $D$, $V$ are known:
$$\\hat\\beta(\\lambda) = V \\,\\mathrm{diag}\\Big(\\frac{d_j}{d_j^2+\\lambda}\\Big) U^\\top y$$
This is exactly how the project's `RidgeModel` tunes $\\lambda$ (`drugresponse.models._ridge_path`).
"""),
    SETUP,
    code("""
import time
n, p = 400, 8000
X = rng.normal(size=(n, p)); y = X[:, :20] @ rng.normal(size=20) + rng.normal(size=n)
lam = 50.0
t = time.time(); b_primal = np.linalg.solve(X.T @ X + lam * np.eye(p), X.T @ y); t_primal = time.time() - t
t = time.time(); b_dual = X.T @ np.linalg.solve(X @ X.T + lam * np.eye(n), y); t_dual = time.time() - t
print(f"primal (p x p): {t_primal:.2f}s   dual (n x n): {t_dual:.3f}s   max difference {np.abs(b_primal - b_dual).max():.1e}")
"""),
    md("## Shrinkage along principal directions"),
    REAL,
    code("""
Xc = X - X.mean(0); yc = y - y.mean()
U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(s ** 2 / np.sum(s ** 2), ".-", ms=3); ax[0].set_xscale("log")
ax[0].set(xlabel="principal component", ylabel="share of variance", title="Expression: a few dominant directions")
for lam in [1e1, 1e3, 1e5]:
    ax[1].plot(s ** 2 / (s ** 2 + lam), label=f"lambda = {lam:g}")
ax[1].set_xscale("log"); ax[1].legend()
ax[1].set(xlabel="principal component", ylabel="shrinkage factor d^2 / (d^2 + lambda)",
          title="Ridge keeps high-variance directions")
df = [np.sum(s ** 2 / (s ** 2 + lam)) for lam in [1e1, 1e3, 1e5]]
print("effective degrees of freedom:", np.round(df, 1), "out of", len(s))
"""),
    md("""
Gene expression is dominated by a few large axes of variation (tissue of origin, proliferation,
epithelial vs mesenchymal state). Ridge keeps those and shrinks the hundreds of small directions,
which are mostly noise with n = 900. PCR does the same with a hard cut: keep the first k
components, drop the rest. In this project PCR is the "dense benchmark".

**Key takeaways**
* Use the n x n (kernel) form when p >> n; tune all lambdas from one SVD.
* Effective degrees of freedom, not the number of genes, measure ridge's complexity.
* Ridge assumes many small effects spread across correlated genes, a natural prior for expression.
"""),
]

# ----------------------------------------------------------------------------- 03
L03 = [
    md("""
# 3. The lasso when p >> n: sparsity, correlated genes, instability

**Learning goals**: why the lasso selects at most n genes, why its gene list changes from sample to
sample, and how to report it honestly (selection frequencies, stability selection).
"""),
    SETUP,
    REAL,
    code("""
from sklearn.linear_model import lasso_path
alphas, coefs, _ = lasso_path(X - X.mean(0), y - y.mean(), n_alphas=60, eps=1e-2)
nnz = (np.abs(coefs) > 0).sum(0)
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(alphas, coefs.T, lw=0.6); ax[0].set_xscale("log"); ax[0].invert_xaxis()
ax[0].set(xlabel="lambda", ylabel="coefficient", title=f"Lasso paths, {X.shape[1]} genes")
ax[1].plot(alphas, nnz); ax[1].set_xscale("log"); ax[1].invert_xaxis()
ax[1].set(xlabel="lambda", ylabel="genes selected", title="Model size along the path");
first = [genes[j] for j in np.argsort([np.argmax(np.abs(coefs[j]) > 0) if np.any(coefs[j]) else 999 for j in range(len(genes))])[:10]]
print("first genes to enter:", first)
"""),
    md("""
Nutlin-3a blocks MDM2, which only kills cells whose **TP53** is intact. Wild-type TP53 switches on
its target genes, such as **MDM2**, **CDKN1A** (p21), **EDA2R**, **SPATA18** and **RPS27L**, so
high expression of those genes marks lines that respond. If the first genes to enter are TP53 targets,
the lasso has found real biology from expression alone.

## Selection is unstable
Refit on 20 random subsamples of 80% of the lines and count how often each gene is chosen.
"""),
    code("""
from sklearn.linear_model import LassoCV
counts = pd.Series(0, index=genes, dtype=float); sizes = []
for b in range(20):
    idx = rng.choice(len(y), int(0.8 * len(y)), replace=False)
    m = LassoCV(cv=3, alphas=20, eps=1e-2, max_iter=2000, tol=1e-3, precompute=True).fit(X[idx], y[idx])
    sel = genes[np.abs(m.coef_) > 0]; counts[sel] += 1; sizes.append(len(sel))
freq = counts / 20
print(f"genes selected per fit: median {np.median(sizes):.0f}")
print(f"selected in >= 90% of fits: {(freq >= 0.9).sum()};  in 10-90%: {((freq > 0.1) & (freq < 0.9)).sum()}")
freq.sort_values(ascending=False).head(15).round(2)
"""),
    md("""
A small core of genes is chosen almost every time, and a long tail flickers in and out. Among
correlated genes the lasso tends to pick one, more or less at random. **Stability selection**
(Meinshausen and Bühlmann, 2010) keeps genes chosen in a high share of subsamples. The project
reports selection frequency across the five outer folds for the same reason.

**Key takeaways**
* The lasso can select at most n variables when p > n, and usually far fewer.
* A single lasso fit's gene list is not a finding; its stable core may be.
* Prediction accuracy and selection stability are different goals; the lasso can do well at the
  first while being poor at the second.
"""),
]

# ----------------------------------------------------------------------------- 04
L04 = [
    md("""
# 4. Elastic net and group lasso: using gene structure

**Learning goals**: the elastic net's grouping effect and whether it matters on real data, and the
group lasso on gene modules, with a from-scratch solver.

Genes work in programmes: dozens of genes rise and fall together. The lasso picks one member of
each programme; the elastic net ($\\alpha\\lVert\\beta\\rVert_1 + \\frac{1-\\alpha}{2}\\lVert\\beta\\rVert_2^2$)
tends to keep the whole correlated set. The group lasso goes further: it is told the groups and
keeps or drops each one as a unit.
"""),
    SETUP,
    REAL,
    code("""
from sklearn.linear_model import ElasticNet, Lasso
Xc, yc = X - X.mean(0), y - y.mean()
# How does the weight spread as the penalty moves from lasso (1.0) towards ridge?
# Each mixing value tuned by 3-fold CV; "effective genes" = (sum |b|)^2 / sum b^2,
# the number of equally weighted genes that would give the same spread.
from sklearn.linear_model import ElasticNetCV
rows = []
for l1 in [1.0, 0.5, 0.2, 0.05]:
    m = ElasticNetCV(l1_ratio=l1, cv=3, alphas=20, max_iter=5000, tol=1e-3).fit(Xc, yc)
    b = m.coef_
    eff = (np.abs(b).sum() ** 2) / max((b ** 2).sum(), 1e-12)
    top5 = np.sort(np.abs(b))[::-1][:5].sum() / max(np.abs(b).sum(), 1e-12)
    rows.append((l1, int((b != 0).sum()), round(eff, 1), round(top5, 2), round(-m.mse_path_.mean(1).min(), 5)))
pd.DataFrame(rows, columns=["l1_ratio", "genes selected", "effective genes", "weight share of top 5", "-CV MSE"])
"""),
    md("""
On this drug the mixing weight hardly matters. Once each setting is tuned by CV, the lasso and
the elastic net select about the same number of genes and spread weight the same way. The L2 part
only changes the solution when it is large relative to the L1 part, and CV prefers penalties too
small for that. This matches the main results: across 281 drugs the lasso and the elastic net have
the same median r (0.382). The textbook grouping effect is real, but it is not guaranteed to matter
on a given dataset, so check it rather than assume it.

## Group lasso on co-expression modules
The project clusters genes into 100 modules by their expression profiles (k-means on standardised
gene vectors, which is clustering on correlation) inside each training fold, then penalises
$\\lambda \\sum_g \\sqrt{p_g}\\,\\lVert\\beta_g\\rVert_2$. The solver is accelerated proximal gradient (FISTA),
written from scratch in `drugresponse.models.group_lasso_path`.
"""),
    code("""
from drugresponse.models import coexpression_modules, group_lasso_path, group_lambda_max
groups = coexpression_modules(Xc, 100)
lmax = group_lambda_max(Xc, yc, groups)
path = group_lasso_path(Xc, yc, groups, lmax * np.logspace(0, -1.5, 8), max_iter=3000, tol=1e-6)
for lam_frac, b in zip(np.logspace(0, -1.5, 8), path):
    active = np.unique(groups[np.abs(b) > 0])
    print(f"lambda = {lam_frac:.3f} x max: {len(active):3d} modules active, {np.sum(np.abs(b) > 0):4d} genes")
"""),
    code("""
b = path[-3]
act = pd.Series(np.abs(b), index=genes).groupby(groups).sum().sort_values(ascending=False)
for g in act.index[:3]:
    members = genes[groups == g]
    top = pd.Series(b[groups == g], index=members).abs().sort_values(ascending=False).index[:8]
    print(f"module {g} ({len(members)} genes): {', '.join(top)}")
"""),
    md("""
A module is selected or not as a whole. The output reads like a short list of biological programmes
rather than a scattered list of genes. The cost is speed: the group lasso is solved with an iterative
proximal method and tuned on a 12-point path, so the project runs it on the 36-drug focus panel only.

**Key takeaways**
* Elastic net: sparse, and in principle shares weight across correlated genes; on this data, at
  CV-chosen penalties, it behaves almost exactly like the lasso.
* Group lasso: selection at the level of a module or pathway; within a module, ridge-like shrinkage.
* Groups can come from biology (curated pathways) or from the data (co-expression), as here.
"""),
]

# ----------------------------------------------------------------------------- 05
L05 = [
    md("""
# 5. Nested cross-validation and the classic genomics leak

**Learning goals**: why tuning and evaluating on the same folds is optimistic, and why
selecting genes on the full data before cross-validating can manufacture accuracy from pure noise.

Ambroise and McLachlan (2002) showed that many published gene-expression classifiers reported
near-perfect accuracy because the genes were chosen using all samples, including those later used
as "test" data. The cure is to put **every** data-dependent step (gene filtering, scaling, batch
correction, tuning) inside the training folds.
"""),
    SETUP,
    md("## Pure noise: 200 samples, 5,000 random genes, a random response"),
    code("""
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_predict
n, p = 200, 5000
X = rng.normal(size=(n, p)); y = rng.normal(size=n)          # no signal at all
cv = KFold(5, shuffle=True, random_state=0)

# WRONG: pick the 50 genes most correlated with y using all samples, then cross-validate
corr = np.abs((X - X.mean(0)).T @ (y - y.mean())) / (X.std(0) * y.std() * n)
top = np.argsort(corr)[-50:]
pred_leak = cross_val_predict(RidgeCV(alphas=np.logspace(-2, 4, 20)), X[:, top], y, cv=cv)

# RIGHT: choose the 50 genes inside each training fold
pred_ok = np.zeros(n)
for tr, te in cv.split(X):
    c = np.abs((X[tr] - X[tr].mean(0)).T @ (y[tr] - y[tr].mean()))
    t = np.argsort(c)[-50:]
    pred_ok[te] = RidgeCV(alphas=np.logspace(-2, 4, 20)).fit(X[tr][:, t], y[tr]).predict(X[te][:, t])
print(f"leaky selection : cross-validated r = {np.corrcoef(pred_leak, y)[0, 1]:.2f}")
print(f"honest selection: cross-validated r = {np.corrcoef(pred_ok, y)[0, 1]:.2f}   (truth: 0)")
"""),
    md("""
The leaky pipeline "predicts" pure noise with a sizeable correlation. In this project the gene filter
uses expression **variance** (it never sees the response), but it is still computed inside each
outer training fold, along with the batch correction and standardisation (`drugresponse.models.Design`).

## Nested CV: tuning is part of the model
If $\\lambda$ is chosen to minimise the cross-validation error and that same error is reported,
the report is biased downwards: the minimum of many noisy estimates is too low. Nested CV tunes
$\\lambda$ in an inner loop on the training fold only and reports the outer-fold error.
"""),
    code("""
from sklearn.linear_model import LassoCV
X = rng.normal(size=(150, 2000)); y = X[:, :5] @ np.full(5, 0.4) + rng.normal(size=150)
outer = KFold(5, shuffle=True, random_state=1)
m = LassoCV(cv=outer, alphas=40).fit(X, y)
non_nested = m.mse_path_.mean(1).min()
nested = []
for tr, te in outer.split(X):
    mm = LassoCV(cv=5, alphas=40).fit(X[tr], y[tr])
    nested.append(np.mean((y[te] - mm.predict(X[te])) ** 2))
print(f"non-nested CV MSE (tuned and scored on the same folds): {non_nested:.3f}")
print(f"nested CV MSE (tuned inside, scored outside):           {np.mean(nested):.3f}")
"""),
    md("""
The gap is usually modest for one tuning parameter, and it grows with the number of choices tried
(genes kept, penalty, mixing weight, model family). The project's headline numbers are all outer-fold.

**Key takeaways**
* Every step that looks at the data belongs inside the training folds.
* Selecting features on all samples can create accuracy out of nothing.
* Report nested-CV error; the inner CV is for tuning only.
"""),
]

# ----------------------------------------------------------------------------- 06
L06 = [
    md("""
# 6. Batch effects and confounding: is the model learning biology or tissue?

**Learning goals**: detect a batch effect, remove it without removing real biology, and evaluate
within groups so a model cannot score by learning group labels.

Two traps in this data:
1. **Batch.** Expression comes from two pipelines (Sanger and Broad). The source can be predicted
   from the profiles almost perfectly.
2. **Tissue.** Blood-cancer lines respond to many drugs more than solid-tumour lines do. A model
   that only recognises "blood vs solid" from expression can post a good overall correlation.
"""),
    SETUP,
    code("""
from drugresponse import data
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
E = data.load_expression(); M = data.load_models().reindex(E.index)
s = (M.expression_source == "Sanger").values.astype(float)
T = pd.get_dummies(M.tissue).values.astype(float)
A = np.c_[np.ones(len(s)), T]
sr = s - A @ np.linalg.lstsq(A, s, rcond=None)[0]          # source, net of tissue
Xr = E.values.astype(float)
beta = (sr @ Xr) / (sr @ sr)                               # each gene's source effect
Xc = Xr - np.outer(s, beta)
for name, XX in [("raw", Xr), ("batch-corrected", Xc)]:
    v = XX.var(0); Z = XX[:, np.argsort(v)[-3000:]]; Z = (Z - Z.mean(0)) / Z.std(0)
    P = PCA(20, random_state=0).fit_transform(Z)
    a_src = cross_val_score(LogisticRegression(max_iter=4000), P, s, cv=5, scoring="roc_auc").mean()
    print(f"{name:<16} source predictable from 20 PCs: AUC {a_src:.3f}")
"""),
    md("""
The correction regresses each gene on [source, tissue] and subtracts only the source part.
Regressing on source alone would also remove tissue differences wherever the two sources sampled
tissues unevenly. In the project the correction is estimated inside each training fold.

## Evaluating beyond tissue
"""),
    code("""
from drugresponse.experiment import Store, run_drug
from drugresponse.evaluate import within_tissue_rho
from drugresponse.models import TissueModel, RidgeModel
S = Store()
d = S.drugs.index[S.drugs.drug_name == "Cytarabine"][0]
p, _ = run_drug(S, d, [TissueModel, RidgeModel], keep_coefs=False)
for m, g in p.groupby("model"):
    wr = within_tissue_rho(g, S.M.tissue) if m != "Tissue" else float("nan")
    print(f"{m:<7} overall r = {np.corrcoef(g.pred, g.actual)[0, 1]:.2f}   within-tissue Spearman = {wr:.2f}")
"""),
    md("""
The tissue-only model already reaches a respectable overall correlation for cytarabine (blood
cancers are far more sensitive). The right question for an expression model is how much it adds
to that, and how well it ranks lines *within* a cancer type. That is why the project reports the
gain over the tissue model and the within-tissue Spearman, and why it tests cancer types held out
entirely (notebooks/05).

**Key takeaways**
* Check whether a technical variable is predictable from your features; if it is, correct for it.
* Correct for batch while protecting the biology you care about (regress on both).
* Always compare with a baseline built from the confounder alone.
"""),
]

# ----------------------------------------------------------------------------- 07
L07 = [
    md("""
# 7. Reading coefficients in genomics

**Learning goals**: what a penalised coefficient does and does not mean; signs and standardisation;
why "selected" is not "significant"; and how to sanity-check a model against known biology.

* A coefficient is the effect of a gene **holding the other genes fixed**. With thousands of
  correlated genes, that conditional effect is rarely what a biologist means by "this gene matters".
* Penalised coefficients are **biased towards zero**, more for the lasso's small coefficients, less
  for the adaptive lasso's large ones.
* Here the response is AUC: a **negative** coefficient means higher expression, lower AUC, i.e. a
  more **sensitive** cell line.
"""),
    SETUP,
    REAL,
    code("""
from scipy import stats
from sklearn.linear_model import ElasticNetCV
m = ElasticNetCV(l1_ratio=[0.2, 0.8], cv=5, alphas=30, max_iter=5000).fit(X, y)
coef = pd.Series(m.coef_, index=genes)
sel = coef[coef != 0].sort_values()
marg = pd.Series([stats.spearmanr(X[:, j], y).statistic for j in range(X.shape[1])], index=genes)
tbl = pd.DataFrame({"elastic-net coef": sel, "marginal Spearman with AUC": marg[sel.index]})
print(f"elastic net (mixing {m.l1_ratio_}): {len(sel)} genes selected")
tbl.head(12).round(3)
"""),
    md("""
For Nutlin-3a the most negative coefficients should belong to TP53 target genes, the
transcriptional readout of wild-type TP53. Lines with an active TP53 pathway respond to MDM2
inhibition. The marginal correlation has the same sign, so the multivariable model agrees with
the simple univariate picture. When the two disagree in sign, it is usually because a correlated
neighbour has taken the weight.

## Selected is not significant
A gene being selected does not come with a p-value. Valid post-selection inference needs sample
splitting or selective-inference methods (Lee et al., 2016). For a predictive model the honest
evidence is out-of-sample accuracy, plus stability of the selected set across folds, plus agreement
with known biology. The project reports all three (notebooks/04).

**Key takeaways**
* Interpret signs on the response's scale (here: negative = sensitising).
* Compare multivariable coefficients with marginal associations; disagreements point to correlation.
* Validate with known biomarkers before trusting new ones.
"""),
]

README = """# Learn: regularised regression when features outnumber samples

Seven notebooks on the ideas this project relies on, using its real data (cancer cell lines x genes).
They build on the [regularisation notebooks in the GDP nowcasting project](https://github.com/TilakHeble/uk-gdp-nowcast/tree/main/learn),
which cover ridge, lasso and elastic net from scratch. Here the focus is what changes when p >> n.

| # | Notebook | What you learn |
|---|---|---|
| 1 | [When features outnumber samples](01_p_much_larger_than_n.ipynb) | Infinitely many perfect fits, minimum-norm OLS, why training fit means nothing |
| 2 | [Ridge: kernel trick and SVD](02_ridge_kernel_and_svd.ipynb) | The n x n dual form, all lambdas from one SVD, effective degrees of freedom, PCR |
| 3 | [The lasso in p >> n](03_lasso_sparsity_and_stability.ipynb) | Paths, at most n genes, unstable selection, stability selection |
| 4 | [Elastic net and group lasso](04_elastic_net_and_group_lasso.ipynb) | The grouping effect on co-expressed genes, group lasso on gene modules (FISTA) |
| 5 | [Nested CV and the genomics leak](05_nested_cv_and_leakage.ipynb) | Selecting genes on all samples creates accuracy from noise; nested vs non-nested CV |
| 6 | [Batch effects and confounding](06_batch_effects_and_confounding.ipynb) | Detecting and removing a batch effect without removing tissue; evaluating within tissue |
| 7 | [Reading coefficients](07_reading_coefficients.ipynb) | Signs, shrinkage bias, selected vs significant, checking against known biology |

Run them from this folder after `python scripts/run_pipeline.py --data` has built `data/processed/`.

**Further reading**
* Hastie, Tibshirani and Wainwright, *Statistical Learning with Sparsity* (free online), ch. 2-4 and 11.
* Ambroise and McLachlan (2002), Selection bias in gene extraction on the basis of microarray gene-expression data, *PNAS*.
* Meinshausen and Bühlmann (2010), Stability selection, *JRSS B*.
* Garnett et al. (2012), Systematic identification of genomic markers of drug sensitivity in cancer cells, *Nature*.
* Costello et al. (2014), A community effort to assess and improve drug sensitivity prediction algorithms, *Nature Biotechnology*.
"""

NOTEBOOKS = {
    "01_p_much_larger_than_n.ipynb": L01, "02_ridge_kernel_and_svd.ipynb": L02,
    "03_lasso_sparsity_and_stability.ipynb": L03, "04_elastic_net_and_group_lasso.ipynb": L04,
    "05_nested_cv_and_leakage.ipynb": L05, "06_batch_effects_and_confounding.ipynb": L06,
    "07_reading_coefficients.ipynb": L07,
}

if __name__ == "__main__":
    only = sys.argv[1:]
    LEARN.mkdir(exist_ok=True)
    (LEARN / "README.md").write_text(README)
    for fname, cells in NOTEBOOKS.items():
        if only and not any(o in fname for o in only):
            continue
        write_and_run(cells, LEARN / fname)
