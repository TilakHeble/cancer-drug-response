"""Build and execute the analysis notebooks in notebooks/.

    python scripts/build_notebooks.py            # all
    python scripts/build_notebooks.py 03 05      # selected
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nbtools import code, md, write_and_run  # noqa: E402

NB = Path(__file__).resolve().parents[1] / "notebooks"

SETUP = code("""
import sys, warnings, pickle
from pathlib import Path
ROOT = Path.cwd().parent
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image, display
pd.set_option("display.width", 160); pd.set_option("display.max_columns", 20)
plt.rcParams.update({"figure.figsize": (9, 4.2), "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.3})
T = ROOT / "outputs" / "tables"; FIG = ROOT / "outputs" / "figures"
from drugresponse import data
""")

# ----------------------------------------------------------------------------- 01
N01 = [
    md("""
# 01. Data: cell lines, drugs, expression and a batch effect

**Question for the project:** can a cancer cell line's gene expression predict how strongly it
responds to a drug, beyond what its cancer type already tells us? And which genes carry the signal?

| Source | Content |
|---|---|
| GDSC2 release 8.5 (Wellcome Sanger Institute) | fitted dose-response curves: 295 drugs x up to 969 cell lines, CellTiter-Glo viability assay |
| GDSC1 release 8.5 | 402 drugs, older Syto60/resazurin assays; used for the cross-assay test |
| Cell Model Passports | RNA-seq (log2 TPM+1), cell-line annotation (tissue, cancer type), driver mutations |

The response is the **area under the dose-response curve (AUC)** over the screened concentration
range: 1 means no killing at any tested dose, lower means more sensitive.
"""),
    SETUP,
    code("""
E = data.load_expression(); M = data.load_models(); R = data.load_response(); drugs = data.load_drugs()
print(f"expression: {E.shape[0]} cell lines x {E.shape[1]} genes (expressed in >= 10% of lines)")
print(f"GDSC2 response: {R.shape[1]} drugs; cell lines per drug: median {R.notna().sum().median():.0f}")
print(f"drugs screened in >= 300 lines (analysed): {(R.notna().sum() >= 300).sum()}")
M.tissue.value_counts().head(15)
"""),
    md("## Response: some drugs barely vary, others split lines sharply"),
    code("""
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].hist(drugs.auc_sd, bins=40); ax[0].set(xlabel="SD of AUC across cell lines", ylabel="drugs",
                                             title="How much each drug's response varies")
for name in ["Dabrafenib", "Nutlin-3a (-)", "Gemcitabine"]:
    d = drugs.index[drugs.drug_name == name][0]
    ax[1].hist(R[d].dropna(), bins=40, alpha=0.5, label=name)
ax[1].legend(); ax[1].set(xlabel="AUC", title="Three response distributions");
"""),
    md("""
Dabrafenib (a BRAF inhibitor) splits the lines into a small, very sensitive group (BRAF-mutant
melanoma) and everyone else. Gemcitabine kills most lines, to varying degrees. A drug with almost
no spread has little to predict.

## Cancer type explains a lot
"""),
    code("""
d = drugs.index[drugs.drug_name == "Cytarabine"][0]
by = R[d].groupby(M.tissue.reindex(R.index)).median().dropna().sort_values()
ax = by.plot.barh(figsize=(7, 6)); ax.set(xlabel="median AUC", title="Cytarabine: median AUC by tissue");
"""),
    md("""
## A batch effect in the expression data
550 profiles come from the Broad Institute's pipeline and 403 from Sanger's. Both are on the
log2(TPM + 1) scale, but the source is almost perfectly predictable from the profiles. Each gene's
source effect is estimated controlling for tissue, and subtracted inside each training fold
(`drugresponse.models.Design`; demonstration in learn/06).
"""),
    code("""
from sklearn.decomposition import PCA
s = (M.reindex(E.index).expression_source == "Sanger").values
from drugresponse.models import Design
d_all = Design(3000).fit(E.values.astype(float), M.reindex(E.index).tissue.values, s)
Zc = d_all.transform(E.values.astype(float), M.reindex(E.index).tissue.values, s)[:, ~d_all.is_tissue()]
Zr = E.values[:, d_all.genes_].astype(float); Zr = (Zr - Zr.mean(0)) / Zr.std(0)
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
for a, Z, title in [(ax[0], Zr, "raw"), (ax[1], Zc, "batch-corrected")]:
    P = PCA(6, random_state=0).fit_transform(Z)
    a.scatter(P[~s, 3], P[~s, 4], s=6, alpha=0.5, label="Broad")
    a.scatter(P[s, 3], P[s, 4], s=6, alpha=0.5, label="Sanger")
    a.set(xlabel="PC4", ylabel="PC5", title=f"{title} expression"); a.legend()
"""),
]

# ----------------------------------------------------------------------------- 02
N02 = [
    md("""
# 02. Inside one drug: Nutlin-3a

Every model, one drug, step by step. Nutlin-3a inhibits MDM2, the protein that keeps TP53 in check.
It only works in cells with functional TP53, so the expected biology is clear: lines expressing
TP53 target genes should be sensitive (low AUC).
"""),
    SETUP,
    code("""
from drugresponse.experiment import Store, run_drug
from drugresponse.models import MAIN_MODELS
from drugresponse.evaluate import drug_metrics
S = Store()
d = S.drugs.index[S.drugs.drug_name == "Nutlin-3a (-)"][0]
print(f"cell lines screened: {S.R[d].notna().sum()}; outer folds: {S.fold.nunique()}")
p, coefs = run_drug(S, d, MAIN_MODELS)
drug_metrics(p, S.M.tissue).set_index("model")[["n", "r", "rho_within", "rmse", "r2"]].round(3)
"""),
    md("## Predicted vs measured (elastic net, out-of-fold)"),
    code("""
g = p[p.model == "Elastic net"]
t = S.M.tissue.reindex(g.line).values
fig, ax = plt.subplots(figsize=(6.5, 6))
for tt in pd.Series(t).value_counts().index[:8]:
    m = t == tt; ax.scatter(g.pred[m], g.actual[m], s=10, alpha=0.6, label=tt)
ax.scatter(g.pred[~pd.Series(t).isin(pd.Series(t).value_counts().index[:8]).values],
           g.actual[~pd.Series(t).isin(pd.Series(t).value_counts().index[:8]).values], s=10, alpha=0.4, c="grey", label="other")
lo, hi = g[["pred", "actual"]].min().min(), g[["pred", "actual"]].max().max()
ax.plot([lo, hi], [lo, hi], "k--", lw=1); ax.legend(fontsize=7)
ax.set(xlabel="predicted AUC", ylabel="measured AUC", title="Nutlin-3a, elastic net");
"""),
    md("## Which genes, and how stable?"),
    code("""
from drugresponse.evaluate import gene_frequency
for m in ["Elastic net", "Lasso"]:
    gf = gene_frequency(coefs, m, d)
    gf = gf[~gf.feature.str.startswith("tissue:")]
    print(f"{m}: {len(gf)} genes selected in at least one fold; {(gf.folds == 5).sum()} in all five")
    display(gf.head(12).round(4))
"""),
    md("""
Read the top genes against TP53 biology. Genes switched on by wild-type TP53 should have
**negative** coefficients, because their expression marks lines that respond. The most stable genes
include **CDKN1A** (p21), **PHLDA3**, **TRIM22**, **PURPL** and **PPP1R13L**, all known TP53 targets,
and all with negative coefficients. The model has rediscovered the drug's mechanism from expression
alone. **CDKN2A** (p16) enters with a *positive* coefficient (less sensitive). High p16 expression
is common in lines whose RB/p53 control is disrupted, which fits, but treat it as a hypothesis, not
a check. Learn/07 explains how to read signs and why a selected gene is not a significant one.

## How the penalty was chosen
"""),
    code("""
cc = pd.DataFrame([c for c in coefs if c["model"] in ("Ridge", "Lasso", "Elastic net", "PCR")])
cc[["model", "fold", "alpha", "l1_ratio", "k", "n_nonzero"]].round(4)
"""),
]

# ----------------------------------------------------------------------------- 03
N03 = [
    md("""
# 03. Results across all drugs

Nested 5-fold cross-validation for every drug screened in at least 300 GDSC2 cell lines. The outer
folds are the same for every drug and model (each line assigned once, stratified by tissue), so
model comparisons are paired.
"""),
    SETUP,
    code("""
summ = pd.read_csv(T / "summary.csv", index_col=0)
summ.round(3)
"""),
    code("display(Image(FIG / 'accuracy_by_model.png'))"),
    md("""
**How to read it.** `median_r` is the median across drugs of the correlation between out-of-fold
predicted and measured AUC. `median_rho_within` is the median within-tissue Spearman correlation:
how well the model ranks lines inside the same cancer type, which tissue alone cannot do.
`share_beats_tissue` is the share of drugs on which the model's r beats the tissue-only model.
The Wilcoxon p-value tests the paired per-drug differences (drugs are not independent: several share
targets, so treat it as descriptive).
"""),
    code("""
pw = pd.read_csv(T / "pairwise_wins.csv", index_col=0)
print("share of drugs on which the row model has higher r than the column model")
pw.round(2)
"""),
    code("""
pd.read_csv(T / "best_model_counts.csv", index_col=0)
"""),
    md("## Where does expression add to cancer type?"),
    code("display(Image(FIG / 'ridge_vs_tissue.png'))"),
    code("""
per = pd.read_csv(T / "per_drug_ridge.csv", dtype={"drug": str})
cols = ["drug_name", "target", "n", "r_tissue", "r", "r_elastic_net", "gain_vs_tissue", "rho_within"]
print("Most predictable drugs (ridge)"); display(per[cols].head(12).round(3))
print("Largest gain over tissue"); display(per.sort_values("gain_vs_tissue", ascending=False)[cols].head(10).round(3))
print("Least predictable"); display(per[cols].tail(8).round(3))
"""),
    md("## By target pathway"),
    code("display(Image(FIG / 'by_pathway.png'))\npd.read_csv(T / 'by_pathway.csv', index_col=0).round(3)"),
]

# ----------------------------------------------------------------------------- 04
N04 = [
    md("""
# 04. Selected genes, stability and known biology

Three checks on what the sparse models select: (1) is the selection stable across the five
training folds, (2) do the models recover biomarkers established in the literature, and (3) do
predictions from expression alone separate mutation-defined sensitive groups the model never saw?
"""),
    SETUP,
    code("""
stab = pd.read_csv(T / "selection_stability.csv")
print("per drug, medians across drugs")
stab.groupby("model")[["median_selected", "n_all_folds", "n_3plus", "mean_jaccard"]].median().round(2)
"""),
    md("""
`n_all_folds`: genes selected in all five folds. `mean_jaccard`: average overlap of the selected
sets between pairs of folds (1 = identical). Each fold drops 20% of the lines, a mild perturbation,
so low overlap means the exact gene list is fragile even when accuracy is stable.

## Known biomarkers
"""),
    code("""
bio = pd.read_csv(T / "biomarker_checks.csv")
bio.round(3)
"""),
    md("""
* `spearman_gene_auc`: marginal correlation of the biomarker gene's expression with AUC (negative =
  higher expression, more sensitive).
* `rank_*` / `folds_*`: where the biomarker gene ranks in each model's selection list, and in how
  many folds it was selected.
* For mutation biomarkers (BRAF for dabrafenib and trametinib) the models never see mutation data.
  `roc_auc_pred_separates_mutants` asks whether the elastic net's predicted AUC still ranks mutant
  lines as more sensitive (0.5 = no separation, 1 = perfect).

## Top genes for the biomarker drugs
"""),
    code("""
top = pd.read_csv(T / "top_genes.csv", dtype={"drug": str})
drugs = data.load_drugs()
for name in bio.drug:
    d = str(bio.loc[bio.drug == name, "drug_id"].iloc[0])
    g = top[(top.drug == d) & (top.model == "Elastic net")].head(8)
    print(f"{name:<14}", ", ".join(f"{f} ({c:+.3f}, {k}/5)" for f, c, k in zip(g.feature, g.mean_coef, g.folds)))
"""),
]

# ----------------------------------------------------------------------------- 05
N05 = [
    md("""
# 05. Robustness: slower models, other targets, unseen cancer types, another assay

All on the **focus panel**: 36 drugs fixed by a rule before any model was evaluated (the 30 drugs
shared with GDSC1 whose response varies most, plus 6 drugs with well-known biomarkers;
`drugresponse.panel`).
"""),
    SETUP,
    code("""
panel = pd.read_csv(T / "panel_summary.csv", index_col=0)
panel.round(3)
"""),
    md("""
## Gene-filter size and response measure
The main analysis keeps the 2,000 most variable genes, a number fixed for compute, not tuned. Here
1,000 and 5,000 are compared, and ln(IC50) replaces AUC as the response.
"""),
    code("pd.read_csv(T / 'sensitivity.csv')"),
    md("""
## Leave one cancer type out
Train on every other tissue, predict the held-out one. Two things can fail: the **level** (an unseen
tissue's average sensitivity cannot be learned from a tissue indicator) and the **ranking** within it.
"""),
    code("display(Image(FIG / 'leave_tissue_out.png'))\npd.read_csv(T / 'leave_tissue_out.csv')"),
    md("""
## Cross-assay: GDSC2-trained models scored against GDSC1
Many drugs were screened twice by Sanger with different viability assays (GDSC1: Syto60 /
resazurin; GDSC2: CellTiter-Glo). A model trained on GDSC2 cannot be expected to agree with GDSC1
better than the two assays agree with each other, so their agreement (`r_assays`) is the ceiling.
"""),
    code("display(Image(FIG / 'cross_assay.png'))\npd.read_csv(T / 'cross_assay.csv')"),
    code("""
ca = pd.read_csv(T / "cross_assay_by_drug.csv")
ca[ca.model == "Ridge"].sort_values("r_assays", ascending=False)[["drug_name", "n", "r_assays", "r_gdsc2", "r_gdsc1"]].round(3)
"""),
]

NOTEBOOKS = {
    "01_data_and_batch_effect.ipynb": N01, "02_inside_one_drug.ipynb": N02,
    "03_results_across_drugs.ipynb": N03, "04_selection_and_biology.ipynb": N04,
    "05_robustness.ipynb": N05,
}

if __name__ == "__main__":
    only = sys.argv[1:]
    for fname, cells in NOTEBOOKS.items():
        if only and not any(o in fname for o in only):
            continue
        write_and_run(cells, NB / fname, timeout=3600)
