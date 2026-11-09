"""Tables and figures for the docs, notebooks and dashboard.

    python scripts/make_reports.py
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from drugresponse import data  # noqa: E402
from drugresponse import evaluate as E  # noqa: E402
from drugresponse.config import FIGURES, OUTPUTS, TABLES  # noqa: E402
from drugresponse.panel import BIOMARKER_DRUGS, focus_panel  # noqa: E402

MAIN_ORDER = ["Mean", "Tissue", "PCR", "Ridge", "Lasso", "Elastic net"]
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.alpha": 0.3, "figure.dpi": 110})


def load(name):
    p = pd.read_parquet(OUTPUTS / f"preds_{name}.parquet")
    with open(OUTPUTS / f"coefs_{name}.pkl", "rb") as f:
        c = pickle.load(f)
    return p, c


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    models = data.load_models()
    tissue = models.tissue
    drugs = data.load_drugs()
    preds, coefs = load("main")

    # ------------------------------------------------------------------ accuracy
    met = E.drug_metrics(preds, tissue)
    met = met.merge(drugs[["drug_name", "target", "pathway"]], left_on="drug", right_index=True)
    met.to_csv(TABLES / "drug_metrics.csv", index=False)
    summ = E.summary(met).loc[[m for m in MAIN_ORDER if m in set(met.model)]]
    summ.to_csv(TABLES / "summary.csv")
    E.pairwise_wins(met, [m for m in MAIN_ORDER if m != "Mean"]).to_csv(TABLES / "pairwise_wins.csv")

    w = met.pivot(index="drug", columns="model", values="r")
    best = w.drop(columns=["Mean"]).idxmax(axis=1).value_counts()
    best.rename("n_drugs_best").to_csv(TABLES / "best_model_counts.csv")

    gain = met[met.model == "Ridge"].set_index("drug")[["drug_name", "target", "pathway", "n", "r", "rho_within"]]
    gain["r_tissue"] = w["Tissue"]
    gain["r_lasso"] = w["Lasso"]
    gain["r_elastic_net"] = w["Elastic net"]
    gain["gain_vs_tissue"] = gain.r - gain.r_tissue
    gain.sort_values("r", ascending=False).to_csv(TABLES / "per_drug_ridge.csv")
    pw = (gain.groupby("pathway")
          .agg(n_drugs=("r", "size"), median_r_ridge=("r", "median"), median_r_tissue=("r_tissue", "median"),
               median_gain=("gain_vs_tissue", "median"), median_rho_within=("rho_within", "median"))
          .sort_values("median_r_ridge", ascending=False))
    pw.to_csv(TABLES / "by_pathway.csv")

    # ------------------------------------------------------------------ selection
    stab = pd.concat([E.selection_stability(coefs, m) for m in ("Lasso", "Elastic net")])
    stab.to_csv(TABLES / "selection_stability.csv", index=False)
    nn = pd.DataFrame([dict(drug=c["drug"], model=c["model"], n_nonzero=c["n_nonzero"],
                            l1_ratio=c["l1_ratio"], k=c["k"]) for c in coefs])
    nn.to_csv(TABLES / "model_complexity.csv", index=False)

    top = []
    for d in w.index:
        for m in ("Elastic net", "Lasso", "Ridge"):
            g = E.gene_frequency(coefs, m, d)
            g = g[~g.feature.str.startswith("tissue:")].head(15)
            g.insert(0, "model", m)
            g.insert(0, "drug", d)
            top.append(g)
    pd.concat(top).to_csv(TABLES / "top_genes.csv", index=False)

    # dashboard predictions (three models, every drug)
    keep = preds[preds.model.isin(["Tissue", "Ridge", "Elastic net"])].copy()
    keep["tissue"] = tissue.reindex(keep.line).values
    keep["pred"] = keep.pred.astype("float32")
    keep["actual"] = keep.actual.astype("float32")
    keep[["drug", "line", "model", "pred", "actual", "tissue"]].to_parquet(TABLES / "predictions.parquet")

    # ------------------------------------------------------------------ independent lab (PRISM)
    from drugresponse import prism
    pr = prism.build()
    if pr is not None:
        sc = prism.score(preds[preds.model.isin(["Tissue", "Ridge", "Lasso", "Elastic net"])], pr)
        sc = sc.merge(drugs[["drug_name", "target"]], left_on="drug", right_index=True)
        sc.to_csv(TABLES / "prism_by_drug.csv", index=False)
        sc.groupby("model")[["n", "r_gdsc2", "r_prism", "r_assays"]].median().round(3).assign(
            n_drugs=sc.groupby("model").size()).to_csv(TABLES / "prism.csv")

    # ------------------------------------------------------------------ panel experiments
    if (OUTPUTS / "preds_panel.parquet").exists():
        panel_reports(tissue, drugs, met)

    figures(met, gain, pw)
    print("reports written to", TABLES)


def panel_reports(tissue, drugs, met_main):
    pp, pc = load("panel")
    panel = focus_panel()
    names = drugs.drug_name

    # slower models on the panel, compared with the main models on the same drugs
    m_slow = E.drug_metrics(pp[pp.design == "main"], tissue)
    m_all = pd.concat([met_main[met_main.drug.isin(panel.index)].drop(columns=["drug_name", "target", "pathway"]),
                       m_slow])
    E.summary(m_all).to_csv(TABLES / "panel_summary.csv")
    m_all.to_csv(TABLES / "panel_drug_metrics.csv", index=False)
    pd.concat([E.selection_stability(pc, m) for m in ("Adaptive lasso", "Group lasso")]).to_csv(
        TABLES / "panel_selection_stability.csv", index=False)

    # gene-filter sensitivity and IC50 target
    rows = []
    for des in ("genes_1000", "genes_5000", "target_ln_ic50"):
        d = pp[pp.design == des]
        if len(d):
            m = E.drug_metrics(d, tissue)
            m["design"] = des
            rows.append(m)
    base = met_main[met_main.drug.isin(panel.index) & met_main.model.isin(["Tissue", "Ridge", "Lasso"])].copy()
    base["design"] = "genes_2000 (main)"
    sens = pd.concat(rows + [base.drop(columns=["drug_name", "target", "pathway"])])
    sens.to_csv(TABLES / "sensitivity_by_drug.csv", index=False)
    sens.groupby(["design", "model"])[["r", "rho_within"]].median().round(3).to_csv(TABLES / "sensitivity.csv")

    # leave one cancer type out: within-tissue rho when the tissue was unseen vs seen
    lo = pp[pp.design.str.startswith("leave_out:")].copy()
    rows = []
    main_preds, _ = load("main")
    for t in lo.design.str.replace("leave_out:", "").unique():
        d_un = lo[lo.design == f"leave_out:{t}"]
        lines_t = set(tissue[tissue == t].index)
        d_seen = main_preds[main_preds.drug.isin(panel.index) & main_preds.line.isin(lines_t)
                            & main_preds.model.isin(["Ridge", "Lasso"])]
        for (dr, m), g in d_un.groupby(["drug", "model"]):
            gs = d_seen[(d_seen.drug == dr) & (d_seen.model == m)]
            rows.append(dict(tissue=t, drug=dr, model=m, n=len(g),
                             rho_unseen=E._rho(g.pred, g.actual), rho_seen=E._rho(gs.pred, gs.actual),
                             bias_unseen=float((g.pred - g.actual).mean()),
                             bias_seen=float((gs.pred - gs.actual).mean())))
    lot = pd.DataFrame(rows)
    lot.to_csv(TABLES / "leave_tissue_out_by_drug.csv", index=False)
    lot.groupby(["tissue", "model"])[["rho_seen", "rho_unseen", "bias_seen", "bias_unseen"]].median().round(3).to_csv(
        TABLES / "leave_tissue_out.csv")

    # cross-assay: predictions from GDSC2-trained models vs GDSC1 measurements of the same lines
    cx = pp[pp.design == "cross_assay"].dropna(subset=["actual_gdsc1"])
    rows = []
    for (dr, m), g in cx.groupby(["drug", "model"]):
        rows.append(dict(drug=dr, drug_name=names.get(dr), model=m, n=len(g),
                         r_gdsc2=E._r(g.pred, g.actual), r_gdsc1=E._r(g.pred, g.actual_gdsc1),
                         r_assays=E._r(g.actual, g.actual_gdsc1)))
    ca = pd.DataFrame(rows)
    ca.to_csv(TABLES / "cross_assay_by_drug.csv", index=False)
    ca.groupby("model")[["r_gdsc2", "r_gdsc1", "r_assays"]].median().round(3).to_csv(TABLES / "cross_assay.csv")

    # biomarker sanity checks
    biomarker_checks(pc, tissue, drugs)


def biomarker_checks(panel_coefs, tissue, drugs):
    _, coefs = load("main")
    expr = data.load_expression()
    mut = data.load_mutations()
    R = data.load_response()
    main_preds, _ = load("main")
    from drugresponse.experiment import Store
    S = Store()
    in_design = {}
    for k in range(5):
        des, _ = S.design(k, S.fold.values != k)
        for g in S.genes[des.genes_]:
            in_design[g] = in_design.get(g, 0) + 1
    rows = []
    for name, (gene, kind, why) in BIOMARKER_DRUGS.items():
        ids = drugs.index[drugs.drug_name == name]
        if not len(ids):
            continue
        d = max(ids, key=lambda i: drugs.loc[i, "n_lines"])
        if d not in R.columns:
            continue
        y = R[d].dropna()
        rec = dict(drug=name, drug_id=d, biomarker=gene, kind=kind, rationale=why, n=len(y),
                   folds_gene_in_features=in_design.get(gene, 0) if kind == "expression" else np.nan)
        if kind == "expression" and gene in expr.columns:
            x = expr.loc[y.index, gene]
            rec["spearman_gene_auc"] = stats.spearmanr(x, y).statistic
        if kind == "mutation" and gene in mut.columns:
            mm = mut.reindex(y.index)[gene].fillna(0)
            rec["n_mutant"] = int(mm.sum())
            rec["auc_mutant"] = float(y[mm == 1].median())
            rec["auc_wildtype"] = float(y[mm == 0].median())
            p = main_preds[(main_preds.drug == d) & (main_preds.model == "Elastic net")].set_index("line").pred
            lab = mm.reindex(p.index)
            if lab.sum() > 5:
                from sklearn.metrics import roc_auc_score
                # model never sees mutations: does its predicted AUC separate mutants anyway?
                rec["roc_auc_pred_separates_mutants"] = float(roc_auc_score(lab, -p))
        for m in ("Elastic net", "Lasso", "Ridge"):
            g = E.gene_frequency(coefs, m, d)
            g = g[~g.feature.str.startswith("tissue:")].reset_index(drop=True)
            hit = g.index[g.feature == gene]
            rec[f"rank_{m}"] = int(hit[0]) + 1 if len(hit) else np.nan
            rec[f"folds_{m}"] = int(g.loc[hit[0], "folds"]) if len(hit) else 0
        rows.append(rec)
    pd.DataFrame(rows).to_csv(TABLES / "biomarker_checks.csv", index=False)


def figures(met, gain, pw):
    # 1. distribution of r by model
    w = met.pivot(index="drug", columns="model", values="r")
    order = [m for m in MAIN_ORDER if m in w.columns and m != "Mean"]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.boxplot([w[m].dropna() for m in order], tick_labels=order, showfliers=False)
    for i, m in enumerate(order, 1):
        ax.scatter(np.random.default_rng(i).normal(i, 0.06, w[m].notna().sum()), w[m].dropna(),
                   s=6, alpha=0.35)
    ax.set_ylabel("Pearson r, predicted vs measured AUC")
    ax.set_title(f"Out-of-fold accuracy across {len(w)} drugs (nested 5-fold CV)")
    fig.tight_layout()
    fig.savefig(FIGURES / "accuracy_by_model.png")
    plt.close(fig)

    # 2. ridge vs tissue
    fig, ax = plt.subplots(figsize=(5.8, 5.4))
    ax.scatter(gain.r_tissue, gain.r, s=12, alpha=0.6)
    lim = [min(gain.r_tissue.min(), gain.r.min()) - 0.05, max(gain.r_tissue.max(), gain.r.max()) + 0.05]
    ax.plot(lim, lim, "k--", lw=1)
    for _, rr in gain.sort_values("gain_vs_tissue", ascending=False).head(6).iterrows():
        ax.annotate(rr.drug_name, (rr.r_tissue, rr.r), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("r, tissue-only model")
    ax.set_ylabel("r, ridge on expression + tissue")
    ax.set_title("Where gene expression adds to cancer type")
    fig.tight_layout()
    fig.savefig(FIGURES / "ridge_vs_tissue.png")
    plt.close(fig)

    # 3. by pathway
    p = pw[pw.n_drugs >= 4].sort_values("median_r_ridge")
    fig, ax = plt.subplots(figsize=(8, 6))
    y = np.arange(len(p))
    ax.barh(y - 0.2, p.median_r_tissue, 0.4, label="Tissue only")
    ax.barh(y + 0.2, p.median_r_ridge, 0.4, label="Ridge (expression + tissue)")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{i} ({n})" for i, n in zip(p.index, p.n_drugs)], fontsize=8)
    ax.set_xlabel("median r across drugs")
    ax.set_title("Predictability by drug target pathway (pathways with 4+ drugs)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "by_pathway.png")
    plt.close(fig)

    # 4. cross-assay
    f = TABLES / "cross_assay_by_drug.csv"
    if f.exists():
        ca = pd.read_csv(f)
        c = ca[ca.model == "Ridge"]
        fig, ax = plt.subplots(figsize=(5.8, 5.4))
        ax.scatter(c.r_assays, c.r_gdsc1, s=16, label="vs GDSC1 (other assay)")
        ax.scatter(c.r_assays, c.r_gdsc2, s=16, marker="x", label="vs GDSC2 (training assay)")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlabel("r between GDSC1 and GDSC2 measurements (replicate agreement)")
        ax.set_ylabel("r of ridge predictions")
        ax.set_title("Prediction accuracy vs agreement between assays")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(FIGURES / "cross_assay.png")
        plt.close(fig)

    # 4b. PRISM
    f = TABLES / "prism_by_drug.csv"
    if f.exists():
        sc = pd.read_csv(f)
        c = sc[sc.model == "Ridge"]
        fig, ax = plt.subplots(figsize=(5.8, 5.4))
        ax.scatter(c.r_assays, c.r_prism, s=14, alpha=0.7)
        ax.plot([-0.2, 1], [-0.2, 1], "k--", lw=1)
        for _, rr in c.sort_values("r_prism", ascending=False).head(6).iterrows():
            ax.annotate(rr.drug_name, (rr.r_assays, rr.r_prism), fontsize=7, xytext=(3, 3), textcoords="offset points")
        ax.set_xlabel("r between GDSC2 and PRISM measurements (between-lab agreement)")
        ax.set_ylabel("r of GDSC2-trained ridge predictions vs PRISM")
        ax.set_title(f"Independent lab: {len(c)} drugs in both screens")
        fig.tight_layout()
        fig.savefig(FIGURES / "prism.png")
        plt.close(fig)

    # 5. leave tissue out
    f = TABLES / "leave_tissue_out_by_drug.csv"
    if f.exists():
        lo = pd.read_csv(f)
        g = lo[lo.model == "Ridge"].groupby("tissue")[["rho_seen", "rho_unseen"]].median().sort_values("rho_seen")
        fig, ax = plt.subplots(figsize=(7.5, 4))
        y = np.arange(len(g))
        ax.barh(y - 0.2, g.rho_seen, 0.4, label="tissue in training data")
        ax.barh(y + 0.2, g.rho_unseen, 0.4, label="tissue held out")
        ax.set_yticks(y)
        ax.set_yticklabels(g.index, fontsize=8)
        ax.set_xlabel("median within-tissue Spearman (ridge, focus panel)")
        ax.set_title("Predicting a cancer type the model has never seen")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(FIGURES / "leave_tissue_out.png")
        plt.close(fig)


if __name__ == "__main__":
    main()
