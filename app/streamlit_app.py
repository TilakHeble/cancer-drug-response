"""Cancer drug response dashboard.

Reads the precomputed tables in outputs/tables (no model fitting at runtime).
    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
T = ROOT / "outputs" / "tables"
MODELS = ["Tissue", "PCR", "Ridge", "Lasso", "Elastic net"]

st.set_page_config(page_title="Cancer drug response", layout="wide")


@st.cache_data
def load(name, **kw):
    f = T / name
    if not f.exists():
        return None
    return pd.read_parquet(f) if name.endswith(".parquet") else pd.read_csv(f, **kw)


summ = load("summary.csv", index_col=0)
met = load("drug_metrics.csv", dtype={"drug": str})
per = load("per_drug_ridge.csv", dtype={"drug": str})
pw = load("by_pathway.csv", index_col=0)
top = load("top_genes.csv", dtype={"drug": str})
preds = load("predictions.parquet")
stab = load("selection_stability.csv", dtype={"drug": str})
ca = load("cross_assay_by_drug.csv", dtype={"drug": str})
lot = load("leave_tissue_out.csv")
bio = load("biomarker_checks.csv")
sens = load("sensitivity.csv")
psumm = load("panel_summary.csv", index_col=0)

st.title("Predicting cancer drug response from gene expression")
st.caption(f"{len(per)} drugs x up to 944 cancer cell lines from the Genomics of Drug Sensitivity in "
           "Cancer (GDSC2, Wellcome Sanger Institute) and Cell Model Passports RNA-seq. Nested 5-fold "
           "cross-validation with the same tissue-stratified folds for every model.")

tabs = st.tabs(["Overview", "Explore a drug", "Pathways", "Robustness", "Biology checks"])

# ------------------------------------------------------------------------------ overview
with tabs[0]:
    c1, c2, c3 = st.columns(3)
    c1.metric("Median r, ridge", f"{summ.loc['Ridge', 'median_r']:.2f}")
    c2.metric("Median r, tissue only", f"{summ.loc['Tissue', 'median_r']:.2f}")
    c3.metric("Drugs where ridge beats tissue", f"{summ.loc['Ridge', 'share_beats_tissue']:.1%}")
    show = summ.drop(index="Mean", errors="ignore")[
        ["n_drugs", "median_r", "median_r2", "median_rho_within", "share_beats_tissue", "median_gain_vs_tissue"]]
    st.dataframe(show.rename(columns={"median_r": "median r", "median_r2": "median R2",
                                      "median_rho_within": "median within-tissue Spearman",
                                      "share_beats_tissue": "share of drugs beating tissue",
                                      "median_gain_vs_tissue": "median gain in r vs tissue"})
                 .round(3), width="stretch")
    st.caption("r: Pearson correlation of out-of-fold predicted and measured AUC across all lines. "
               "Within-tissue Spearman: how well a model ranks lines inside the same cancer type, "
               "the part tissue alone cannot explain.")
    w = met.pivot(index="drug", columns="model", values="r")
    fig = go.Figure()
    for m in MODELS:
        if m in w:
            fig.add_trace(go.Box(y=w[m], name=m, boxpoints="all", jitter=0.4, pointpos=0, marker_size=3))
    fig.update_layout(height=420, yaxis_title="r across drugs", showlegend=False,
                      margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, width="stretch")

# ------------------------------------------------------------------------------ explore
with tabs[1]:
    names = per.set_index("drug").drug_name
    label = {d: f"{n} ({d})" for d, n in names.items()}
    order = per.sort_values("r", ascending=False).drug.tolist()
    d = st.selectbox("Drug (sorted by ridge accuracy)", order, format_func=lambda k: label[k])
    info = per.set_index("drug").loc[d]
    st.markdown(f"**Target:** {info.target} · **Pathway:** {info.pathway} · **Cell lines:** {int(info.n)}")
    m = met[met.drug == d].set_index("model").loc[[x for x in MODELS if x in set(met[met.drug == d].model)]]
    st.dataframe(m[["r", "rho", "rho_within", "rmse", "r2"]].round(3).T, width="stretch")
    c1, c2 = st.columns([3, 2])
    mm = c1.radio("Model", ["Elastic net", "Ridge", "Tissue"], horizontal=True)
    p = preds[(preds.drug == d) & (preds.model == mm)]
    fig = px.scatter(p, x="pred", y="actual", color="tissue", hover_data=["line"], opacity=0.75,
                     labels={"pred": "predicted AUC", "actual": "measured AUC"})
    lo, hi = float(min(p.pred.min(), p.actual.min())), float(max(p.pred.max(), p.actual.max()))
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(color="black", dash="dash"),
                             showlegend=False))
    fig.update_layout(height=520, margin=dict(l=10, r=10, t=10, b=10), legend=dict(font_size=9))
    c1.plotly_chart(fig, width="stretch")
    c2.markdown("**Most consistently selected genes**")
    tm = c2.radio("Gene list from", ["Elastic net", "Lasso", "Ridge"], horizontal=True, key="tm")
    g = top[(top.drug == d) & (top.model == tm)][["feature", "folds", "mean_coef"]]
    c2.dataframe(g.rename(columns={"feature": "gene", "folds": "folds selected (of 5)",
                                   "mean_coef": "mean standardised coefficient"}).round(4),
                 width="stretch", hide_index=True)
    c2.caption("Negative coefficient: higher expression, lower AUC (more sensitive). Ridge keeps every "
               "gene; its list shows the 50 largest coefficients per fold.")

# ------------------------------------------------------------------------------ pathways
with tabs[2]:
    p = pw[pw.n_drugs >= 3].sort_values("median_r_ridge")
    fig = go.Figure()
    fig.add_trace(go.Bar(y=p.index, x=p.median_r_tissue, orientation="h", name="Tissue only"))
    fig.add_trace(go.Bar(y=p.index, x=p.median_r_ridge, orientation="h", name="Ridge"))
    fig.update_layout(barmode="group", height=650, xaxis_title="median r", margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, width="stretch")
    st.dataframe(pw.round(3), width="stretch")

# ------------------------------------------------------------------------------ robustness
with tabs[3]:
    if psumm is not None:
        st.subheader("All models on the 36-drug focus panel")
        st.dataframe(psumm.drop(index="Mean", errors="ignore")[
            ["n_drugs", "median_r", "median_rho_within", "share_beats_tissue"]].round(3), width="stretch")
    if ca is not None:
        st.subheader("Cross-assay: trained on GDSC2, scored against GDSC1")
        c = ca[ca.model == "Ridge"]
        fig = px.scatter(c, x="r_assays", y="r_gdsc1", hover_data=["drug_name"],
                         labels={"r_assays": "agreement between the two assays (r)",
                                 "r_gdsc1": "r of predictions vs GDSC1"})
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="black"),
                                 showlegend=False))
        fig.update_layout(height=450, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, width="stretch")
        st.dataframe(ca.groupby("model")[["r_gdsc2", "r_gdsc1", "r_assays"]].median().round(3), width="stretch")
    if lot is not None:
        st.subheader("Leave one cancer type out")
        st.dataframe(lot.round(3), width="stretch")
        st.caption("rho: median within-tissue Spearman across panel drugs. bias: mean predicted minus "
                   "measured AUC; an unseen tissue's average sensitivity cannot be learned.")
    if sens is not None:
        st.subheader("Gene-filter size and response measure")
        st.dataframe(sens.round(3), width="stretch")

# ------------------------------------------------------------------------------ biology
with tabs[4]:
    if bio is not None:
        st.subheader("Do the models recover known biomarkers?")
        st.dataframe(bio.round(3), width="stretch", hide_index=True)
        st.caption("rank_*: rank of the biomarker gene in each model's selection list (1 = most often "
                   "selected). The models see expression only, never mutations: "
                   "roc_auc_pred_separates_mutants measures whether their predictions still separate "
                   "mutant from wild-type lines.")
    if stab is not None:
        st.subheader("Selection stability across the five training folds")
        st.dataframe(stab.groupby("model")[["median_selected", "n_all_folds", "n_3plus", "mean_jaccard"]]
                     .median().round(2), width="stretch")
