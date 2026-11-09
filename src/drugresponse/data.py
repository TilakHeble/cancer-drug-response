"""Download and process the raw data into analysis-ready matrices.

Outputs in data/processed/:
  expression.parquet     cell lines x genes, log2(TPM + 1), float32 (GDSC-screened lines only)
  response_gdsc2.parquet cell lines x drugs, AUC (GDSC2: CellTiter-Glo assay)
  response_gdsc1.parquet cell lines x drugs, AUC (GDSC1: Syto60 / resazurin assays)
  ic50_gdsc2.parquet     cell lines x drugs, ln(IC50 / uM)
  drugs.csv              GDSC2 drug annotation (name, target, pathway, n lines)
  models.csv             cell-line annotation (tissue, cancer type, MSI, ...)
  mutations.parquet      cell lines x genes, 1 if a cancer-driver mutation is present
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import requests

from .config import MIN_FRAC_EXPRESSED, MIN_LOG_TPM, PROCESSED, RAW, SANGER_FILES

SANGER = RAW / "sanger"


def _download(url, dest, retries=4):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    for k in range(retries):
        try:
            with requests.get(url, stream=True, timeout=600,
                              headers={"User-Agent": "cancer-drug-response research project"}) as r:
                r.raise_for_status()
                tmp = dest.with_suffix(dest.suffix + ".part")
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
                tmp.rename(dest)
            return dest
        except requests.RequestException:
            if k == retries - 1:
                raise
            time.sleep(5 * (k + 1))


def download_all():
    for url in SANGER_FILES.values():
        _download(url, SANGER / url.rsplit("/", 1)[1])


def _path(key):
    return SANGER / SANGER_FILES[key].rsplit("/", 1)[1]


# --------------------------------------------------------------------------- response
def read_gdsc(key) -> pd.DataFrame:
    p = _path(key)
    cache = p.with_suffix(".parquet")
    if cache.exists():
        return pd.read_parquet(cache)
    d = pd.read_excel(p)
    d.to_parquet(cache)
    return d


def response_matrix(d: pd.DataFrame, value="AUC") -> pd.DataFrame:
    # a few drugs were screened twice under the same DRUG_ID in one release: average them
    w = d.pivot_table(index="SANGER_MODEL_ID", columns="DRUG_ID", values=value, aggfunc="mean")
    w.columns = w.columns.astype(str)
    return w


def drug_table(d: pd.DataFrame) -> pd.DataFrame:
    t = (d.groupby("DRUG_ID")
         .agg(drug_name=("DRUG_NAME", "first"), target=("PUTATIVE_TARGET", "first"),
              pathway=("PATHWAY_NAME", "first"), n_lines=("SANGER_MODEL_ID", "nunique"),
              min_conc=("MIN_CONC", "min"), max_conc=("MAX_CONC", "max"),
              auc_mean=("AUC", "mean"), auc_sd=("AUC", "std")))
    t.index = t.index.astype(str)
    return t


# --------------------------------------------------------------------------- expression
def build_expression(models: set[str]) -> pd.DataFrame:
    """Long RNA-seq table -> wide log2(TPM+1), one profile per cell line.

    The Cell Model Passports `rsem_tpm` column is already on the log2(TPM + 1) scale
    (GAPDH ~ 11-12, maximum ~ 18) for both sources, so no further transform is applied.

    Several lines were sequenced by both Sanger and the Broad Institute. Sanger profiles
    are preferred (same institute as the drug screen); Broad profiles fill the gaps.
    """
    # Memory-lean: the file has ~1e8 rows. Keep only screened lines, encode strings as
    # integer codes, and accumulate into a dense (source x line x gene) array.
    cols = ["model_id", "gene_symbol", "rsem_tpm", "data_source"]
    line_ix = {m: i for i, m in enumerate(sorted(models))}
    gene_ix: dict[str, int] = {}
    rows = {"Sanger": [], "Broad": []}
    for chunk in pd.read_csv(_path("rnaseq"), usecols=cols, chunksize=4_000_000,
                             dtype={"rsem_tpm": "float32", "model_id": "category",
                                    "gene_symbol": "category", "data_source": "category"}):
        chunk = chunk[chunk.model_id.isin(line_ix.keys()) & chunk.rsem_tpm.notna()
                      & chunk.gene_symbol.notna()]
        for g in chunk.gene_symbol.cat.categories:
            if g not in gene_ix:
                gene_ix[g] = len(gene_ix)
        li = chunk.model_id.astype(str).map(line_ix).to_numpy(np.int32)
        gi = chunk.gene_symbol.astype(str).map(gene_ix).to_numpy(np.int32)
        for s in rows:
            m = (chunk.data_source == s).to_numpy()
            rows[s].append((li[m], gi[m], chunk.rsem_tpm.to_numpy()[m]))
    n_l, n_g = len(line_ix), len(gene_ix)
    mats, seen = {}, {}
    for s, parts in rows.items():
        tot = np.zeros((n_l, n_g), np.float32)
        cnt = np.zeros((n_l, n_g), np.uint8)
        for li, gi, v in parts:
            np.add.at(tot, (li, gi), v)
            np.add.at(cnt, (li, gi), 1)
        mats[s] = np.divide(tot, cnt, out=np.zeros_like(tot), where=cnt > 0)
        seen[s] = cnt.sum(axis=1) > 0
    use_sanger = seen["Sanger"]
    logtpm = np.where(use_sanger[:, None], mats["Sanger"], mats["Broad"])
    has = use_sanger | seen["Broad"]
    lines = np.array(sorted(line_ix, key=line_ix.get))
    genes = np.array(sorted(gene_ix, key=gene_ix.get))
    logt = pd.DataFrame(logtpm[has].astype("float32"), index=lines[has], columns=genes)
    src = pd.Series(np.where(use_sanger, "Sanger", "Broad")[has], index=lines[has])
    keep = (logt >= MIN_LOG_TPM).mean() >= MIN_FRAC_EXPRESSED
    return logt.loc[:, keep].sort_index(axis=1), src


# --------------------------------------------------------------------------- mutations
def build_mutations(models: set[str]) -> pd.DataFrame:
    m = pd.read_csv(_path("mutations"))
    m = m[m.model_id.isin(models) & (m.cancer_driver.astype(str).str.lower().isin(["t", "true"]))]
    w = pd.crosstab(m.model_id, m.gene_symbol).clip(upper=1).astype("int8")
    return w


# --------------------------------------------------------------------------- pipeline
def process_all():
    PROCESSED.mkdir(parents=True, exist_ok=True)
    g2, g1 = read_gdsc("gdsc2"), read_gdsc("gdsc1")
    models = pd.read_csv(_path("models"))
    screened = set(g2.SANGER_MODEL_ID) | set(g1.SANGER_MODEL_ID)

    expr, src = build_expression(screened)
    expr.to_parquet(PROCESSED / "expression.parquet")

    lines = expr.index
    response_matrix(g2).reindex(lines).dropna(how="all").to_parquet(PROCESSED / "response_gdsc2.parquet")
    response_matrix(g1).reindex(lines).dropna(how="all").to_parquet(PROCESSED / "response_gdsc1.parquet")
    response_matrix(g2, "LN_IC50").reindex(lines).dropna(how="all").to_parquet(PROCESSED / "ic50_gdsc2.parquet")
    drug_table(g2).to_csv(PROCESSED / "drugs.csv")
    drug_table(g1).to_csv(PROCESSED / "drugs_gdsc1.csv")

    keep = ["model_id", "model_name", "tissue", "cancer_type", "cancer_type_detail", "msi_status",
            "growth_properties", "ploidy", "mutational_burden"]
    mod = models[models.model_id.isin(lines)][keep].set_index("model_id").reindex(lines)
    mod["expression_source"] = src.reindex(lines)
    mod.to_csv(PROCESSED / "models.csv")

    mut = build_mutations(set(lines)).reindex(lines)
    mut.to_parquet(PROCESSED / "mutations.parquet")
    print(f"expression {expr.shape}, GDSC2 drugs {g2.DRUG_ID.nunique()}, "
          f"mutation data for {mut.notna().all(axis=1).sum()} lines")


# --------------------------------------------------------------------------- loaders
def load_expression() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / "expression.parquet")


def load_response(dataset="gdsc2", value="AUC") -> pd.DataFrame:
    name = {"AUC": "response", "LN_IC50": "ic50"}[value]
    return pd.read_parquet(PROCESSED / f"{name}_{dataset}.parquet")


def load_models() -> pd.DataFrame:
    return pd.read_csv(PROCESSED / "models.csv", index_col=0)


def load_drugs(dataset="gdsc2") -> pd.DataFrame:
    f = "drugs.csv" if dataset == "gdsc2" else "drugs_gdsc1.csv"
    return pd.read_csv(PROCESSED / f, index_col=0, dtype={"DRUG_ID": str})


def load_mutations() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / "mutations.parquet")
