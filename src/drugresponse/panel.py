"""The focus panel: drugs that get the slower models and every robustness design.

Chosen by a rule fixed before any model was evaluated (it uses the response data, never
model performance):
  1. drugs screened in both GDSC2 and GDSC1 (same name), with >= 300 lines in each, so the
     cross-assay test is possible;
  2. the 30 of those whose GDSC2 AUC varies most across cell lines (a drug that kills
     everything or nothing has little to predict);
  3. plus drugs with a well-established expression or mutation biomarker, used as
     biological sanity checks (BIOMARKER_DRUGS), if not already included.
"""
from __future__ import annotations

import pandas as pd

from . import data

# drug name -> (biomarker gene, expected direction of association with AUC, rationale)
BIOMARKER_DRUGS = {
    "Dabrafenib": ("BRAF", "mutation", "BRAF V600 inhibitor; BRAF-mutant melanoma lines are sensitive"),
    "Nutlin-3a (-)": ("MDM2", "expression", "MDM2 inhibitor; needs wild-type TP53, whose targets (MDM2, CDKN1A) are expressed"),
    "Lapatinib": ("ERBB2", "expression", "EGFR/HER2 inhibitor; HER2-amplified lines are sensitive"),
    "Venetoclax": ("BCL2", "expression", "BCL-2 inhibitor; high BCL2 expression predicts sensitivity"),
    "Afatinib": ("EGFR", "expression", "EGFR/HER2 inhibitor; EGFR-dependent lines are sensitive"),
    "Trametinib": ("BRAF", "mutation", "MEK inhibitor; MAPK-pathway mutant lines are sensitive"),
    "Olaparib": ("SLFN11", "expression", "PARP inhibitor; SLFN11 expression predicts DNA-damage drug sensitivity"),
    "Erlotinib": ("EGFR", "expression", "EGFR inhibitor"),
}

N_VARIABLE = 30
MIN_LINES = 300
HOLDOUT_TISSUES = ["Lung", "Haematopoietic and Lymphoid", "Skin", "Central Nervous System",
                   "Breast", "Large Intestine"]


def focus_panel(S=None) -> pd.DataFrame:
    d2 = data.load_drugs("gdsc2")
    d1 = data.load_drugs("gdsc1")
    d1 = d1[d1.n_lines >= MIN_LINES].copy()
    d1["key"] = d1.drug_name.str.lower()
    best1 = d1.sort_values("n_lines", ascending=False).drop_duplicates("key")
    id_by_name = pd.Series(best1.index.astype(str), index=best1.key)
    d2 = d2[d2.n_lines >= MIN_LINES].copy()
    d2 = d2.sort_values("n_lines", ascending=False)
    d2 = d2[~d2.drug_name.duplicated()]            # one GDSC2 entry per drug name (most lines)
    d2["gdsc1_id"] = d2.drug_name.str.lower().map(id_by_name)
    shared = d2[d2.gdsc1_id.notna()]
    top = shared.sort_values("auc_sd", ascending=False).head(N_VARIABLE)
    bio = d2[d2.drug_name.isin(BIOMARKER_DRUGS)]
    bio = bio.sort_values("n_lines", ascending=False).drop_duplicates("drug_name")
    panel = pd.concat([top, bio[~bio.index.isin(top.index)]])
    panel["reason"] = ["most variable response" if i in top.index else "biomarker check"
                       for i in panel.index]
    panel["biomarker"] = panel.drug_name.map(lambda n: BIOMARKER_DRUGS.get(n, (None,))[0])
    return panel


def holdout_tissues(S=None) -> list[str]:
    return HOLDOUT_TISSUES
