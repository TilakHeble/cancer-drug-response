"""Paths, data sources and experiment settings."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"
TABLES = OUTPUTS / "tables"
FIGURES = OUTPUTS / "figures"

COG = "https://cog.sanger.ac.uk"
SANGER_FILES = {
    # Genomics of Drug Sensitivity in Cancer, release 8.5 (fitted dose-response curves)
    "gdsc2": f"{COG}/cancerrxgene/GDSC_release8.5/GDSC2_fitted_dose_response_27Oct23.xlsx",
    "gdsc1": f"{COG}/cancerrxgene/GDSC_release8.5/GDSC1_fitted_dose_response_27Oct23.xlsx",
    "compounds": f"{COG}/cancerrxgene/GDSC_release8.5/screened_compounds_rel_8.5.csv",
    # Cell Model Passports
    "models": f"{COG}/cmp/download/model_list_latest.csv.gz",
    "rnaseq": f"{COG}/cmp/download/rnaseq_latest.csv.gz",
    "mutations": f"{COG}/cmp/download/mutations_summary_latest.csv.gz",
}

# Response variable: area under the fitted dose-response curve over the screened
# concentration range (0-1; lower = more sensitive). LN_IC50 is a robustness check:
# it is often extrapolated beyond the highest tested concentration.
TARGET = "AUC"

# Feature construction
MIN_LOG_TPM = 1.0          # gene kept if log2(TPM+1) >= this in at least MIN_FRAC_EXPRESSED of lines
MIN_FRAC_EXPRESSED = 0.10
N_TOP_VARIABLE = 2000      # most variable genes (chosen inside each training fold); 1000 and
                           # 5000 are run as a sensitivity check (fixed for compute, not tuned)
N_MODULES = 100            # co-expression modules used as groups by the group lasso

# Evaluation
N_OUTER = 5
N_INNER = 5
SEED = 20261108
MIN_LINES_PER_DRUG = 300   # drugs screened in fewer GDSC2 cell lines are excluded
