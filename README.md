# Cancer drug response: regularised regression when genes outnumber samples

[![tests](https://github.com/TilakHeble/cancer-drug-response/actions/workflows/tests.yml/badge.svg)](https://github.com/TilakHeble/cancer-drug-response/actions/workflows/tests.yml)

**Live dashboard:** https://cancer-drug-response.streamlit.app/

Can a cancer cell line's gene expression predict how strongly it responds to a drug, beyond what its cancer type already tells us? This project fits ridge, lasso, elastic net, adaptive lasso, group lasso and principal components regression for **281 anti-cancer drugs** across up to **944 cell lines** and **18,407 genes**. It uses data from the Wellcome Sanger Institute's Genomics of Drug Sensitivity in Cancer (GDSC) and Cell Model Passports. Every model is compared with a model that knows only the cancer type, using nested cross-validation.

## Key results

| Model | Median r (281 drugs) | Beats cancer type alone | Median within-cancer-type Spearman |
|---|---|---|---|
| Cancer type only | 0.261 | | n/a |
| PCR | 0.364 | 89% | 0.156 |
| **Ridge** | **0.402** | **97.5%** | **0.194** |
| Lasso | 0.382 | 88% | 0.151 |
| Elastic net | 0.382 | 88% | 0.155 |

*Out-of-fold Pearson correlation between predicted and measured AUC of the dose-response curve. [Full table](outputs/tables/summary.csv).*

* **Expression adds to cancer type for 274 of 281 drugs.** For Nutlin-3a, r rises from 0.38 to 0.67.
* **Dense beats sparse.** Ridge beats the lasso and elastic net on 85% of drugs. The response is spread across many co-expressed genes.
* **Known biology recovered.** ERBB2 (lapatinib), BCL2 (venetoclax), SLFN11 (olaparib) and EGFR (erlotinib) are each the #1 selected gene. The TP53 targets CDKN1A, PHLDA3 and TRIM22 lead for Nutlin-3a. Expression-only predictions separate BRAF-mutant lines for dabrafenib with ROC AUC 0.82.
* **Measurement noise is the ceiling.** For the same drug and cell line, Sanger's two assays agree at r = 0.53, and Sanger and the Broad Institute's PRISM screen only at 0.24. Scored against those other measurements, the models reach 71% and 77% of these ceilings.
* **Unseen cancer types are a blind spot.** Holding a tissue out of training roughly halves within-tissue accuracy for blood cancers and skin cancers.

Read the [executive summary](docs/executive_summary.md), the [model development document](docs/model_development_document.md) and the [validation report](docs/validation_report.md).

## What makes it different

* **Honest p >> n evaluation.** Nested cross-validation. Batch correction, gene filtering and scaling are fitted inside every training fold, and corrupting test responses changes no prediction (validated). [Learn notebook 05](learn/05_nested_cv_and_leakage.ipynb) shows how leaky gene selection manufactures accuracy from pure noise.
* **A batch effect found and fixed.** The two RNA-seq pipelines are almost perfectly distinguishable from expression (AUC 0.998). The correction removes source effects while protecting tissue differences, because the two pipelines sampled tissues unevenly.
* **A tissue baseline and within-tissue metrics,** so a model cannot score by recognising blood cancers.
* **Robustness on a 36-drug panel fixed in advance:**
  * group lasso over co-expression modules (from-scratch FISTA, checked against the KKT conditions) and the adaptive lasso;
  * 1,000, 2,000 or 5,000 genes, and ln(IC50) instead of AUC as the response;
  * leave-one-cancer-type-out;
  * cross-assay transfer (GDSC2 → GDSC1);
  * an independent lab: 107 drugs scored against the Broad Institute's PRISM screen;
  * eight biomarker sanity checks.

## Repository

```
src/drugresponse/   data, fold design (batch correction), models, experiments, evaluation, PRISM scoring
scripts/            pipeline, experiment runner, reports, validation, notebook builders
notebooks/          01-05 analysis: data and batch effect -> robustness
learn/              01-07 regularisation when p >> n, on this data
app/                Streamlit dashboard (reads outputs/tables, no fitting at runtime)
docs/               executive summary, model development document, validation report
outputs/tables      every result table; outputs/figures
tests/              leakage, batch correction, solver and evaluation unit tests (synthetic data)
```

## Learn: regularisation when p >> n

[`learn/`](learn/README.md) covers what changes when features vastly outnumber samples:
* infinitely many perfect fits;
* the kernel form of ridge and its link to PCR;
* why lasso gene lists are unstable, and stability selection;
* elastic net and group lasso on gene modules;
* nested CV and the classic genomics leak;
* batch effects and confounding;
* reading coefficients against known biology.

It builds on the ridge/lasso foundations in the [GDP nowcasting project](https://github.com/TilakHeble/uk-gdp-nowcast/tree/main/learn).

## Reproduce

```bash
pip install -r requirements.txt
python scripts/run_pipeline.py --data          # download Sanger data (~1.3 GB) and build data/processed
python scripts/run_pipeline.py --experiments   # 281 drugs + 36-drug panel (~3 h on 2 cores)
python scripts/run_pipeline.py --reports       # tables, figures and validation checks
python scripts/run_pipeline.py --notebooks     # execute notebooks/ and learn/
pytest
streamlit run app/streamlit_app.py
```

The processed expression matrix (76 MB) is not committed; the dashboard and documents need only `outputs/tables/`.

## Data

Genomics of Drug Sensitivity in Cancer, release 8.5 (GDSC1 and GDSC2 fitted dose-response data) and Cell Model Passports (RNA-seq, model annotation, driver mutations), Wellcome Sanger Institute. Please cite Iorio et al. (2016), *Cell*, and van der Meer et al. (2019), *Nucleic Acids Research*, when using these data. Independent-lab test: PRISM Repurposing secondary screen (Corsello et al. 2020, *Nature Cancer*), Broad Institute, via figshare.

## Related projects

* [UK credit stress engine](https://github.com/TilakHeble/uk-credit-stress-engine): linear regression for bank credit losses under the Bank of England stress scenario.
* [NHS readmission risk](https://github.com/TilakHeble/nhs-readmission-risk): logistic regression for 30-day emergency readmissions.
* [UK GDP nowcast](https://github.com/TilakHeble/uk-gdp-nowcast): regularised regression in real time on a short macroeconomic panel.
