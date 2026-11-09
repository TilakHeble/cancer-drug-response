# Validation report: cancer drug response models

*Independent review of the [model development document](model_development_document.md). Checks can be rerun with `python scripts/validate.py` and `pytest`.*

## Summary

| Area | Rating | Finding |
|---|---|---|
| Data processing | 🟢 Green | Processed AUC identical to the Sanger release file; expression scale verified; batch effect found and corrected |
| Leakage and evaluation design | 🟢 Green | Corrupting test responses changes no prediction; all preprocessing inside folds; nested CV |
| Implementation | 🟢 Green | Ridge equals scikit-learn to 1e−16; from-scratch group lasso satisfies the KKT conditions on real data (production iteration cap leaves its gene counts overstated) |
| Accuracy on known cancer types | 🟢 Green | Ridge beats the tissue baseline on 97.5% of 281 drugs; known biomarkers recovered |
| Transfer to another assay and another lab | 🟡 Amber | 71% of the between-assay ceiling (GDSC1) and 77% of the between-lab ceiling (Broad PRISM, 107 drugs); both ceilings are low |
| Unseen cancer types | 🔴 Red for blood cancers | Within-tissue ranking roughly halves for haematological and skin lines when their tissue is absent from training |
| Gene lists | 🟡 Amber | Accurate models, but only ~6 of ~50 selected genes recur in all five folds; report frequencies, not lists |

**Conclusion.** Ridge on batch-corrected expression plus tissue is fit for ranking cell lines of *known* cancer types by expected sensitivity, for example to prioritise models for a screen. It is not fit for cancer types absent from training, and its gene lists are hypotheses, not biomarkers.

## 1. Data

| Check | Result |
|---|---|
| Processed AUC vs the GDSC2 release file | 493 random (line, drug) pairs; maximum difference 0 |
| Expression scale | The Cell Model Passports `rsem_tpm` column is already log2(TPM + 1) (GAPDH ≈ 12, maximum 18.4). The first build applied the log twice; caught by a range check and fixed before any modelling. |
| Batch effect | Data source predictable from 20 principal components with AUC 0.998 (whole sample). Fitted on folds 1–4 and applied to fold 0, the correction reduces it from 0.857 to 0.492 (chance level). Tissue classification is unaffected (0.705 → 0.707). |
| Duplicate profiles | Lines with both Sanger and Broad profiles use Sanger's (same institute as the screen) |

## 2. Leakage and evaluation design

| Check | Result |
|---|---|
| Test responses never used | Fold-2 AUCs set to 5.0 for Nutlin-3a and cytarabine: fold-2 predictions unchanged (maximum difference 0) |
| Preprocessing inside folds | Batch correction, gene filter and scaling are fitted on outer-training lines only (unit test `test_design_parameters_come_from_training_lines_only`). The design uses no response values. |
| Folds | Each of 953 lines in exactly one fold; every tissue with ≥ 25 lines differs by at most one line across folds |
| Nested tuning | Every penalty, mixing weight and component count chosen by inner CV on the outer-training lines |
| Baseline | A tissue-only model is reported for every drug, along with within-tissue accuracy, so tissue recognition cannot pass for biology |

Learn notebook 05 demonstrates on pure noise how selecting genes on all samples manufactures accuracy, the error these checks guard against.

## 3. Implementation

| Check | Result |
|---|---|
| Ridge (kernel SVD path) vs scikit-learn `Ridge` | max coefficient difference 6e−17 (Nutlin-3a, fold 0) |
| Group lasso KKT conditions on real data | stationarity gap 1.2e−4, no zero group violates its bound (Nutlin-3a, λ = 0.2 λmax, 16 active groups) |
| Group lasso with singleton groups equals the lasso | unit test passes |
| Batch correction protects confounded tissue effects | unit test with source and tissue 80% aligned passes |
| Group lasso convergence at production settings (400 iterations) | 🟡 out-of-fold r within ±0.02 of a tight solve (4,000 iterations, Nutlin-3a fold 0), but at small penalties more modules stay active (113 vs 75). Panel accuracy stands; its reported gene counts are overstated. |
| PRISM scoring | unit test with synthetic data passes; run on 107 real drugs (section 5) |

## 4. Performance

* **Against the baseline.** Ridge beats tissue alone on 274 of 281 drugs, raising the median r from 0.26 to 0.40 (Wilcoxon p = 3 × 10⁻⁴⁷; descriptive, since drugs sharing a target are not independent). Within cancer types, the median Spearman is 0.19.
* **Between models.** Ridge beats the lasso and the elastic net on 85% of drugs, and the group lasso and adaptive lasso on 92% of the panel. The margins are small but consistent: dense shrinkage suits this problem better than selection.
* **Robustness of the conclusion.** The ranking of models is the same with 1,000, 2,000 or 5,000 genes and with ln(IC50) as the response.

## 5. Generalisation

* **Unseen cancer types (red for blood cancers).** With the tissue held out of training, within-tissue Spearman falls from 0.36 to 0.14 for haematological lines and from 0.28 to 0.15 for skin. Solid tumours keep 79–94%. Haematological lines are also predicted as less sensitive than they are (median bias +0.07 AUC).
* **Another assay (amber).** Against GDSC1, predictions reach a median r of 0.39 against a between-assay agreement of 0.53. Drugs whose two assays agree transfer well; drugs whose assays disagree, such as paclitaxel and BI-2536, do not.
* **Another lab (amber).** Against the Broad Institute's PRISM screen (107 drugs, identity-verified lines, out-of-fold predictions only), ridge reaches a median r of 0.18 against a between-lab agreement of 0.24 (77%), and beats tissue on 79% of drugs. Lab agreement explains transfer almost entirely (correlation 0.85 across drugs).

## 6. Limitations and recommendations

1. **Use 5,000 genes, or a supervised-free filter that keeps key genes, when compute allows.** 5,000 genes improved ridge on 81% of panel drugs, and the variance filter dropped MDM2.
2. **Do not deploy to unseen cancer types**, especially haematological ones, without retraining on representative lines.
3. **Treat cross-lab accuracy as bounded by reproducibility.** GDSC and PRISM agree at only r = 0.24. Before acting on a prediction for a drug, check that drug's between-lab agreement (`prism_by_drug.csv`).
4. **Report gene selection frequencies with intervals**, never a single list. Confirm candidate biomarkers experimentally or in a second dataset.
5. **Add mutation and copy-number features.** Expression captures BRAF-mutant biology (ROC AUC 0.82 for dabrafenib) only indirectly.
