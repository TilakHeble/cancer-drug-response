# Learn: regularised regression when features outnumber samples

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
