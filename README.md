# Validation protocol for radiomic survival analysis in small cohorts

Code for the paper "A four-step validation protocol for radiomic survival analysis in small cohorts: a body composition case study".

doi.org/10.5281/zenodo.23867744

The protocol has four steps: FDR-corrected feature selection, a dual-path comparison (selected features vs all features with regularisation), repeated nested cross-validation, and a negative control with Gaussian noise. It is applied to L3 CT body composition radiomics in 78 GEP-NET patients.

The patient data are not public. `data/README.md` says what file the code expects, and `scripts/make_synthetic_data.py` makes a fake table with the same columns to try things out.

## Layout

```
src/radiomics_protocol/
    pipeline/   selection.py (step 1, stability), nested_cv.py (steps 2-4), summaries.py
    utils/      data loading, preprocessing, models, metrics, cohort table
    config.py   seeds, thresholds, grids, paths
notebooks/01_validation_protocol.ipynb
R/              figure scripts
scripts/        synthetic data generator
tests/
data/           put the dataset here (not tracked)
results/        written by the notebook
figures/        written by the R scripts
```

## Running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Put the dataset in `data/radiomics_dataset.csv`, run `notebooks/01_validation_protocol.ipynb`, then from the repo root:

```bash
Rscript R/install_packages.R      # first time only
Rscript R/figures_protocol.R
Rscript R/figures_selected_features.R
```

The full run takes roughly 8 hours (about 25 min per seed, ten seeds, real and noise). Each seed is saved when it finishes, so an interrupted run resumes where it stopped. If you change a parameter, use a new results folder: `config.json` inside it stops you from mixing settings.

## Outputs

- `results/tables/`: table 1, table 2 and the stability tables
- `results/single_split/`: inputs for figure 4
- `results/real/summary/` and `results/null/summary/`: inputs for figures 5 and 6 and tables 3-4
- `results/{real,null}/long_seed*.csv`: per-fold results

Only aggregated numbers are written, no patient-level values.

## Tests

```bash
pytest -q        # about 3 min, synthetic data and small grids
```

`RADIOMICS_FAST=1` shrinks the grids and uses 2 seeds, and writes to `results_smoke/` and `figures_smoke/`. It is for checking that everything runs, not for the real analysis.

## Notes

- Scaling, feature selection and hyper-parameter tuning are always done on training data only.
- If step 1 selects nothing in a training set, path A gets C-index = AUC = 0.5 and the IBS is left undefined.
- Seeds 42-51 control folds, splits, noise features and the stochastic models. Results can still change a bit between library versions; `compare_runs` diffs two runs.
- Tables for reading (summaries, tables, R inputs) are `;`-separated with `.` as decimal mark (`CSV_SEP` in `config.py`). The per-seed files in `results/<run>/` are plain comma CSV and only used to resume runs.

Tested with Python 3.13, scikit-survival 0.28, lifelines 0.30, scikit-learn 1.9 and R 4.3.

## License

MIT
