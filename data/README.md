# Data

The study data are not public (patient privacy). Nothing in this folder is tracked except this file.

Put the table in `data/radiomics_dataset.csv`, `;`-separated, or point `RADIOMICS_DATA` to another path.

Columns used:

- `Corr_T1_<family>_<feature>_seg<k>`: radiomic features (535 = 107 x 5 compartments). Every column starting with `Corr_T1_` is a predictor. seg3 = intramuscular fat, seg4 = skeletal muscle, seg5 = subcutaneous fat.
- `exitus`: 1 = death, 0 = censored
- `dias_supervivencia`: survival time in days
- `sexo`: F / M
- `age_at_diag`: age in years
- `age_ranges`: `<55`, `55-65`, `65-75`, `>75`

Patients with a missing radiomic value are dropped (113 to 78 in the study). Other columns are ignored.

To try the code without the real data:

```bash
python scripts/make_synthetic_data.py --n-features 60
RADIOMICS_FAST=1 RADIOMICS_DATA=data/synthetic/radiomics_dataset.csv jupyter nbconvert --to notebook --execute notebooks/01_validation_protocol.ipynb
```

The synthetic table only has the right shape, its results mean nothing.
