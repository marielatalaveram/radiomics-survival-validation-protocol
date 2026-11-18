"""Constants, hyper-parameter grids and paths.

Env vars: RADIOMICS_DATA (path to the csv), RADIOMICS_PROJECT_ROOT, RADIOMICS_FAST=1
(tiny grids and 2 seeds, only for smoke tests on synthetic data).
"""
import os
import warnings
from pathlib import Path

import numpy as np

MODELS = ["Cox_RIDGE", "Cox_LASSO", "RSF", "GBM", "SSVM"]
PATHS = ["A", "B"]          # A: selected features, B: all features
METRICS = ["ci_test", "auc", "ibs"]
KEYS = ["seed", "path", "model", "metric"]

DEFAULT_SEED = 42
N_SEEDS = 10                # seeds 42-51

# step 1
ALPHA_FDR = 0.20
CORR_METHOD = "single"
CORR_THRESHOLD = 0.80
CRITERION = "concordance"
ASCENDING = False

N_OUTER = 5
TEST_SIZE = 0.30
N_TIME_POINTS = 10
N_STABILITY_REPS = 100

# seg1 and seg2 are not mapped (they never showed up in the selected features)
COMPARTMENTS = {
    "seg3": "Intramuscular fat",
    "seg4": "Skeletal muscle",
    "seg5": "Subcutaneous fat",
}

FAST_MODE = os.environ.get("RADIOMICS_FAST", "0") == "1"

if FAST_MODE:
    GRIDS = {
        "cox_penalizers": np.logspace(-1, 2, 3),
        "rsf": {"n_estimators": [20], "min_samples_split": [15],
                "min_samples_leaf": [15], "max_features": ["sqrt"]},
        "gbm": {"n_estimators": [5], "learning_rate": [0.1],
                "subsample": [1.0], "max_depth": [1]},
        "ssvm": {"alpha": [1.0, 4.0], "kernel": ["linear", "rbf"], "gamma": [None]},
    }
    N_SEEDS = 2
    N_STABILITY_REPS = 6
else:
    GRIDS = {
        "cox_penalizers": np.logspace(-1, 2, 10),
        "rsf": {
            "n_estimators": [500, 1000],
            "min_samples_split": [15, 20, 25],
            "min_samples_leaf": [15, 20, 25, 30],
            "max_features": ["sqrt", "log2", 0.5, 0.3],
        },
        "gbm": {
            "n_estimators": [5, 7, 10, 20],
            "learning_rate": [0.01, 0.05, 0.1, 0.5],
            "subsample": [1.0, 0.5, 0.7],
            "max_depth": [1, 2],
        },
        "ssvm": {
            "alpha": [2.0 ** v for v in range(-10, 10)],
            "kernel": ["rbf", "poly", "linear"],
            # None = 1/p in scikit-survival ("scale" is not accepted by FastKernelSurvivalSVM)
            "gamma": [None, 0.001, 0.01, 0.1, 1.0],
        },
    }


def default_seeds(start=DEFAULT_SEED, n=None):
    return [start + i for i in range(n if n is not None else N_SEEDS)]


def silence_warnings():
    # the env var is needed so that joblib workers inherit it
    os.environ["PYTHONWARNINGS"] = "ignore"
    warnings.filterwarnings("ignore")


DATASET_FILENAME = "radiomics_dataset.csv"
DATASET_SEP = ";"
# separator of every table written for people (summaries, tables, R inputs).
# per-seed files in results/<run>/ are intermediate and stay comma-separated
CSV_SEP = ";"


def get_project_root(start=None):
    env = os.environ.get("RADIOMICS_PROJECT_ROOT")
    if env:
        return Path(env).resolve()
    candidate = (start or Path.cwd()).resolve()
    for path in [candidate, *candidate.parents]:
        if (path / "pyproject.toml").exists() and (path / "src").exists():
            return path
    return Path(__file__).resolve().parents[2]


def get_data_dir():
    return get_project_root() / "data"


def get_results_dir():
    return get_project_root() / ("results_smoke" if FAST_MODE else "results")
