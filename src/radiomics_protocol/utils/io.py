import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config

RADIOMIC_PREFIX = "Corr_T1_"
CLINICAL_COLUMNS = ["sexo", "age_ranges", "dias_supervivencia", "exitus"]


def get_data_path(filename=None):
    env = os.environ.get("RADIOMICS_DATA")
    if env and filename is None:
        path = Path(env).expanduser()
        if not path.is_absolute() and not path.exists():
            path = config.get_project_root() / path
    else:
        path = config.get_data_dir() / (filename or config.DATASET_FILENAME)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Put the csv there (see data/README.md) "
            f"or set RADIOMICS_DATA."
        )
    return path


def load_radiomics_dataset(filename=None, sep=config.DATASET_SEP):
    path = get_data_path(filename)
    return pd.read_csv(path, sep=sep), path


@dataclass
class Dataset:
    X: pd.DataFrame           # radiomic features
    y: np.ndarray             # structured: exitus (bool), dias_supervivencia (float)
    clinical: pd.DataFrame
    full: pd.DataFrame        # complete cases of the raw table
    feature_cols: list
    n_raw: int


def prepare_dataset(df_raw):
    """Drop patients with missing radiomics and build X / y."""
    feature_cols = [c for c in df_raw.columns if c.startswith(RADIOMIC_PREFIX)]
    if not feature_cols:
        raise ValueError(f"No columns starting with '{RADIOMIC_PREFIX}'")
    missing = [c for c in CLINICAL_COLUMNS if c not in df_raw.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df_raw.dropna(subset=feature_cols)
    X = df[feature_cols].copy()
    clinical = df[CLINICAL_COLUMNS].copy()
    clinical["exitus"] = clinical["exitus"].astype(bool)

    y = np.empty(len(df), dtype=[("exitus", bool), ("dias_supervivencia", float)])
    y["exitus"] = df["exitus"].astype(bool).values
    y["dias_supervivencia"] = df["dias_supervivencia"].astype(float).values
    return Dataset(X=X, y=y, clinical=clinical, full=df, feature_cols=feature_cols,
                   n_raw=len(df_raw))
