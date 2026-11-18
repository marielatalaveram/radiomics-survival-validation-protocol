import os

# has to be set before importing radiomics_protocol
os.environ["RADIOMICS_FAST"] = "1"

import pytest

from radiomics_protocol.utils import prepare_dataset
from radiomics_protocol.utils.synthetic import make_synthetic_dataset


@pytest.fixture(scope="session")
def dataset():
    return prepare_dataset(make_synthetic_dataset(n_patients=70, n_complete=60, n_features=24, seed=1, signal=1.5))
