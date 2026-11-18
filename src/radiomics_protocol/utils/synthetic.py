"""Fake data with the same columns as the real table. Only for tests and smoke runs."""
import numpy as np
import pandas as pd

# 107 pyradiomics features per compartment
FAMILIES = {
    "shape": ["Elongation", "Flatness", "LeastAxisLength", "MajorAxisLength", "Maximum2DDiameterColumn",
              "Maximum2DDiameterRow", "Maximum2DDiameterSlice", "Maximum3DDiameter", "MeshVolume",
              "MinorAxisLength", "Sphericity", "SurfaceArea", "SurfaceVolumeRatio", "VoxelVolume"],
    "firstorder": ["10Percentile", "90Percentile", "Energy", "Entropy", "InterquartileRange", "Kurtosis",
                   "Maximum", "MeanAbsoluteDeviation", "Mean", "Median", "Minimum", "Range",
                   "RobustMeanAbsoluteDeviation", "RootMeanSquared", "Skewness", "TotalEnergy",
                   "Uniformity", "Variance"],
    "glcm": ["Autocorrelation", "ClusterProminence", "ClusterShade", "ClusterTendency", "Contrast",
             "Correlation", "DifferenceAverage", "DifferenceEntropy", "DifferenceVariance", "Id", "Idm",
             "Idmn", "Idn", "Imc1", "Imc2", "InverseVariance", "JointAverage", "JointEnergy",
             "JointEntropy", "MCC", "MaximumProbability", "SumAverage", "SumEntropy", "SumSquares"],
    "gldm": ["DependenceEntropy", "DependenceNonUniformity", "DependenceNonUniformityNormalized",
             "DependenceVariance", "GrayLevelNonUniformity", "GrayLevelVariance", "HighGrayLevelEmphasis",
             "LargeDependenceEmphasis", "LargeDependenceHighGrayLevelEmphasis",
             "LargeDependenceLowGrayLevelEmphasis", "LowGrayLevelEmphasis", "SmallDependenceEmphasis",
             "SmallDependenceHighGrayLevelEmphasis", "SmallDependenceLowGrayLevelEmphasis"],
    "glrlm": ["GrayLevelNonUniformity", "GrayLevelNonUniformityNormalized", "GrayLevelVariance",
              "HighGrayLevelRunEmphasis", "LongRunEmphasis", "LongRunHighGrayLevelEmphasis",
              "LongRunLowGrayLevelEmphasis", "LowGrayLevelRunEmphasis", "RunEntropy",
              "RunLengthNonUniformity", "RunLengthNonUniformityNormalized", "RunPercentage", "RunVariance",
              "ShortRunEmphasis", "ShortRunHighGrayLevelEmphasis", "ShortRunLowGrayLevelEmphasis"],
    "glszm": ["GrayLevelNonUniformity", "GrayLevelNonUniformityNormalized", "GrayLevelVariance",
              "HighGrayLevelZoneEmphasis", "LargeAreaEmphasis", "LargeAreaHighGrayLevelEmphasis",
              "LargeAreaLowGrayLevelEmphasis", "LowGrayLevelZoneEmphasis", "SizeZoneNonUniformity",
              "SizeZoneNonUniformityNormalized", "SmallAreaEmphasis", "SmallAreaHighGrayLevelEmphasis",
              "SmallAreaLowGrayLevelEmphasis", "ZoneEntropy", "ZonePercentage", "ZoneVariance"],
    "ngtdm": ["Busyness", "Coarseness", "Complexity", "Contrast", "Strength"],
}


def make_synthetic_dataset(n_patients: int = 113, n_complete: int = 78, n_features: int = 535,
                           seed: int = 0, signal: float = 0.45):
    rng = np.random.default_rng(seed)
    names = [f"Corr_T1_{fam}_{f}_seg{k}" for k in range(1, 6) for fam, fs in FAMILIES.items() for f in fs]
    assert len(names) == 535
    if n_features < 535:
        keep = np.sort(rng.choice(535, n_features, replace=False))
        names = [names[i] for i in keep]

    n, p = n_patients, len(names)
    k = max(3, p // 20)                      # latent factors, gives correlated blocks
    latent = rng.standard_normal((n, k))
    loadings = rng.standard_normal((k, p)) * (rng.random((k, p)) < 0.15)
    X = latent @ loadings + rng.standard_normal((n, p))

    lin = signal * latent[:, 0] - (2 / 3) * signal * latent[:, 1]   # weak signal
    t_event = rng.exponential(scale=9500 * np.exp(-lin))
    t_cens = rng.uniform(500, 6000, n)
    time = np.minimum(t_event, t_cens).round()
    event = (t_event <= t_cens).astype(int)

    age = np.clip(rng.normal(63, 14, n), 20, 90).round(1)
    age_ranges = np.select([age < 55, age < 65, age < 75], ["<55", "55-65", "65-75"], ">75")
    df = pd.DataFrame(X, columns=names)
    df.insert(0, "patient_id", [f"SYN{i:03d}" for i in range(n)])
    df["sexo"] = rng.choice(["F", "M"], n)
    df["age_at_diag"] = age
    df["age_ranges"] = age_ranges
    df["dias_supervivencia"] = time
    df["exitus"] = event
    incomplete = rng.choice(n, n - n_complete, replace=False)
    df.loc[incomplete, names] = np.nan
    return df
