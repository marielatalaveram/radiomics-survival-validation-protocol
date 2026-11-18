import glob
from pathlib import Path
import numpy as np
import pandas as pd

from .. import config
from ..config import KEYS


# single split vs nested cv
def isolated_results(long):
    # nested mean/sd per seed next to the single-split value
    nested = (long[long.scheme == "nested"].groupby(KEYS)["value"]
              .agg(nested_mean="mean", nested_sd=lambda x: x.std(ddof=1), n_folds="count")
              .reset_index())
    single = (long[long.scheme == "single"][KEYS + ["value", "empty"]]
              .rename(columns={"value": "single", "empty": "single_empty"}))
    return nested.merge(single, on=KEYS, how="left")


def delta_table(long, z_threshold=2.0):
    """single - nested per seed/path/model/metric, with z = delta / sd between outer folds.
    For the IBS a positive delta means the single split is worse."""
    d = isolated_results(long)
    d["delta"] = d["single"] - d["nested_mean"]
    d["abs_delta"] = d["delta"].abs()
    d["z"] = d["delta"] / d["nested_sd"].replace(0, np.nan)
    d["abs_z_ge"] = (d["z"].abs() >= z_threshold).where(d["z"].notna()).astype("boolean")
    return d


def summarize_delta(delta):
    g = delta.groupby(["path", "model", "metric"])
    out = g.agg(n_seeds=("seed", "nunique"), n_valid=("delta", "count"),
                nested_mean=("nested_mean", "mean"),
                sd_nested_mean_between_seeds=("nested_mean", "std"),
                mean_fold_sd=("nested_sd", "mean"),
                single_mean=("single", "mean"),
                single_sd=("single", "std"),
                mean_delta=("delta", "mean"),
                sd_delta=("delta", "std"),
                mean_abs_delta=("abs_delta", "mean"),
                frac_delta_pos=("delta", lambda x: (x.dropna() > 0).mean()),
                mean_z=("z", "mean"),
                sd_z=("z", "std"),
                frac_abs_z_ge2=("abs_z_ge", "mean"))
    return out.round(3)


# path a vs path b
def path_comparison(long, scheme="nested"):
    """A - B paired by outer fold, summarised per seed/model/metric."""
    L = long[long.scheme == scheme]
    W = L.pivot_table(index=["seed", "model", "metric", "fold"], columns="path", values="value")
    W["diff"] = W["A"] - W["B"]
    g = W.groupby(["seed", "model", "metric"])["diff"]
    out = g.agg(diff_mean="mean", diff_sd=lambda x: x.std(ddof=1)).reset_index()
    out["z_pair"] = out["diff_mean"] / out["diff_sd"].replace(0, np.nan)
    # lower is better for the IBS
    a_better = pd.Series(np.where(out["metric"] == "ibs", out["diff_mean"] < 0, out["diff_mean"] > 0),
                         index=out.index)
    out["A_better"] = a_better.where(out["diff_mean"].notna()).astype("boolean")
    return out


def summarize_path_comparison(pc):
    g = pc.groupby(["model", "metric"])
    return g.agg(n_seeds=("diff_mean", "count"), mean_diff=("diff_mean", "mean"),
                 sd_diff_between_seeds=("diff_mean", "std"),
                 mean_fold_sd_of_diff=("diff_sd", "mean"),
                 frac_seeds_A_better=("A_better", "mean")).round(3)


# empty selections
def empty_report(long, sel=None):
    # splits where step 1 selected nothing (path A)
    ref_model = config.MODELS[0] if (long.model == config.MODELS[0]).any() else long.model.iloc[0]
    e = long[(long.path == "A") & (long.metric == "ci_test") & (long.model == ref_model)]
    return e.groupby("scheme")["empty"].agg(n_empty="sum", n_splits="count")


# loading / exporting / comparing runs
SUMMARY_FILES = {
    "delta": "delta_by_seed.csv",
    "delta_summary": "delta_summary.csv",
    "dual_path": "dual_path_by_seed.csv",
    "dual_path_summary": "dual_path_summary.csv",
    "empty": "empty_selections.csv",
}


def load_run(folder):
    folder = Path(folder)
    long = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(folder / "long_seed*.csv")))],
                     ignore_index=True)
    sel = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(folder / "selection_seed*.csv")))],
                    ignore_index=True)
    return long, sel


def export_summaries(long, sel, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    delta = delta_table(long)
    tables = {
        "delta": delta,
        "delta_summary": summarize_delta(delta),
        "dual_path": path_comparison(long),
        "empty": empty_report(long, sel),
    }
    tables["dual_path_summary"] = summarize_path_comparison(tables["dual_path"])
    for key, df in tables.items():
        index = key in ("delta_summary", "dual_path_summary", "empty")
        df.to_csv(out / SUMMARY_FILES[key], sep=config.CSV_SEP, index=index)
    return tables


def merge_model_results(long_base, long_new, models):
    # only needed if a single model was re-run on its own
    models = list(models)
    out = pd.concat([long_base[~long_base.model.isin(models)],
                     long_new[long_new.model.isin(models)]], ignore_index=True)
    assert set(out.seed.unique()) == set(long_base.seed.unique()), "seed sets differ"
    assert len(out) == len(long_base), (len(out), len(long_base))
    return out


def compare_runs(long_a, long_b, *, models=None, atol=1e-8):
    """Rows that differ between two runs (empty frame = identical)."""
    key = ["seed", "scheme", "fold", "path", "model", "metric"]
    a, b = long_a, long_b
    if models is not None:
        models = list(models)
        a, b = a[a.model.isin(models)], b[b.model.isin(models)]
    m = a[key + ["value"]].merge(b[key + ["value"]], on=key, how="outer",
                                 suffixes=("_a", "_b"), indicator=True)
    both = m["_merge"] == "both"
    same = (np.isclose(m["value_a"], m["value_b"], atol=atol, rtol=0, equal_nan=True)) & both
    return m[~same].reset_index(drop=True)
