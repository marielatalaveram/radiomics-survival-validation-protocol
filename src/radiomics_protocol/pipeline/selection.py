import itertools
import re
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

from .. import config
from ..utils.preprocessing import (as_structured, correlation_filter, preprocess_split,
                                   structured_to_frame, univariate_cox_filter_fdr)


# Step 1 on one training set
def select_features(X_train_scaled, y_train_df, *,
                    alpha_fdr=config.ALPHA_FDR,
                    corr_threshold=config.CORR_THRESHOLD,
                    corr_method=config.CORR_METHOD,
                    criterion=config.CRITERION,
                    ascending=config.ASCENDING,
                    univariate_filter=univariate_cox_filter_fdr):
    """FDR filter then redundancy filter, on already standardised training data.
    Returns dict with selected, n_fdr and df_cox."""
    sel_cox, df_cox = univariate_filter(X_train_scaled, y_train_df, alpha_fdr)
    sel_cox = list(sel_cox)
    if len(sel_cox) == 0:
        selected = []
    elif len(sel_cox) == 1:
        selected = sel_cox
    else:
        selected = correlation_filter(X_train_scaled[sel_cox], corr_threshold, df_cox,
                                      corr_method, criterion, ascending)
    return {"selected": selected, "n_fdr": len(sel_cox), "df_cox": df_cox}


# Reference single split (Table 2 / Figure 4 of the paper)
def single_split_report(X_raw, y, *, seed=config.DEFAULT_SEED, test_size=config.TEST_SIZE,
                        **step1_kwargs):
    """Step 1 on the training part of the 70/30 split of `seed` (the same split the
    repeated validation uses as "single"). Used for Table 2 / Figure 4."""
    y = as_structured(y)
    idx = np.arange(len(X_raw))
    tr, te = train_test_split(idx, test_size=test_size, stratify=y["exitus"], random_state=seed)
    X_tr, X_te = X_raw.iloc[tr], X_raw.iloc[te]
    y_tr_df = structured_to_frame(y[tr], X_tr.index)
    X_tr_p, _, _ = preprocess_split(X_tr, X_te)

    res = select_features(X_tr_p, y_tr_df, **step1_kwargs)
    sel = res["selected"]
    cox_sel = res["df_cox"][res["df_cox"]["variable"].isin(sel)].reset_index(drop=True)
    spearman = X_tr[sel].corr(method="spearman") if sel else pd.DataFrame()
    return {**res, "cox_selected": cox_sel, "spearman": spearman,
            "train_idx": tr, "test_idx": te}


# Selection stability (100 repetitions of Step 1 on random 70/30 splits)
def _one_repetition(tr, X_raw, y_df, step1_kwargs):
    X_tr_p, _, _ = preprocess_split(X_raw.iloc[tr], X_raw.iloc[tr])
    res = select_features(X_tr_p, y_df.iloc[tr], **step1_kwargs)
    return set(res["selected"])


def selection_stability(X_raw, y, n_rep=config.N_STABILITY_REPS, *,
                        alpha_fdr=config.ALPHA_FDR,
                        corr_method=config.CORR_METHOD,
                        corr_threshold=config.CORR_THRESHOLD,
                        criterion=config.CRITERION,
                        ascending=config.ASCENDING,
                        test_size=config.TEST_SIZE,
                        seed=config.DEFAULT_SEED,
                        n_jobs=-1):
    """Repeat step 1 on n_rep splits from StratifiedShuffleSplit(random_state=seed).
    Returns (freq, sizes, jaccard, sel_sets): selection frequency per feature, size of each
    selection, pairwise Jaccard (pairs where both sets are empty are left out) and the sets."""
    y = as_structured(y)
    y_df = structured_to_frame(y, X_raw.index)
    step1_kwargs = dict(alpha_fdr=alpha_fdr, corr_threshold=corr_threshold, corr_method=corr_method,
                        criterion=criterion, ascending=ascending)

    sss = StratifiedShuffleSplit(n_splits=n_rep, test_size=test_size, random_state=seed)
    splits = [tr for tr, _ in sss.split(X_raw, y["exitus"])]
    sel_sets = Parallel(n_jobs=n_jobs)(
        delayed(_one_repetition)(tr, X_raw, y_df, step1_kwargs) for tr in splits)

    counts = pd.Series(0.0, index=X_raw.columns)
    for s in sel_sets:
        for f in s:
            counts[f] += 1
    freq = (counts / n_rep).sort_values(ascending=False, kind="stable")
    sizes = np.array([len(s) for s in sel_sets])
    jaccard = np.array([len(a & b) / len(a | b)
                        for a, b in itertools.combinations(sel_sets, 2) if (a | b)])
    return freq, sizes, jaccard, sel_sets


def stability_summary(freq, sizes, jaccard, sel_sets):
    nonempty = sizes > 0
    ne_sets = [s for s in sel_sets if s]
    jac_ne = np.array([len(a & b) / len(a | b) for a, b in itertools.combinations(ne_sets, 2)])
    sd = lambda x: float(np.std(x, ddof=1)) if len(x) > 1 else float("nan")
    return {
        "n_rep": len(sel_sets),
        "size_mean": float(sizes.mean()), "size_sd": sd(sizes),
        "n_empty": int((~nonempty).sum()),
        "size_nonempty_mean": float(sizes[nonempty].mean()) if nonempty.any() else float("nan"),
        "size_nonempty_sd": sd(sizes[nonempty]),
        "jaccard_mean": float(jaccard.mean()), "jaccard_sd": sd(jaccard),
        "jaccard_nonempty_mean": float(jac_ne.mean()) if len(jac_ne) else float("nan"),
        "jaccard_nonempty_sd": sd(jac_ne),
    }


def conditional_frequency(freq, sizes):
    # CSF: only repetitions with at least one feature count
    return freq * len(sizes) / (sizes > 0).sum()


# Table 2
_NAME_RE = re.compile(r"^Corr_T1_(?P<family>[a-z0-9]+)_(?P<feature>.+)_(?P<seg>seg\d+)$")


def parse_feature_name(name):
    m = _NAME_RE.match(name)
    if not m:
        return {"family": "", "feature": name, "compartment": ""}
    seg = m.group("seg")
    return {"family": m.group("family"), "feature": m.group("feature"),
            "compartment": config.COMPARTMENTS.get(seg, seg)}


def selected_features_table(cox_selected, freq, freq_cond):
    # Table 2
    t = cox_selected.copy()
    parsed = pd.DataFrame([parse_feature_name(v) for v in t["variable"]], index=t.index)
    t = pd.concat([parsed, t], axis=1)
    t["SF"] = t["variable"].map(freq).values
    t["CSF"] = t["variable"].map(freq_cond).values
    cols = ["compartment", "family", "feature", "variable", "HR", "HR_lower95", "HR_upper95",
            "p_raw", "p_adj_fdr", "concordance", "p_schoenfeld", "SF", "CSF"]
    return t[cols].sort_values(["compartment", "p_raw"]).reset_index(drop=True)
