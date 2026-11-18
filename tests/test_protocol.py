import numpy as np
import pandas as pd
import pytest

from radiomics_protocol import config
from radiomics_protocol.pipeline import (compare_runs, delta_table, empty_report, load_run,
                                          merge_model_results, path_comparison,
                                          run_repeated_validation, selection_stability,
                                          single_split_report, stability_summary,
                                          summarize_delta, summarize_path_comparison)
from radiomics_protocol.utils import (cohort_table, correlation_filter, prepare_dataset,
                                      preprocess_split, ssvm)

KW = dict(alpha_fdr=0.2, corr_method="single", corr_threshold=0.8)


def test_fast_mode_is_on():
    assert config.FAST_MODE and config.N_SEEDS == 2


def test_prepare_dataset_drops_incomplete_rows(dataset):
    assert dataset.X.shape == (60, 24)
    assert dataset.y.dtype.names == ("exitus", "dias_supervivencia")
    assert dataset.n_raw == 70
    assert np.isfinite(dataset.X.values).all()


def test_cohort_table_overall_counts(dataset):
    t = cohort_table(dataset.full)
    assert t.iloc[0, 0].startswith("60 (100.0%)")


def test_correlation_filter_keeps_best_of_each_cluster():
    rng = np.random.default_rng(0)
    a = rng.standard_normal(80)
    X = pd.DataFrame({"a": a, "b": a * 2 + 1e-3 * rng.standard_normal(80),   # |rho| ~ 1 with a
                      "c": rng.standard_normal(80)})
    cox = pd.DataFrame({"variable": ["a", "b", "c"], "concordance": [0.60, 0.70, 0.55]})
    kept = correlation_filter(X, 0.8, cox, "single", "concordance", False)
    assert sorted(kept) == ["b", "c"]


def test_ssvm_uses_fixed_negative_sign(dataset):
    X, y = dataset.X.iloc[:45], dataset.y
    Xp, Xtp, _ = preprocess_split(X, dataset.X.iloc[45:])
    model, res = ssvm(Xp, Xtp, y[:45], y[45:])
    r = res["SSVM (all)"]
    assert r["negated"] is True
    np.testing.assert_allclose(r["risk_test"], -model.predict(Xtp.values))


def test_stability_outputs(dataset):
    freq, sizes, jac, sets = selection_stability(dataset.X, dataset.y, n_rep=6, n_jobs=1,
                                                 alpha_fdr=0.2, corr_threshold=0.8)
    assert len(sets) == 6 and len(sizes) == 6 and len(jac) <= 15
    assert freq.between(0, 1).all() and freq.is_monotonic_decreasing
    s = stability_summary(freq, sizes, jac, sets)
    assert 0 <= s["jaccard_mean"] <= 1


def test_full_run_summaries_and_single_split_consistency(dataset, tmp_path):
    out = tmp_path / "run"
    long, sel = run_repeated_validation(dataset.X, dataset.y, seeds=[42], out_dir=out,
                                        models=["Cox_RIDGE", "RSF"], **KW)
    # loose alpha so that something gets selected
    kw_loose = dict(KW, alpha_fdr=0.6)
    _, sel_loose = run_repeated_validation(dataset.X, dataset.y, seeds=[42], models=["Cox_RIDGE"], **kw_loose)
    rep = single_split_report(dataset.X, dataset.y, seed=42, alpha_fdr=0.6)
    chosen = sel_loose[(sel_loose.scheme == "single")].iloc[0]["selected"]
    chosen = set(chosen.split(";")) - {""} if isinstance(chosen, str) else set()
    assert len(rep["selected"]) > 0
    assert chosen == set(rep["selected"])

    d = delta_table(long)
    for col in ["delta", "abs_delta", "z", "abs_z_ge", "nested_mean", "nested_sd", "single"]:
        assert col in d.columns
    assert not summarize_delta(d).empty
    pc = path_comparison(long)
    assert not summarize_path_comparison(pc).empty
    assert empty_report(long, sel).loc["nested", "n_splits"] == 5

    # second call loads from out_dir
    long2, _ = run_repeated_validation(dataset.X, dataset.y, seeds=[42], out_dir=out,
                                       models=["Cox_RIDGE", "RSF"], **KW)
    assert compare_runs(long, long2).empty
    # different settings in the same out_dir are refused
    with pytest.raises(ValueError):
        run_repeated_validation(dataset.X, dataset.y, seeds=[42], out_dir=out,
                                models=["Cox_RIDGE", "RSF"], alpha_fdr=0.1,
                                corr_method="single", corr_threshold=0.8)
    long3, _ = load_run(out)
    assert compare_runs(long, long3).empty


def test_one_complete_run_equals_old_two_step_workflow(dataset):
    # one run with all models == other models + SSVM run separately
    seeds = [42]
    full, _ = run_repeated_validation(dataset.X, dataset.y, seeds=seeds, **KW)
    others, _ = run_repeated_validation(dataset.X, dataset.y, seeds=seeds, models=["Cox_RIDGE", "Cox_LASSO", "RSF", "GBM"], **KW)
    only_ssvm, _ = run_repeated_validation(dataset.X, dataset.y, seeds=seeds, models=["SSVM"], **KW)
    merged = pd.concat([others, only_ssvm], ignore_index=True)
    assert len(full) == len(merged)
    diff = compare_runs(full, merged)
    assert diff.empty, diff.head()
    # deterministic
    full2, _ = run_repeated_validation(dataset.X, dataset.y, seeds=seeds, **KW)
    assert compare_runs(full, full2).empty


def test_gaussian_null_changes_features_not_folds(dataset):
    real, _ = run_repeated_validation(dataset.X, dataset.y, seeds=[43], models=["Cox_RIDGE"], **KW)
    null, _ = run_repeated_validation(dataset.X, dataset.y, seeds=[43], models=["Cox_RIDGE"], null="gaussian", **KW)
    assert set(zip(real.scheme, real.fold)) == set(zip(null.scheme, null.fold))
    assert not compare_runs(real, null).empty


def test_merge_model_results_guards(dataset):
    a, _ = run_repeated_validation(dataset.X, dataset.y, seeds=[42], models=["Cox_RIDGE", "SSVM"], **KW)
    b, _ = run_repeated_validation(dataset.X, dataset.y, seeds=[42], models=["SSVM"], **KW)
    assert compare_runs(a, merge_model_results(a, b, ["SSVM"])).empty


def test_stability_matches_original_function(dataset):
    # the function used in the first version of the notebook
    import itertools
    from collections import Counter
    from sklearn.model_selection import StratifiedShuffleSplit
    from sklearn.preprocessing import StandardScaler
    from radiomics_protocol.utils import correlation_filter, univariate_cox_filter_fdr

    def original(X, y_df, n_rep, test_size=0.30, alpha=0.20, corr_thr=0.80, base_seed=42, corr_method="single"):
        sss = StratifiedShuffleSplit(n_splits=n_rep, test_size=test_size, random_state=base_seed)
        sel_sets = []
        for tr_idx, _ in sss.split(X, y_df["exitus"]):
            X_tr, y_tr = X.iloc[tr_idx], y_df.iloc[tr_idx]
            X_tr_proc = pd.DataFrame(StandardScaler().fit_transform(X_tr), columns=X_tr.columns, index=X_tr.index)
            sel_cox, df_cox = univariate_cox_filter_fdr(X_tr_proc, y_tr, alpha)
            if len(sel_cox) == 0:
                sel_sets.append(set())
                continue
            sel = correlation_filter(X_tr_proc[sel_cox], corr_thr, df_cox, corr_method, "concordance", False)
            sel_sets.append(set(sel))
        freq = pd.Series(Counter(f for s in sel_sets for f in s)).sort_values(ascending=False) / n_rep
        sizes = np.array([len(s) for s in sel_sets])
        jac = [len(a & b) / len(a | b) for a, b in itertools.combinations(sel_sets, 2) if (a | b)]
        return freq, sizes, np.array(jac), sel_sets

    y_df = pd.DataFrame({"exitus": dataset.y["exitus"], "dias_supervivencia": dataset.y["dias_supervivencia"]},
                        index=dataset.X.index)
    f0, s0, j0, sets0 = original(dataset.X, y_df, 8, alpha=0.5)
    f1, s1, j1, sets1 = selection_stability(dataset.X, dataset.y, n_rep=8, alpha_fdr=0.5, n_jobs=1)
    assert sets0 == sets1
    np.testing.assert_array_equal(s0, s1)
    np.testing.assert_allclose(j0, j1)
    np.testing.assert_allclose(f1[f0.index].values, f0.values)


def test_summary_tables_roundtrip_with_csv_sep(dataset, tmp_path):
    from radiomics_protocol.pipeline import export_summaries
    from radiomics_protocol.pipeline.summaries import SUMMARY_FILES
    long, sel = run_repeated_validation(dataset.X, dataset.y, seeds=[42], models=["Cox_RIDGE", "RSF"], **KW)
    tables = export_summaries(long, sel, tmp_path)
    for key, name in SUMMARY_FILES.items():
        raw = (tmp_path / name).read_text()
        assert "," not in raw.splitlines()[0]          # header is split by CSV_SEP only
        df = tables[key]
        levels = df.index.nlevels if key in ("delta_summary", "dual_path_summary", "empty") else 0
        back = pd.read_csv(tmp_path / name, sep=config.CSV_SEP, index_col=list(range(levels)) if levels else None)
        assert back.shape == df.shape, key
    # no semicolon inside a cell, otherwise the tables could not be split reliably
    for key, df in tables.items():
        assert not df.astype(str).apply(lambda c: c.str.contains(config.CSV_SEP)).any().any(), key
