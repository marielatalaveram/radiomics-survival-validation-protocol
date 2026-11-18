import contextlib
import io
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from .. import config
from ..utils.metrics import metrics_from_result
from ..utils.models import train_and_evaluate_models as _default_train_models
from ..utils.preprocessing import as_structured
from ..utils.preprocessing import preprocess_split as _default_preprocess
from ..utils.preprocessing import structured_to_frame, univariate_cox_filter_fdr
from .selection import select_features

LONG_COLUMNS = ["seed", "scheme", "fold", "path", "model", "metric", "value",
                "n_features", "empty", "failed"]


def evaluate_split(X_tr, X_te, y_tr, y_te, *, global_times, cfg, deps, verbose=False):
    """Step 1 + both paths on one train/test partition. Returns (rows, selection)."""
    models = cfg["models"]
    y_tr_df = structured_to_frame(y_tr, X_tr.index)
    X_tr_p, X_te_p, _ = deps["preprocess_split"](X_tr, X_te)

    # AUC / IBS times have to fall inside both follow-up ranges
    lo = max(y_tr["dias_supervivencia"].min(), y_te["dias_supervivencia"].min())
    hi = min(y_tr["dias_supervivencia"].max(), y_te["dias_supervivencia"].max())
    mask = (global_times > lo) & (global_times < hi)
    times = global_times[mask] if mask.sum() >= 2 else None

    rows = []

    def add(path, model, vals, n_feat, empty, failed):
        for m in config.METRICS:
            rows.append({"path": path, "model": model, "metric": m,
                         "value": vals.get(m, np.nan),
                         "n_features": n_feat, "empty": empty, "failed": failed})

    def run_models(path, X_tr_eval, X_te_eval):
        with (contextlib.nullcontext() if verbose else contextlib.redirect_stdout(io.StringIO())):
            try:
                results = deps["train_models"](X_tr_eval, X_te_eval, y_tr, y_te, y_tr_df,
                                               seed=cfg["_seed"], models=models)
            except Exception as e:
                print(f"  path {path} failed: {e}")
                results = {}
        for model in models:
            if model in results:
                add(path, model, metrics_from_result(results[model], X_te_eval, y_tr, y_te, times),
                    X_tr_eval.shape[1], 0, 0)
            else:
                add(path, model, {}, X_tr_eval.shape[1], 0, 1)

    # path A
    step1 = select_features(X_tr_p, y_tr_df, alpha_fdr=cfg["alpha_fdr"],
                            corr_threshold=cfg["corr_threshold"], corr_method=cfg["corr_method"],
                            criterion=cfg["criterion"], ascending=cfg["ascending"],
                            univariate_filter=deps["univariate_filter"])
    sel_final = step1["selected"]
    selection = {"n_fdr": step1["n_fdr"], "n_final": len(sel_final), "selected": ";".join(sel_final)}

    if len(sel_final) == 0:
        # nothing selected: chance level, IBS undefined
        for model in models:
            if cfg["empty_policy"] == "chance":
                add("A", model, {"ci_test": 0.5, "auc": 0.5 if times is not None else np.nan}, 0, 1, 0)
            else:
                add("A", model, {}, 0, 1, 0)
    else:
        run_models("A", X_tr_p[sel_final], X_te_p[sel_final])

    # path B
    run_models("B", X_tr_p, X_te_p)
    return rows, selection


def run_repeated_validation(X_raw, y_structured, alpha_fdr=config.ALPHA_FDR, seeds=None, *,
                            corr_method=config.CORR_METHOD,
                            corr_threshold=config.CORR_THRESHOLD,
                            criterion=config.CRITERION,
                            ascending=config.ASCENDING,
                            n_outer=config.N_OUTER,
                            test_size=config.TEST_SIZE,
                            n_time_points=config.N_TIME_POINTS,
                            empty_policy="chance",
                            null=None,
                            models=None,
                            out_dir=None,
                            verbose=False,
                            preprocess_split=None,
                            univariate_filter=None,
                            train_models=None):
    """Repeated nested CV + single split, both paths, for every seed.

    null="gaussian" swaps the features for N(0,1) noise (same folds and outcomes).
    models runs only a subset. With out_dir each seed is saved and reused on the next call.
    The last three arguments let you inject other functions (default: radiomics_protocol.utils).
    Returns (long, sel).
    """
    config.silence_warnings()
    seeds = list(seeds) if seeds is not None else config.default_seeds()
    models = list(models) if models is not None else list(config.MODELS)
    unknown = set(models) - set(config.MODELS)
    assert not unknown, f"Unknown models: {unknown}"

    y = as_structured(y_structured)
    if not isinstance(X_raw, pd.DataFrame):
        X_raw = pd.DataFrame(X_raw)
    assert len(X_raw) == len(y), "X and y don't have the same number of rows."
    assert np.isfinite(X_raw.values).all(), "X contains NaN/inf: clean before calling."
    assert empty_policy in ("chance", "nan") and null in (None, "gaussian")

    deps = {"preprocess_split": preprocess_split or _default_preprocess,
            "univariate_filter": univariate_filter or univariate_cox_filter_fdr,
            "train_models": train_models or _default_train_models}

    cfg = dict(alpha_fdr=alpha_fdr, corr_method=corr_method, corr_threshold=corr_threshold,
               criterion=criterion, ascending=ascending, n_outer=n_outer, test_size=test_size,
               n_time_points=n_time_points, empty_policy=empty_policy, null=null, models=models)

    # evaluation times: percentiles 10-90 of the event times
    ev_times = y["dias_supervivencia"][y["exitus"]]
    global_times = np.percentile(ev_times, np.linspace(10, 90, n_time_points))

    out_path = Path(out_dir) if out_dir else None
    if out_path:
        out_path.mkdir(parents=True, exist_ok=True)
        cfg_file = out_path / "config.json"
        if cfg_file.exists():
            old = json.loads(cfg_file.read_text())
            if old != json.loads(json.dumps(cfg)):
                raise ValueError(f"{out_dir}/config.json does not match the current parameters:\n"
                                 f"  saved:   {old}\n  current: {cfg}\n"
                                 "Use another out_dir or delete the folder.")
        else:
            cfg_file.write_text(json.dumps(cfg, indent=2))

    all_long, all_sel = [], []
    idx_all = np.arange(len(X_raw))

    for s in seeds:
        f_long = out_path / f"long_seed{s}.csv" if out_path else None
        f_sel = out_path / f"selection_seed{s}.csv" if out_path else None
        if f_long and f_long.exists() and f_sel.exists():
            all_long.append(pd.read_csv(f_long))
            all_sel.append(pd.read_csv(f_sel))
            print(f"[seed {s}] loaded from {out_dir}")
            continue

        t0 = time.time()
        X_s = X_raw
        if null == "gaussian":
            rng = np.random.RandomState(s)
            X_s = pd.DataFrame(rng.randn(*X_raw.shape), columns=X_raw.columns, index=X_raw.index)

        cfg_s = dict(cfg, _seed=s)
        seed_long, seed_sel = [], []

        def _run(scheme, fold, tr, te):
            rows, sel = evaluate_split(X_s.iloc[tr], X_s.iloc[te], y[tr], y[te],
                                       global_times=global_times, cfg=cfg_s, deps=deps,
                                       verbose=verbose)
            for r in rows:
                r.update(seed=s, scheme=scheme, fold=fold)
            seed_long.extend(rows)
            seed_sel.append({"seed": s, "scheme": scheme, "fold": fold, **sel})

        outer = StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=s)
        for k, (tr, te) in enumerate(outer.split(X_s, y["exitus"]), start=1):
            _run("nested", k, tr, te)

        tr, te = train_test_split(idx_all, test_size=test_size, stratify=y["exitus"], random_state=s)
        _run("single", 0, tr, te)

        df_long = pd.DataFrame(seed_long)[LONG_COLUMNS]
        df_sel = pd.DataFrame(seed_sel)
        if f_long:
            df_long.to_csv(f_long, index=False)
            df_sel.to_csv(f_sel, index=False)
        all_long.append(df_long)
        all_sel.append(df_sel)
        print(f"[seed {s}] ready in {(time.time() - t0) / 60:.1f} min", flush=True)

    return pd.concat(all_long, ignore_index=True), pd.concat(all_sel, ignore_index=True)

