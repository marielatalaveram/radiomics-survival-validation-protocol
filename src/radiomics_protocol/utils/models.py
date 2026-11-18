import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import k_fold_cross_validation
from sklearn.model_selection import GridSearchCV, ShuffleSplit
from sksurv.ensemble import GradientBoostingSurvivalAnalysis, RandomSurvivalForest
from sksurv.metrics import concordance_index_censored
from sksurv.svm import FastKernelSurvivalSVM

from .. import config

SEED = config.DEFAULT_SEED


# cox regression (ridge: l1_ratio=0, lasso: l1_ratio=1)
def coxl1(X_train_proc, X_test_proc, y_train_df, y_test_df, l1, seed=SEED):
    penalizers = config.GRIDS["cox_penalizers"]
    best_score, best_p = -np.inf, penalizers[0]
    df_train_all = y_train_df.join(X_train_proc)
    df_train_shuffled = df_train_all.sample(frac=1, random_state=seed).reset_index(drop=True)
    fit_opts = {"step_size": 0.5}

    print("Selecting lambda with 5-fold CV on the training set...")
    for p in penalizers:
        try:
            cph_r = CoxPHFitter(penalizer=p, l1_ratio=l1)
            scores = k_fold_cross_validation(
                cph_r, df_train_shuffled, "dias_supervivencia", "exitus",
                k=5, scoring_method="concordance_index", seed=seed,
                fitter_kwargs={"fit_options": fit_opts},
            )
            avg = np.mean(scores)
            if avg > best_score:
                best_score, best_p = avg, p
        except Exception as e:
            print(f"  penalizer={p:.3f} failed / did not converge: {e}")

    print(f"Best lambda: {best_p:.4f} (CV C-index: {best_score:.3f})")

    cph_all = CoxPHFitter(penalizer=best_p, l1_ratio=l1)
    cph_all.fit(df_train_all, duration_col="dias_supervivencia", event_col="exitus",
                fit_options=fit_opts)

    df_test_all = y_test_df.join(X_test_proc)
    ci_train = cph_all.concordance_index_
    ci_test = cph_all.score(df_test_all, scoring_method="concordance_index")
    print(f"\nCox: train C-index={ci_train:.3f}, test C-index={ci_test:.3f}")

    risk_test = cph_all.predict_partial_hazard(df_test_all).values.ravel()
    results = {
        "Cox (all)": {
            "ci_train": ci_train, "ci_test": ci_test,
            "risk_test": risk_test, "model": cph_all, "type": "lifelines",
            "best_penalizer": best_p,
        }
    }
    return cph_all, results, best_p


# random survival forest
def rsf(X_train_proc, X_test_proc, y_train_struct, y_test_struct, seed=SEED):
    results = {}
    fk_best = None
    try:
        cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=seed)
        rsf_base = RandomSurvivalForest(n_jobs=1, random_state=seed)
        gcv = GridSearchCV(estimator=rsf_base, param_grid=config.GRIDS["rsf"], cv=cv,
                           n_jobs=-1, error_score=np.nan)

        print("Training RSF (searching hyper-parameters)...")
        gcv.fit(X_train_proc.values, y_train_struct)
        print(f"Best combination: {gcv.best_params_}")

        fk_best = gcv.best_estimator_
        ci_tr = fk_best.score(X_train_proc.values, y_train_struct)
        ci_te = fk_best.score(X_test_proc.values, y_test_struct)
        risk_te = fk_best.predict(X_test_proc.values)
        results["RSF (all)"] = {
            "ci_train": ci_tr, "ci_test": ci_te, "risk_test": risk_te,
            "model": fk_best, "type": "sksurv", "best_params": gcv.best_params_,
        }
        print(f"RSF:     C-index train={ci_tr:.3f}, test={ci_te:.3f}")
    except Exception as e:
        print(f"RSF (all) failed: {e}")
    return fk_best, results


# gradient boosting survival analysis
def gbm(X_train_proc, X_test_proc, y_train_struct, y_test_struct, seed=SEED):
    results = {}
    fk_best = None
    try:
        cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=seed)
        gbm_base = GradientBoostingSurvivalAnalysis(random_state=seed)
        gcv = GridSearchCV(gbm_base, config.GRIDS["gbm"], cv=cv, n_jobs=-1, verbose=0,
                           error_score=np.nan)

        print("Training GBM (searching hyper-parameters)...")
        gcv.fit(X_train_proc.values, y_train_struct)
        fk_best = gcv.best_estimator_

        ci_tr = fk_best.score(X_train_proc.values, y_train_struct)
        ci_te = fk_best.score(X_test_proc.values, y_test_struct)
        risk_te = fk_best.predict(X_test_proc.values)
        results["GBM (all)"] = {
            "ci_train": ci_tr, "ci_test": ci_te, "risk_test": risk_te,
            "model": fk_best, "type": "sksurv", "best_params": gcv.best_params_,
        }
        print(f"GBM:     C-index train={ci_tr:.3f}, test={ci_te:.3f}")
    except Exception as e:
        print(f"GBM (all) failed: {e}")
    return fk_best, results


# survival svm (regression mode)
def ssvm(X_train_proc, X_test_proc, y_train_struct, y_test_struct, seed=SEED):
    # predict() returns a survival time, so risk = -predict. The sign is fixed here,
    # it must not be picked with the test C-index.
    results = {}
    try:
        inner_cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=seed)
        fk_base = FastKernelSurvivalSVM(
            rank_ratio=0.0, max_iter=10000, verbose=False, tol=1e-5,
            optimizer="rbtree", fit_intercept=True, random_state=seed,
        )
        gcv = GridSearchCV(estimator=fk_base, param_grid=config.GRIDS["ssvm"], cv=inner_cv,
                           n_jobs=-1, error_score=np.nan)

        print("Training SSVM (searching hyper-parameters)...")
        gcv.fit(X_train_proc.values, y_train_struct)

        fk_best = gcv.best_estimator_
        best_params = gcv.best_params_

        risk_test = -fk_best.predict(X_test_proc.values)           
        ci_test = concordance_index_censored(
            y_test_struct["exitus"], y_test_struct["dias_supervivencia"], risk_test)[0]
        risk_train = -fk_best.predict(X_train_proc.values)
        ci_train = concordance_index_censored(
            y_train_struct["exitus"], y_train_struct["dias_supervivencia"], risk_train)[0]

        results["SSVM (all)"] = {
            "ci_train": ci_train, "ci_test": ci_test, "risk_test": risk_test,
            "model": fk_best, "type": "sksurv_nosurv", "negated": True,
            "best_params": best_params,
        }
        print(f"SSVM:    C-index train={ci_train:.3f}, test={ci_test:.3f}")
        print(f"Best parameters found: {best_params}")
        return fk_best, results
    except Exception as e:
        print(f"SSVM (all) failed: {e}")
        return None, results


# dispatcher
def train_and_evaluate_models(X_train, X_test, y_train_struct, y_test_struct, y_train_df,
                              seed=SEED, models=None):
    # returns {model_name: result}; a model that fails is just missing from the dict
    models = list(models) if models is not None else list(config.MODELS)
    if X_train.shape[1] == 0:
        print("No features selected, nothing to train")
        return {}

    y_test_df = pd.DataFrame({"exitus": y_test_struct["exitus"],
                              "dias_supervivencia": y_test_struct["dias_supervivencia"]},
                             index=X_test.index)
    results = {}

    if "Cox_RIDGE" in models:
        try:
            print("\n--- Running Cox (Ridge) ---")
            _, res, _ = coxl1(X_train, X_test, y_train_df, y_test_df, l1=0.0, seed=seed)
            if "Cox (all)" in res:
                results["Cox_RIDGE"] = res["Cox (all)"]
        except Exception as e:
            print(f"Cox Ridge failed: {e}")
    if "Cox_LASSO" in models:
        try:
            print("\n--- Running Cox (LASSO) ---")
            _, res, _ = coxl1(X_train, X_test, y_train_df, y_test_df, l1=1.0, seed=seed)
            if "Cox (all)" in res:
                results["Cox_LASSO"] = res["Cox (all)"]
        except Exception as e:
            print(f"Cox LASSO failed: {e}")
    if "RSF" in models:
        try:
            print("\n--- Running RSF ---")
            _, res = rsf(X_train, X_test, y_train_struct, y_test_struct, seed=seed)
            if "RSF (all)" in res:
                results["RSF"] = res["RSF (all)"]
        except Exception as e:
            print(f"RSF failed: {e}")
    if "GBM" in models:
        try:
            print("\n--- Running GBM ---")
            _, res = gbm(X_train, X_test, y_train_struct, y_test_struct, seed=seed)
            if "GBM (all)" in res:
                results["GBM"] = res["GBM (all)"]
        except Exception as e:
            print(f"GBM failed: {e}")
    if "SSVM" in models:
        try:
            print("\n--- Running SSVM ---")
            _, res = ssvm(X_train, X_test, y_train_struct, y_test_struct, seed=seed)
            if "SSVM (all)" in res:
                results["SSVM"] = res["SSVM (all)"]
        except Exception as e:
            print(f"SSVM failed: {e}")
    return results
