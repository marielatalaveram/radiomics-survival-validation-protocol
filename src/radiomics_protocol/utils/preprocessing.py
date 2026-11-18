import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests


def as_structured(y):
    if isinstance(y, pd.DataFrame):
        arr = np.empty(len(y), dtype=[("exitus", bool), ("dias_supervivencia", float)])
        arr["exitus"] = y["exitus"].astype(bool).values
        arr["dias_supervivencia"] = y["dias_supervivencia"].astype(float).values
        return arr
    return y


def structured_to_frame(y, index):
    return pd.DataFrame({"exitus": y["exitus"], "dias_supervivencia": y["dias_supervivencia"]},
                        index=index)


def preprocess_split(X_train_raw, X_test_raw):
    # scaler fitted on train only
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)
    X_train_df = pd.DataFrame(X_train_scaled, columns=X_train_raw.columns, index=X_train_raw.index)
    X_test_df = pd.DataFrame(X_test_scaled, columns=X_test_raw.columns, index=X_test_raw.index)
    return X_train_df, X_test_df, scaler


def univariate_cox_filter_fdr(X_train, y_train_df, alpha):
    """Univariate Cox per feature + Benjamini-Hochberg. Features whose fit fails are skipped."""
    results, p_values = [], []
    for col in X_train.columns:
        try:
            cph = CoxPHFitter()
            data = y_train_df[["dias_supervivencia", "exitus"]].join(X_train[[col]])
            cph.fit(data, duration_col="dias_supervivencia", event_col="exitus")
            s = cph.summary.loc[col]
            ph_results = proportional_hazard_test(cph, data, time_transform="rank")
            p_schoenfeld = ph_results.summary.loc[col, "p"]
            p_values.append(s["p"])
            results.append({
                "variable": col, "coef": s["coef"], "HR": s["exp(coef)"],
                "HR_lower95": s["exp(coef) lower 95%"], "HR_upper95": s["exp(coef) upper 95%"],
                "p_raw": s["p"],
                "concordance": cph.concordance_index_,
                "p_schoenfeld": p_schoenfeld,
            })
        except Exception:
            pass

    if len(p_values) == 0:
        return [], pd.DataFrame()

    rejected, p_adj, _, _ = multipletests(p_values, method="fdr_bh", alpha=alpha)
    df_results = pd.DataFrame(results)
    df_results["p_adj_fdr"] = p_adj
    df_results["significant_fdr"] = rejected
    df_results = df_results.sort_values("p_adj_fdr")
    selected = df_results.loc[df_results["significant_fdr"], "variable"].tolist()
    return selected, df_results


def correlation_filter(X, threshold, df_cox, corr_method, criterion, ascending):
    """Cluster features by Spearman distance, keep the best one (by `criterion`) of each cluster."""
    if X.shape[1] <= 1:
        return list(X.columns)
    corr = X.corr(method="spearman").values
    dist = 1 - np.abs(corr)
    np.fill_diagonal(dist, 0)
    dist = np.clip((dist + dist.T) / 2, 0, None)
    Z = linkage(squareform(dist, checks=False), method=corr_method)
    clusters = fcluster(Z, t=1 - threshold, criterion="distance")
    scores = df_cox.set_index("variable")[criterion]
    selected = []
    for cl in np.unique(clusters):
        cols = X.columns[clusters == cl]
        s = scores.loc[cols]
        selected.append(s.idxmin() if ascending else s.idxmax())
    return selected
