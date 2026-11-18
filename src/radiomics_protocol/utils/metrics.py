import numpy as np
import pandas as pd
from sksurv.metrics import brier_score, cumulative_dynamic_auc

_trapz = getattr(np, "trapezoid", None) or np.trapz


def metrics_from_result(res, X_te_eval, y_tr, y_te, times):
    # IBS stays NaN for SSVM (no survival function) and when there are < 2 time points
    out = {"ci_test": res.get("ci_test", np.nan), "auc": np.nan, "ibs": np.nan}
    if times is None:
        return out
    try:
        _, out["auc"] = cumulative_dynamic_auc(y_tr, y_te, res["risk_test"], times)
    except Exception:
        pass
    model, mtype = res.get("model"), res.get("type")
    if model is None or mtype == "sksurv_nosurv":
        return out
    try:
        if mtype == "lifelines":
            df_te = pd.DataFrame({"exitus": y_te["exitus"],
                                  "dias_supervivencia": y_te["dias_supervivencia"]},
                                 index=X_te_eval.index).join(X_te_eval)
            sp = model.predict_survival_function(df_te, times=times).T.values
        else:
            sp = np.vstack([fn(times) for fn in model.predict_survival_function(X_te_eval.values)])
        _, bs = brier_score(y_tr, y_te, sp, times)
        out["ibs"] = _trapz(bs, times) / (times[-1] - times[0])
    except Exception:
        pass
    return out
