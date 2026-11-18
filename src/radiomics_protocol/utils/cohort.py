import pandas as pd

AGE_ORDER = ["<55", "55-65", "65-75", ">75"]


def _n_pct(n: int, total: int) -> str:
    return f"{n} ({100 * n / total:.1f}%)" if total > 0 else f"{n} (nan%)"


def _median_iqr(s: pd.Series) -> str:
    return f"{s.median():.1f} ({s.quantile(0.25):.1f}–{s.quantile(0.75):.1f})"


def cohort_table(df):
    """Table 1: overall / events / censored."""
    df = df.copy()
    df["exitus"] = df["exitus"].astype(bool)
    groups = {
        "Overall": df,
        "Events": df[df["exitus"]],
        "Censored": df[~df["exitus"]],
    }
    n_total = len(df)
    rows = {}

    def add(label, fn):
        rows[label] = {name: fn(g) for name, g in groups.items()}

    add("Study cohort, N (%)", lambda g: _n_pct(len(g), n_total))
    for sex in sorted(df["sexo"].dropna().unique()):
        add(f"Sex: {sex}, n (%)", lambda g, s=sex: _n_pct(int((g["sexo"] == s).sum()), len(g)))
    add("Age (years), median (IQR)", lambda g: _median_iqr(g["age_at_diag"]))

    observed = [a for a in df["age_ranges"].dropna().unique()]
    ordered = [a for a in AGE_ORDER if a in observed] + sorted(a for a in observed if a not in AGE_ORDER)
    for a in ordered:
        add(f"Age group {a}, n (%)", lambda g, a=a: _n_pct(int((g["age_ranges"] == a).sum()), len(g)))
    add("Overall survival time (days), median (IQR)", lambda g: _median_iqr(g["dias_supervivencia"]))

    out = pd.DataFrame(rows).T
    out.columns = [f"{c} (N={len(groups[c])})" for c in out.columns]
    return out
