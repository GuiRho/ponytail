import logging, numpy as np, pandas as pd, scipy.stats as ss
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.preprocessing import PowerTransformer, RobustScaler, StandardScaler

logger = logging.getLogger(__name__)


def cramers_v(ct):
    chi2 = ss.chi2_contingency(ct)[0]
    n = ct.sum().sum()
    phi2 = chi2 / n
    r, k = ct.shape
    phi2c = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rc = r - (r - 1) ** 2 / (n - 1)
    kc = k - (k - 1) ** 2 / (n - 1)
    return float(np.sqrt(phi2c / min(kc - 1, rc - 1)))


def print_top_correlations(dataset, n=5, threshold=0.85):
    numerical = dataset.select_dtypes(include=np.number).columns.tolist()
    if not numerical:
        return [], set()
    corr = dataset[numerical].corr()
    high, added = [], set()
    for f in numerical:
        c = corr[f].drop(f, errors='ignore')
        for o, _ in c.abs().sort_values(ascending=False).head(n).items():
            p = tuple(sorted((f, o)))
            if abs(c[o]) > threshold and p not in added:
                high.append((f, o, c[o]))
                added.add(p)
    return high, added


def corr_matrix(df):
    _, ax = plt.subplots(figsize=(18, 14))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f",
                 linewidths=0.5, annot_kws={"size": 6}, ax=ax)
    plt.show()


def analyse_cat_cat(df, alpha=0.05):
    cat = df.select_dtypes(exclude=np.number).columns.tolist()
    if not cat:
        return None
    results = []
    for i in range(len(cat)):
        for j in range(i + 1, len(cat)):
            c1, c2 = cat[i], cat[j]
            if df[c1].value_counts().iloc[0] < 15 or df[c2].value_counts().iloc[0] < 15:
                continue
            try:
                ct = pd.crosstab(df[c1], df[c2])
                stat, p, dof, exp = ss.chi2_contingency(ct)
                if p < alpha:
                    results.append({"col1": c1, "col2": c2, "p_value": p, "cramers_v": cramers_v(ct)})
            except Exception:
                pass
    return pd.DataFrame(results) if results else None


def bar_chart(df, feature, target_variable, roundto=4, p_threshold=0.05,
              sig_ttest_only=True, min_group_size=2, max_t_tests=5):
    _, ax = plt.subplots(figsize=(10, 6))
    if pd.api.types.is_numeric_dtype(df[feature]):
        cat_col, num_col = target_variable, feature
    else:
        cat_col, num_col = feature, target_variable
    sns.barplot(x=cat_col, y=num_col, data=df, ci=None, ax=ax)
    groups, valid, lists = df[cat_col].unique(), [], []
    for g in groups:
        d = df[df[cat_col] == g][num_col]
        if len(d) >= min_group_size:
            lists.append(d)
            valid.append(g)
    f_stat, p_val = ss.f_oneway(*lists) if len(lists) >= 2 else (np.nan, np.nan)
    ttests, nc = [], 0
    for i1, g1 in enumerate(valid):
        for i2 in range(i1 + 1, len(valid)):
            l1, l2 = df[df[cat_col] == g1][num_col], df[df[cat_col] == g2][num_col]
            t, tp = ss.ttest_ind(l1, l2)
            ttests.append([f"{g1} - {g2}", round(t, roundto), round(tp, roundto)])
            nc += 1
    bonf = p_threshold / nc if nc else np.nan
    ttests.sort(key=lambda x: abs(x[1]) if not np.isnan(x[1]) else 0, reverse=True)
    text = f"ANOVA:\nF = {round(f_stat, roundto)}\np = {round(p_val, roundto)}\nBonferroni p: {round(bonf, roundto)}\n"
    sc = 0
    for t in ttests[:max_t_tests]:
        if not np.isnan(bonf) and t[2] <= bonf:
            text += f"\n{t[0]}: t:{t[1]}, p:{t[2]}"
            sc += 1
        elif not sig_ttest_only:
            text += f"\n{t[0]}: t:{t[1]}, p:{t[2]}"
    if sig_ttest_only and sc == 0:
        text += "\nNo significant t-tests"
    if df[feature].nunique() > 7:
        plt.setp(ax.get_xticklabels(), rotation=90)
    ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=10, verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.5", edgecolor="black", facecolor="white", alpha=0.8))
    plt.tight_layout()
    plt.show()


def _corr(s, target):
    if s.isnull().all() or target.isnull().all() or s.nunique() <= 1:
        return np.nan
    try:
        return float(s.replace([np.inf, -np.inf], np.nan).corr(target))
    except Exception:
        return np.nan


def _power(s, p):
    try:
        if p < 0:
            return s.where(s == 0, s[s != 0] ** p).where(s != 0, np.nan)
        if p < 1:
            return s.where(s <= 0, s[s > 0] ** p).where(s > 0, np.nan)
        return (s ** p).replace([np.inf, -np.inf], np.nan)
    except (TypeError, ValueError):
        return pd.Series(np.nan, index=s.index)


def _log(s, f):
    try:
        return s.where(s <= 0, f(s[s > 0])).where(s > 0, np.nan)
    except (TypeError, ValueError):
        return pd.Series(np.nan, index=s.index)


def _scale(s, sc):
    try:
        sc.fit(s.values.reshape(-1, 1))
        return pd.Series(sc.transform(s.values.reshape(-1, 1)).flatten(), index=s.index)
    except Exception:
        return pd.Series(np.nan, index=s.index)


TRANSFORMATIONS = {
    "power_0.25": lambda s: _power(s, 0.25),
    "power_0.33": lambda s: _power(s, 1 / 3),
    "power_0.50": lambda s: _power(s, 0.5),
    "power_2": lambda s: _power(s, 2),
    "power_5": lambda s: _power(s, 5),
    "log2": lambda s: _log(s, np.log2),
    "log10": lambda s: _log(s, np.log10),
    "standard_scale": lambda s: _scale(s, StandardScaler()),
    "robust_scale": lambda s: _scale(s, RobustScaler()),
    "yeo_johnson": lambda s: _scale(s, PowerTransformer(method="yeo-johnson", standardize=True)),
}


def apply_and_test_all(current_series, target_series, transformations=None):
    return {n: _corr(f(current_series), target_series) for n, f in (transformations or TRANSFORMATIONS).items()}


def analyze_iterative_transformations(df, target, max_turns=5, min_improvement=5e-4):
    if target not in df.columns or not pd.api.types.is_numeric_dtype(df[target]) or df[target].isnull().all():
        return None, None
    num = [c for c in df.select_dtypes(include=np.number).columns if c != target]
    if not num:
        return None, None
    ts = df[target].dropna()
    dn = df[num].loc[ts.index]
    if dn.empty:
        return None, None
    states, hist = {}, {}
    for c in num:
        init = _corr(dn[c], ts) or 0.0
        states[c] = {"cur": dn[c], "best_abs": abs(init), "best_sig": init, "stop": False}
        hist[c] = [("original", init)]
    for _ in range(max_turns):
        improved = 0
        for c in num:
            s = states[c]
            if s["stop"]:
                continue
            corrs = apply_and_test_all(s["cur"], ts)
            best = max(((n, v) for n, v in corrs.items() if pd.notna(v)),
                        key=lambda x: abs(x[1]), default=(None, np.nan))
            if best[0] and abs(best[1]) > s["best_abs"] + min_improvement:
                improved += 1
                s["cur"] = TRANSFORMATIONS[best[0]](s["cur"])
                s["best_abs"], s["best_sig"] = abs(best[1]), best[1]
                hist[c].append((best[0], best[1]))
            else:
                s["stop"] = True
        if not improved:
            break
    summary = pd.DataFrame(
        [(c, hist[c][0][1], states[c]["best_sig"], len(hist[c]) - 1) for c in num],
        columns=["Feature", "Initial Correlation", "Final Correlation", "Num Transformations"]
    ).set_index("Feature")
    hc = [f"{p}_{i}" for i in range(1, max_turns + 1) for p in ("Transform", "Corr")]
    hd = {}
    for c in num:
        hd[c] = {}
        for i in range(1, max_turns + 1):
            hd[c][f"Transform_{i}"] = hist[c][i][0] if i < len(hist[c]) else None
            hd[c][f"Corr_{i}"] = hist[c][i][1] if i < len(hist[c]) else np.nan
    return summary, pd.DataFrame.from_dict(hd, orient="index").reindex(columns=hc)


def create_transformed_dataframe(df, history_results):
    r = df.copy()
    for feat in history_results.index:
        if feat not in r.columns:
            continue
        cur = r[feat].copy()
        for i in range(1, 6):
            tn = history_results.loc[feat, f"Transform_{i}"]
            if pd.isna(tn) or tn in (None, "original"):
                break
            fn = TRANSFORMATIONS.get(tn)
            if fn:
                cur = fn(cur)
        r[feat] = cur
    return r


if __name__ == "__main__":
    import pytest
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0], "c": ["x", "y", "z"], "t": [10, 20, 30]})
    high, added = print_top_correlations(df, n=2, threshold=0.0)
    assert len(high) > 0
    assert analyse_cat_cat(df) is None or isinstance(analyse_cat_cat(df), pd.DataFrame)
    ct = pd.crosstab(pd.Series(["a", "a", "a", "b", "b", "b", "c", "c", "c"]),
                      pd.Series(["a", "a", "a", "b", "b", "b", "c", "c", "c"]))
    assert abs(cramers_v(ct) - 1.0) < 0.01
    summary, history = analyze_iterative_transformations(df, "t", max_turns=2)
    assert summary is not None and history is not None
    tdf = create_transformed_dataframe(df, history)
    assert "a" in tdf.columns
