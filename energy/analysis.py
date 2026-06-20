import logging, itertools, numpy as np, pandas as pd, scipy.stats as ss, statsmodels.api as sm
import matplotlib.pyplot as plt, seaborn as sns

logger = logging.getLogger(__name__)


def univariate_analysis(df):
    out = pd.DataFrame(index=df.columns)
    out.index.name = "feature"
    out["type"] = df.dtypes
    out["count"] = df.count()
    out["missing"] = df.isna().sum()
    out["unique"] = df.nunique()
    try:
        out["mode"] = df.astype(str).mode().iloc[0]
    except Exception:
        out["mode"] = "N/A"
    num = df.select_dtypes(include=np.number).columns
    if not num.empty:
        ns = df[num].agg(["min", "mean", "max", "std", "skew", "kurt",
                           lambda x: x.quantile(0.25), lambda x: x.quantile(0.5), lambda x: x.quantile(0.75)]).T
        ns.columns = ["min", "mean", "max", "std", "skew", "kurt", "q1", "median", "q3"]
        out = out.combine_first(ns)
    plot_histogram(df)
    plot_bar_chart(df)
    return out


def plot_na_distribution(distrib_na_row=None, totalna_row=None):
    if distrib_na_row is None:
        logger.error("distrib_na_row is None.")
        return
    plt.figure(figsize=(14, 6))
    bars = plt.bar(distrib_na_row.index, distrib_na_row.values, color="blue")
    plt.xlabel("Number of NaN Values per Row")
    plt.ylabel("Number of Rows")
    plt.title("Distribution of NA Values per Row")
    plt.xticks(distrib_na_row.index)
    plt.ylim(0, distrib_na_row.values.max() * 1.1)
    for b in bars:
        plt.text(b.get_x() + b.get_width() / 2, b.get_height(), int(b.get_height()), ha="center", va="bottom")
    plt.tight_layout()
    plt.show()


def plot_correlation_matrix(df):
    plt.figure(figsize=(18, 14))
    sns.heatmap(df.select_dtypes(include=np.number).corr(), annot=True, cmap="coolwarm",
                fmt=".2f", linewidths=0.5, annot_kws={"size": 6}, cbar_kws={"shrink": 0.7})
    plt.title("Correlation Matrix", fontsize=16)
    plt.tight_layout()
    plt.show()


def plot_histogram(dataset):
    for f in dataset.select_dtypes(include=np.number).columns:
        plt.figure(figsize=(12, 6))
        ax = sns.histplot(dataset[f], bins=30, kde=True, color="blue")
        total, max_h = len(dataset[f]), 0
        for p in ax.patches:
            h = p.get_height()
            if h > 0:
                ax.text(p.get_x() + p.get_width() / 2, h, f"{100 * h / total:.1f}%", ha="center", va="bottom")
            max_h = max(max_h, h)
        plt.title(f"Histogram of {f}")
        plt.xlabel(f)
        plt.ylabel("Number of Rows")
        ax.set_ylim(0, max_h * 1.05)
        plt.show()


def plot_bar_chart(dataset):
    for f in dataset.select_dtypes(exclude=np.number).columns:
        vc = dataset[f].value_counts().nlargest(10)
        if vc.max() < 20:
            continue
        total = len(dataset[f])
        plt.figure(figsize=(10, 6))
        bars = plt.bar(range(len(vc)), vc.values)
        plt.xlabel(f)
        plt.ylabel("Count")
        plt.title(f"Bar Graph of Top 10 Values for {f}")
        plt.xticks(range(len(vc)), [str(l)[:25] for l in vc.index], rotation=45, ha="right")
        for b in bars:
            y = b.get_height()
            plt.text(b.get_x() + b.get_width() / 2, y + vc.max() * 0.01, f"{int(y)}\n({100 * y / total:.1f}%)", ha="center", va="bottom")
        plt.tight_layout()
        plt.show()


def run_ols_with_two_features(df, target_col, feat_col):
    fnames = [c for c in feat_col if c != target_col and pd.api.types.is_numeric_dtype(df[c])]
    if len(fnames) < 2:
        return []
    results = []
    for f1, f2 in itertools.combinations(fnames, 2):
        try:
            results.append({"feature1": f1, "feature2": f2, "r_squared": sm.OLS(df[target_col], sm.add_constant(df[[f1, f2]])).fit().rsquared})
        except Exception as e:
            logger.error("OLS error %s, %s: %s", f1, f2, e)
    return results


def run_ols_with_three_features(df, target_col, feat_col):
    fnames = [c for c in feat_col if c != target_col and pd.api.types.is_numeric_dtype(df[c])]
    if len(fnames) < 3:
        return []
    results = []
    for f1, f2, f3 in itertools.combinations(fnames, 3):
        try:
            results.append({"feature1": f1, "feature2": f2, "feature3": f3, "r_squared": sm.OLS(df[target_col], sm.add_constant(df[[f1, f2, f3]])).fit().rsquared})
        except Exception as e:
            logger.error("OLS error %s, %s, %s: %s", f1, f2, f3, e)
    return results


def compare_r_squared_two(results):
    for i, r in enumerate(sorted(results, key=lambda x: x["r_squared"], reverse=True)):
        logger.info("Result %d: (%s, %s) R²=%.4f", i, r["feature1"], r["feature2"], r["r_squared"])


def compare_r_squared_three(results):
    for i, r in enumerate(sorted(results, key=lambda x: x["r_squared"], reverse=True)):
        logger.info("Result %d: (%s, %s, %s) R²=%.4f", i, r["feature1"], r["feature2"], r["feature3"], r["r_squared"])


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    df = pd.DataFrame({"a": rng.normal(0, 1, 30), "b": rng.normal(0, 1, 30), "c": ["x"] * 15 + ["y"] * 15, "t": rng.normal(10, 2, 30)})
    ua = pd.DataFrame(index=df.columns)
    ua["count"] = df.count()
    ua["missing"] = df.isna().sum()
    ua["unique"] = df.nunique()
    assert list(ua.index) == ["a", "b", "c", "t"]
    two = run_ols_with_two_features(df, "t", ["a", "b"])
    assert len(two) == 1
    df3 = pd.DataFrame({"a": range(100), "b": range(100, 200), "c": range(200, 300), "t": range(300, 400)})
    three = run_ols_with_three_features(df3, "t", ["a", "b", "c"])
    assert len(three) == 0
    df2 = pd.DataFrame({"x": range(100), "z": range(100, 200), "t": range(200, 300)})
    two2 = run_ols_with_two_features(df2, "t", ["x", "z"])
    assert len(two2) == 1
    assert all("r_squared" in r for r in two2)
