import logging, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

logger = logging.getLogger(__name__)


def type_definition(df):
    num, cat = [], []
    for f in df.columns:
        (num if pd.api.types.is_numeric_dtype(df[f]) else cat).append(f)
    return num, cat


def get_modalities(df):
    low, rank = [], []
    for f in df.columns:
        n = df[f].nunique()
        (low if n < 2 else rank).append((f, df[f].unique().tolist() if n < 2 else n))
    return tuple(low), sorted(rank, key=lambda x: x[1])


def drop_low_modalities(df, low_info_col):
    return df.drop(columns=[c[0] for c in low_info_col], errors='ignore')


def clean_strings(df, cat_col):
    r = df.copy()
    for c in cat_col:
        if c in r.columns:
            r[c] = (r[c].astype(str).apply(lambda x: ','.join(s.strip() for s in x.split(',')).lower())
                     .str.replace(r'\(.*?\)', '', regex=True).str.strip())
    return r


def find_error_col(df, col, errors):
    return df.drop(index=df[df[col].isin(errors)].index)


def keep_value_col(df, col, values):
    return df[df[col].isin(values)].copy()


def keep_unique(df, pkey, keep='first'):
    return df[~df.duplicated(subset=pkey, keep=keep)].copy()


def get_duplicate(df, pkey, keep='first'):
    return int(df.duplicated(subset=pkey, keep=keep).sum())


def fill_na_values(df, na_filling_rules):
    r = df.copy()
    for feat, rule in na_filling_rules.items():
        if feat in r.columns:
            r[feat] = r[feat].fillna(rule(r))
    return r


def conditional_fill_na(df, rules):
    r = df.copy()
    for col, rule_list in rules.items():
        for cond_fn, val in rule_list:
            m = r[col].isna() & cond_fn(r)
            r[col] = np.where(m, val, r[col])
    return r


def check_data(df, target_column):
    if target_column not in df.columns:
        raise ValueError(f"Target '{target_column}' not found.")
    if df.isnull().sum().sum() > 0:
        logger.warning("Data contains missing values.")


def _validate_target(df, target_column):
    if target_column not in df.columns:
        raise ValueError(f"Target '{target_column}' not found.")
    if not pd.api.types.is_numeric_dtype(df[target_column]):
        raise ValueError(f"Target '{target_column}' must be numeric.")


def _drop_na_target(df, target_column):
    df_c = df.dropna(subset=[target_column])
    if len(df_c) < 2:
        raise ValueError("Not enough samples after dropping missing target.")
    return df_c


def split_data(df, target_column, test_size, random_state):
    _validate_target(df, target_column)
    df_c = _drop_na_target(df, target_column)
    X, y = df_c.drop(target_column, axis=1), df_c[target_column]
    X_num = X.select_dtypes(include=np.number)
    if X_num.shape[1] == 0:
        raise ValueError("No numeric features found.")
    X_tr, X_te, y_tr, y_te = train_test_split(X_num, y, test_size=test_size, random_state=random_state)
    return X_tr, X_te, y_tr, y_te, list(X_num.columns)


def split_data_cat(df, target_column, test_size, random_state):
    X_tr, X_te, y_tr, y_te, fnames = split_data(df, target_column, test_size, random_state)
    if X_tr.isnull().sum().sum() > 0 or X_te.isnull().sum().sum() > 0:
        imp = X_tr.median()
        X_tr, X_te = X_tr.fillna(imp), X_te.fillna(imp)
    return X_tr, X_te, y_tr, y_te, fnames


def split_and_select_numeric_data(df, target_col, test_size=0.2, random_state=42):
    _validate_target(df, target_col)
    df_c = _drop_na_target(df, target_col)
    X, y = df_c.drop(target_col, axis=1), df_c[target_col]
    X_num = X.select_dtypes(include=np.number)
    fnames = list(X_num.columns)
    if not fnames:
        raise ValueError("No numeric features found.")
    X_tr, X_te, y_tr, y_te = train_test_split(X_num, y, test_size=test_size, random_state=random_state)
    if X_tr.isnull().sum().sum() > 0 or X_te.isnull().sum().sum() > 0:
        imp = X_tr.median()
        X_tr, X_te = X_tr.fillna(imp), X_te.fillna(imp)
    return X_tr, X_te, y_tr, y_te, fnames, X_te.index.copy(), X_tr.index.copy()


def onehot_encode_column(df, column_name):
    enc = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    enc.fit(df[[column_name]])
    edf = pd.DataFrame(enc.transform(df[[column_name]]),
                        columns=enc.get_feature_names_out([column_name]), index=df.index)
    return pd.concat([df, edf], axis=1).drop(columns=[column_name])


def bin_categories(df, features=None, cutoff=0.007, replace_with='other_grouped'):
    r = df.copy()
    if features is None:
        features = [c for c in r.columns if not pd.api.types.is_numeric_dtype(r[c])]
    for f in features:
        if f not in r.columns or pd.api.types.is_numeric_dtype(r[f]):
            continue
        vc = r[f].value_counts()
        r.loc[r[f].isin(vc[vc / len(r) < cutoff].index), f] = replace_with
    return r


def encode_by_value(df, label_cols, value_col_prefix):
    r = df.copy()
    for col in label_cols:
        if col not in r.columns:
            continue
        val_col = f"{value_col_prefix}_{col}"
        if val_col not in r.columns:
            continue
        for name in r[col].dropna().unique():
            m = r[col] == name
            r[f"{col}_{name}"] = 0
            r.loc[m, f"{col}_{name}"] = r.loc[m, val_col]
    return r


def sum_grouped_columns(df, prefix, group_pos=1):
    r = df.copy()
    group_names = set()
    for c in r.columns:
        p = c.split("_")
        if len(p) > group_pos:
            group_names.add(p[group_pos])
    for g in group_names:
        cols = [c for c in r.columns if f"_{g}_" in c]
        if cols:
            r[f"{g}_total"] = r[cols].sum(axis=1)
    return r


def outlier_stat(df, num_col, cat_col):
    out = pd.DataFrame(index=df.columns)
    for f in num_col:
        q1, q3 = df[f].quantile(0.25), df[f].quantile(0.75)
        iqr = q3 - q1
        mu, std = df[f].mean(), df[f].std()
        out.loc[f, 'outlier_max_iqr'] = q3 + 1.5 * iqr
        out.loc[f, 'outlier_min_iqr'] = max(q1 - 1.5 * iqr, 0)
        out.loc[f, 'outlier_max_zscore'] = mu + 3 * std
        out.loc[f, 'outlier_min_zscore'] = mu - 3 * std
        out.loc[f, 'IQR_OUT_NB'] = ((df[f] < q1 - 1.5 * iqr) | (df[f] > q3 + 1.5 * iqr)).sum()
        out.loc[f, 'Z_OUT_NB'] = ((df[f] < mu - 3 * std) | (df[f] > mu + 3 * std)).sum()
    for f in cat_col:
        ms = df[f].mode()
        if not ms.empty:
            m = ms.iloc[0]
            out.loc[f, 'mode'] = m
            out.loc[f, 'mode_occurrence'] = int(df[f].value_counts().get(m, 0))
    return out.reindex(index=num_col + cat_col)


def remove_z_outlier(df, num_col):
    idxs = []
    for f in num_col:
        mu, std = df[f].mean(), df[f].std()
        idxs.extend(df[(df[f] < mu - 3 * std) | (df[f] > mu + 3 * std)].index.tolist())
    return df.drop(index=list(set(idxs)))


def remove_1percent_outliers(df, num_col):
    idxs = []
    for f in num_col:
        idxs.extend(df[df[f] > df[f].quantile(0.99)].index.tolist())
    return df.drop(index=list(set(idxs)))


if __name__ == "__main__":
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": ["x", "y", "z"], "t": [10, 20, 30]})
    num, cat = type_definition(df)
    assert num == ["a", "t"] and cat == ["b"]
    low, rank = get_modalities(df)
    assert len(low) == 0 and len(rank) == 3
    df2 = drop_low_modalities(df, ())
    assert list(df2.columns) == list(df.columns)
    assert clean_strings(df, ["b"])["b"].tolist() == ["x", "y", "z"]
    assert len(find_error_col(df, "b", ["x"])) == 2
    assert len(keep_value_col(df, "b", ["x", "y"])) == 2
    dups = pd.concat([df, df.iloc[[0]]])
    assert len(keep_unique(dups, "a")) == 3
    assert get_duplicate(dups, "a") == 1
    assert get_duplicate(df, "a") == 0
    check_data(df, "t")
    try:
        check_data(df, "x")
        assert False
    except ValueError:
        pass
    X_tr, X_te, y_tr, y_te, fn = split_data(df, "t", 0.33, 42)
    assert len(X_tr) > 0 and set(fn) == {"a"}
    X_tr2, X_te2, y_tr2, y_te2, fn2 = split_data_cat(df, "t", 0.33, 42)
    assert len(X_tr2) > 0
    r = split_and_select_numeric_data(df, "t", 0.33, 42)
    assert len(r) == 7
    enc = onehot_encode_column(df, "b")
    assert "b" not in enc.columns and "b_x" in enc.columns
    bc = bin_categories(pd.DataFrame({"a": ["x"] * 90 + ["y"] * 10, "t": range(100)}), cutoff=0.15)
    assert bc["a"].value_counts().get("other_grouped", 0) == 10
    out = outlier_stat(df, ["a", "t"], ["b"])
    assert "IQR_OUT_NB" in out.columns and "mode" in out.columns
    assert len(remove_z_outlier(df, ["a"])) <= 3
    assert len(remove_1percent_outliers(pd.DataFrame({"a": range(100), "b": range(100)}), ["a"])) <= 99
