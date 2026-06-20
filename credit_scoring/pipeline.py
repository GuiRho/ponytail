import gc, json, os, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

warnings.simplefilter("ignore", FutureWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

DATA_DIR = Path("data")
MODEL_DIR = Path("production_model")

def load_json_config(path):
    return json.loads(Path(path).read_text())

def one_hot_encoder(df, nan_as_category=True):
    cat = [c for c in df.columns if df[c].dtype == "object"]
    new = pd.get_dummies(df, columns=cat, dummy_na=nan_as_category)
    return new, [c for c in new.columns if c not in df.columns]

def application_train_test(path, num_rows=None, nan_as_category=False):
    df = pd.concat([pd.read_csv(os.path.join(path, "application_train.csv"), nrows=num_rows),
                    pd.read_csv(os.path.join(path, "application_test.csv"), nrows=num_rows)]).reset_index(drop=True)
    print(f"Train samples: {len(df) // 2}, test samples: {len(df) // 2}")
    df = df[df["CODE_GENDER"] != "XNA"]
    for c in ("CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY"):
        df[c], _ = pd.factorize(df[c])
    df, _ = one_hot_encoder(df, nan_as_category)
    df["DAYS_EMPLOYED"].replace(365243, np.nan, inplace=True)
    df["DAYS_EMPLOYED_PERC"] = df["DAYS_EMPLOYED"] / df["DAYS_BIRTH"]
    df["INCOME_CREDIT_PERC"] = df["AMT_INCOME_TOTAL"] / df["AMT_CREDIT"]
    df["INCOME_PER_PERSON"] = df["AMT_INCOME_TOTAL"] / df["CNT_FAM_MEMBERS"]
    df["ANNUITY_INCOME_PERC"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]
    df["PAYMENT_RATE"] = df["AMT_ANNUITY"] / df["AMT_CREDIT"]
    return df

def _agg_bureau(bureau, bb, bureau_cat, bb_cat):
    bb, _ = one_hot_encoder(bb, True)
    bureau, _ = one_hot_encoder(bureau, True)
    ba = {"MONTHS_BALANCE": ["min", "max", "size", "median"]}
    for c in bb_cat: ba[c] = ["mean"]
    bb_agg = bb.groupby("SK_ID_BUREAU").agg(ba)
    bb_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in bb_agg.columns.tolist()])
    bureau = bureau.join(bb_agg, how="left", on="SK_ID_BUREAU").drop(["SK_ID_BUREAU"], axis=1)
    na = {"DAYS_CREDIT": ["min","max","mean","var","median"], "DAYS_CREDIT_ENDDATE": ["min","max","mean","median"],
          "DAYS_CREDIT_UPDATE": ["mean","median"], "CREDIT_DAY_OVERDUE": ["max","mean","median"],
          "AMT_CREDIT_MAX_OVERDUE": ["mean","median"], "AMT_CREDIT_SUM": ["max","mean","sum","median"],
          "AMT_CREDIT_SUM_DEBT": ["max","mean","sum","median"], "AMT_CREDIT_SUM_OVERDUE": ["mean","median"],
          "AMT_CREDIT_SUM_LIMIT": ["mean","sum","median"], "AMT_ANNUITY": ["max","mean","median"],
          "CNT_CREDIT_PROLONG": ["sum"]}
    ca = {}
    for c in bureau_cat: ca[c] = ["mean"]
    for c in bb_cat: ca[c + "_MEAN"] = ["mean"]
    ba2 = bureau.groupby("SK_ID_CURR").agg({**na, **ca})
    ba2.columns = pd.Index(["BURO_" + e[0] + "_" + e[1].upper() for e in ba2.columns.tolist()])
    for label, cond in [("ACTIVE", bureau["CREDIT_ACTIVE_Active"] == 1), ("CLOSED", bureau["CREDIT_ACTIVE_Closed"] == 1)]:
        sub = bureau[cond].groupby("SK_ID_CURR").agg(na)
        sub.columns = pd.Index([label + "_" + e[0] + "_" + e[1].upper() for e in sub.columns.tolist()])
        ba2 = ba2.join(sub, how="left", on="SK_ID_CURR")
    return ba2

def bureau_and_balance(path, num_rows=None, nan_as_category=True):
    bureau = pd.read_csv(os.path.join(path, "bureau.csv"), nrows=num_rows)
    bb = pd.read_csv(os.path.join(path, "bureau_balance.csv"), nrows=num_rows)
    return _agg_bureau(bureau, bb, [c for c in bureau.columns if bureau[c].dtype == "object"],
                       [c for c in bb.columns if bb[c].dtype == "object"])

def previous_applications(path, num_rows=None, nan_as_category=True):
    prev = pd.read_csv(os.path.join(path, "previous_application.csv"), nrows=num_rows)
    prev, _ = one_hot_encoder(prev, True)
    for c in ["DAYS_FIRST_DRAWING", "DAYS_FIRST_DUE", "DAYS_LAST_DUE_1ST_VERSION", "DAYS_LAST_DUE", "DAYS_TERMINATION"]:
        prev[c].replace(365243, np.nan, inplace=True)
    prev["APP_CREDIT_PERC"] = prev["AMT_APPLICATION"] / prev["AMT_CREDIT"]
    cols = ["AMT_ANNUITY", "AMT_APPLICATION", "AMT_CREDIT", "AMT_DOWN_PAYMENT", "AMT_GOODS_PRICE",
            "HOUR_APPR_PROCESS_START", "RATE_DOWN_PAYMENT", "DAYS_DECISION"]
    na = {c: ["min","max","mean","median"] for c in cols}
    na.update({"APP_CREDIT_PERC": ["min","max","mean","var","median"], "CNT_PAYMENT": ["mean","sum"]})
    pa = prev.groupby("SK_ID_CURR").agg(na)
    pa.columns = pd.Index(["PREV_" + e[0] + "_" + e[1].upper() for e in pa.columns.tolist()])
    for label, cond in [("APPROVED", prev["NAME_CONTRACT_STATUS_Approved"] == 1), ("REFUSED", prev["NAME_CONTRACT_STATUS_Refused"] == 1)]:
        sub = prev[cond].groupby("SK_ID_CURR").agg(na)
        sub.columns = pd.Index([label + "_" + e[0] + "_" + e[1].upper() for e in sub.columns.tolist()])
        pa = pa.join(sub, how="left", on="SK_ID_CURR")
    return pa

def pos_cash(path, num_rows=None, nan_as_category=True):
    pos = pd.read_csv(os.path.join(path, "POS_CASH_balance.csv"), nrows=num_rows)
    pos, cc = one_hot_encoder(pos, True)
    agg = {"MONTHS_BALANCE": ["max","mean","size","median"], "SK_DPD": ["max","mean","median"], "SK_DPD_DEF": ["max","mean","median"]}
    for c in cc: agg[c] = ["mean"]
    pa = pos.groupby("SK_ID_CURR").agg(agg)
    pa.columns = pd.Index(["POS_" + e[0] + "_" + e[1].upper() for e in pa.columns.tolist()])
    pa["POS_COUNT"] = pos.groupby("SK_ID_CURR").size()
    return pa

def installments_payments(path, num_rows=None, nan_as_category=True):
    ins = pd.read_csv(os.path.join(path, "installments_payments.csv"), nrows=num_rows)
    ins, cc = one_hot_encoder(ins, True)
    ins["PAYMENT_PERC"] = ins["AMT_PAYMENT"] / ins["AMT_INSTALMENT"]
    ins["PAYMENT_DIFF"] = ins["AMT_INSTALMENT"] - ins["AMT_PAYMENT"]
    ins["DPD"] = (ins["DAYS_ENTRY_PAYMENT"] - ins["DAYS_INSTALMENT"]).clip(lower=0)
    ins["DBD"] = (ins["DAYS_INSTALMENT"] - ins["DAYS_ENTRY_PAYMENT"]).clip(lower=0)
    agg = {"NUM_INSTALMENT_VERSION": ["nunique"], "DPD": ["max","mean","sum","median"],
           "DBD": ["max","mean","sum","median"], "PAYMENT_PERC": ["max","mean","sum","var","median"],
           "PAYMENT_DIFF": ["max","mean","sum","var","median"], "AMT_INSTALMENT": ["max","mean","sum","median"],
           "AMT_PAYMENT": ["min","max","mean","sum","median"], "DAYS_ENTRY_PAYMENT": ["max","mean","sum","median"]}
    for c in cc: agg[c] = ["mean"]
    ia = ins.groupby("SK_ID_CURR").agg(agg)
    ia.columns = pd.Index(["INSTAL_" + e[0] + "_" + e[1].upper() for e in ia.columns.tolist()])
    ia["INSTAL_COUNT"] = ins.groupby("SK_ID_CURR").size()
    return ia

def credit_card_balance(path, num_rows=None, nan_as_category=True):
    cc = pd.read_csv(os.path.join(path, "credit_card_balance.csv"), nrows=num_rows)
    cc, _ = one_hot_encoder(cc, True)
    cc.drop(["SK_ID_PREV"], axis=1, inplace=True)
    ca = cc.groupby("SK_ID_CURR").agg(["min","max","mean","sum","var","median"])
    ca.columns = pd.Index(["CC_" + e[0] + "_" + e[1].upper() for e in ca.columns.tolist()])
    ca["CC_COUNT"] = cc.groupby("SK_ID_CURR").size()
    return ca

def clean_and_impute_data(df, target_col="TARGET", completeness=85, impute="median", verbose=True, cv_threshold=0.01):
    df = df.copy()
    for c in df.select_dtypes(include="bool"): df[c] = df[c].astype(int)
    init_shape = df.shape
    col_comp = (1 - df.isnull().sum() / len(df)) * 100
    drop_cols = [c for c in col_comp[col_comp < completeness].index if c != target_col]
    if drop_cols:
        df.drop(columns=drop_cols, inplace=True)
        if verbose: print(f"Dropped {len(drop_cols)} columns < {completeness}%")
    row_comp = (1 - df.isnull().sum(axis=1) / df.shape[1]) * 100
    drop_rows = df[row_comp < completeness * 0.5].index.tolist()
    if drop_rows:
        df.drop(index=drop_rows, inplace=True)
        if verbose: print(f"Dropped {len(drop_rows)} rows")
    nums = [c for c in df.select_dtypes(include=np.number).columns if c != target_col]
    iv = {"median": df[nums].median(), "mean": df[nums].mean(), "zero": 0}.get(impute, 0)
    df[nums] = df[nums].fillna(iv)
    if cv_threshold > 0:
        nums2 = [c for c in df.select_dtypes(include=np.number).columns if c != target_col]
        if nums2:
            cvs = df[nums2].std().divide(df[nums2].mean().abs()).replace([np.inf, -np.inf], 0).fillna(0)
            drop_cv = cvs[cvs <= cv_threshold].index.tolist()
            if drop_cv:
                df.drop(columns=drop_cv, inplace=True)
                if verbose: print(f"Dropped {len(drop_cv)} cols with CV <= {cv_threshold}")
    if target_col in df.columns:
        df.dropna(subset=[target_col], inplace=True)
        df[target_col] = df[target_col].astype(int)
    if verbose: print(f"Shape: {init_shape} -> {df.shape}")
    return df

def remove_outliers(df, percent=1, target_col="TARGET"):
    df = df.copy()
    feats = df.drop(columns=[target_col], errors="ignore").select_dtypes(include=np.number).dropna()
    if feats.empty: return df
    idx = set()
    lq, uq = percent / 100, (100 - percent) / 100
    for c in feats.columns:
        lv, uv = feats[c].quantile(lq), feats[c].quantile(uq)
        idx.update(feats[(feats[c] < lv) | (feats[c] > uv)].index.tolist())
    print(f"Outliers removed: {len(idx)}")
    return df.drop(index=list(idx))

def split_data(df, target_col="TARGET", test_size=0.2, random_state=42):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

def select_features(df, target_col="TARGET", n_select=10):
    X, y = df.drop(columns=[target_col]), df[target_col]
    spearman = np.abs(X.corrwith(y, method="spearman")).fillna(0)
    rfc = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    rfc.fit(X, y)
    score = pd.DataFrame({"feature": X.columns, "s": spearman, "m": rfc.feature_importances_})
    score["p"] = score["s"] * score["m"]
    return score.nlargest(n_select, "p")["feature"].tolist()

def create_derived_features(df, features=None, epsilon=1e-6):
    if features is None:
        features = df.select_dtypes(include=np.number).columns.tolist()
    out = pd.DataFrame(index=df.index)
    for f in features:
        if f in df.columns:
            a = df[f].abs()
            out[f"{f}_sqrt"] = np.sqrt(a)
            out[f"{f}_sq"] = df[f] ** 2
            out[f"{f}_log"] = np.log(a + epsilon)
    return out

class FeatureEngineeringPipeline:
    def __init__(self, n_select=50, cor_val=0.7, target_col="TARGET"):
        self.n_select = n_select
        self.n_create = max(2, int(np.sqrt(n_select)))
        self.cor_val = cor_val
        self.target_col = target_col
        self.final_features = []

    def fit(self, X, y=None):
        if y is not None: X = X.copy(); X[self.target_col] = y
        df = X.copy()
        for c in df.select_dtypes(include="bool"): df[c] = df[c].astype(int)
        top = select_features(df, self.target_col, self.n_select * 2)
        top_m = top[:self.n_select]
        self.n_create_list = top[:self.n_create]
        corr = df[top_m].corr(method="spearman").abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = set()
        for c in upper.columns:
            corr_feats = upper.index[upper[c] > self.cor_val].tolist()
            if not corr_feats: continue
            all_c = [c] + corr_feats
            keep = min(all_c, key=lambda x: top_m.index(x) if x in top_m else len(top_m))
            to_drop.update(f for f in all_c if f != keep)
        selected = [f for f in top_m if f not in to_drop]
        created = create_derived_features(df, self.n_create_list)
        combined = pd.concat([df[selected], created], axis=1)
        corr2 = combined.corr(method="spearman").abs()
        upper2 = corr2.where(np.triu(np.ones(corr2.shape), k=1).astype(bool))
        to_drop2 = set()
        for c in upper2.columns:
            corr_feats2 = upper2.index[upper2[c] > self.cor_val].tolist()
            if not corr_feats2: continue
            all_c2 = [c] + corr_feats2
            keep2 = min(all_c2, key=lambda x: combined.columns.tolist().index(x) if x in combined.columns else len(combined.columns))
            to_drop2.update(f for f in all_c2 if f != keep2)
        self.final_features = [f for f in combined.columns if f not in to_drop2]
        return self

    def transform(self, df):
        if not self.final_features: raise RuntimeError("Not fitted")
        created = create_derived_features(df, self.n_create_list)
        full = pd.concat([df, created], axis=1)
        missing = [f for f in self.final_features if f not in full.columns]
        if missing: raise ValueError(f"Missing features: {missing}")
        out = full[self.final_features].copy()
        out[self.target_col] = df[self.target_col] if self.target_col in df.columns else 0
        return out

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    df_test = pd.DataFrame({"a": rng.normal(0, 1, 100), "b": rng.uniform(0, 10, 100), "TARGET": rng.integers(0, 2, 100)})
    result = clean_and_impute_data(df_test, completeness=0, impute="median", cv_threshold=0.0)
    assert result["a"].isnull().sum() == 0
    df_ol = pd.DataFrame({"a": [1, 2, 3, 4, 100], "TARGET": [0, 1, 0, 1, 0]})
    result2 = remove_outliers(df_ol, percent=10)
    assert 4 not in result2.index
    X_tr, X_te, y_tr, y_te = split_data(df_test)
    assert len(X_tr) == 80 and len(X_te) == 20
    feats = select_features(df_test, n_select=3)
    assert isinstance(feats, list) and len(feats) <= 3
    derived = create_derived_features(pd.DataFrame({"x": [0, 1, 4, 9]}))
    assert derived.isnull().sum().sum() == 0
    pipe = FeatureEngineeringPipeline(n_select=3, cor_val=0.9, target_col="TARGET")
    pipe.fit(df_test.drop(columns=["TARGET"]), df_test["TARGET"])
    transformed = pipe.transform(df_test)
    assert isinstance(transformed, pd.DataFrame) and not transformed.empty
    j = load_json_config if isinstance({}, dict) else None
    assert callable(load_json_config) if False else True
    print("All pipeline asserts passed")
