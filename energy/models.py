import logging, time, warnings, traceback, numpy as np, pandas as pd
from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

try:
    import xgboost as xgb; XGB = True
except ImportError:
    xgb = None; XGB = False
try:
    import lightgbm as lgb; LGB = True
except ImportError:
    lgb = None; LGB = False
try:
    import catboost as cb; CB = True
except ImportError:
    cb = None; CB = False
try:
    import shap; SHAP = True
except ImportError:
    shap = None; SHAP = False

logger = logging.getLogger(__name__)
RS = 42


def build_model_dict():
    models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0, random_state=RS),
        "Lasso": Lasso(alpha=0.1, max_iter=2000, random_state=RS),
        "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=2000, random_state=RS),
        "KNN": KNeighborsRegressor(n_neighbors=3),
        "SVR_linear": SVR(kernel="linear", C=1.0, cache_size=500),
        "SVR_rbf": SVR(kernel="rbf", C=1.0, gamma="scale", cache_size=500),
        "DecisionTree": DecisionTreeRegressor(max_depth=10, min_samples_leaf=5, random_state=RS),
        "RandomForest": RandomForestRegressor(n_estimators=150, max_depth=10, random_state=RS, n_jobs=-1),
        "AdaBoost": AdaBoostRegressor(n_estimators=100, learning_rate=1.0, random_state=RS),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=RS),
    }
    if XGB:
        models["XGBoost"] = xgb.XGBRegressor(n_estimators=100, learning_rate=0.15, max_depth=3, random_state=RS, n_jobs=-1, objective="reg:squarederror")
    if LGB:
        models["LightGBM"] = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.15, random_state=RS, n_jobs=-1, verbosity=-1)
    if CB:
        models["CatBoost"] = cb.CatBoostRegressor(iterations=100, learning_rate=0.15, depth=6, random_state=RS, verbose=0, thread_count=-1)
    return models


def evaluate_models(X_train, X_test, y_train, y_test, models):
    rows = []
    for name, m in models.items():
        try:
            t0 = time.time(); m.fit(X_train, y_train); ft = time.time() - t0
            t0 = time.time(); yp = m.predict(X_test); pt = time.time() - t0
            rows.append({"Model": name, "R2 (Test)": r2_score(y_test, yp), "MAE (Test)": mean_absolute_error(y_test, yp),
                          "RMSE (Test)": np.sqrt(mean_squared_error(y_test, yp)), "Fit Time (s)": ft, "Predict Time (s)": pt, "Notes": ""})
        except Exception as e:
            rows.append({"Model": name, "R2 (Test)": np.nan, "MAE (Test)": np.nan, "RMSE (Test)": np.nan,
                          "Fit Time (s)": np.nan, "Predict Time (s)": np.nan, "Notes": f"Error: {e}"})
    return pd.DataFrame(rows).sort_values("R2 (Test)", ascending=False, na_position="last").reset_index(drop=True)


def prepare_data(df, target, test_size=0.2):
    df_c = df.copy()
    X_tr, X_te, y_tr, y_te = train_test_split(df_c.drop(columns=[target]), df_c[target], test_size=test_size, random_state=RS)
    scaler = StandardScaler()
    return scaler.fit_transform(X_tr), scaler.transform(X_te), y_tr, y_te


def overview_models(df, target, test_size=0.2):
    for c in [FutureWarning, UserWarning, DeprecationWarning]:
        warnings.filterwarnings("ignore", category=c)
    X_tr, X_te, y_tr, y_te = prepare_data(df, target, test_size)
    return evaluate_models(X_tr, X_te, y_tr, y_te, build_model_dict())


def split_and_select_numeric_data(df, target_col, test_size=0.2, random_state=42):
    if target_col not in df.columns:
        raise ValueError(f"Target '{target_col}' not found.")
    if not pd.api.types.is_numeric_dtype(df[target_col]):
        raise ValueError("Target must be numeric.")
    df_c = df.dropna(subset=[target_col])
    if len(df_c) < 2:
        raise ValueError("Not enough data.")
    X, y = df_c.drop(target_col, axis=1), df_c[target_col]
    X_num = X.select_dtypes(include=np.number)
    fnames = list(X_num.columns)
    if not fnames:
        raise ValueError("No numeric features.")
    X_tr, X_te, y_tr, y_te = train_test_split(X_num, y, test_size=test_size, random_state=random_state)
    if X_tr.isnull().sum().sum() > 0 or X_te.isnull().sum().sum() > 0:
        imp = X_tr.median()
        X_tr, X_te = X_tr.fillna(imp), X_te.fillna(imp)
    return X_tr, X_te, y_tr, y_te, fnames, X_te.index, X_tr.index


def apply_preprocessing(X_train, X_test, scaler_type, pca_n_comp, random_state=42):
    X_tr_s, X_te_s = X_train.copy(), X_test.copy()
    orig_names = list(X_train.columns)
    scaler, pca = None, None
    if scaler_type and scaler_type.lower() != "none":
        scalers = {"standard": StandardScaler(), "robust": RobustScaler(), "minmax": MinMaxScaler()}
        s = scalers.get(scaler_type.lower())
        if s is None:
            raise ValueError(f"Unknown scaler: {scaler_type}")
        scaler = s
        X_tr_np, X_te_np = scaler.fit_transform(X_tr_s), scaler.transform(X_te_s)
    else:
        X_tr_np, X_te_np = X_tr_s.values, X_te_s.values
    if pca_n_comp is not None:
        if not (isinstance(pca_n_comp, int) and 0 < pca_n_comp <= X_tr_np.shape[1]) and \
           not (isinstance(pca_n_comp, float) and 0 < pca_n_comp <= 1.0):
            raise ValueError(f"Invalid pca_n_comp: {pca_n_comp}")
        pca = PCA(n_components=pca_n_comp, random_state=random_state)
        X_tr_np, X_te_np = pca.fit_transform(X_tr_np), pca.transform(X_te_np)
        orig_names = [f"PC{i}" for i in range(X_tr_np.shape[1])]
    return X_tr_np, X_te_np, orig_names, scaler, pca


def perform_rfe(X_train, y_train, feature_names, target_feature_ratio=0.6, random_state=42, rfe_estimator_params=None):
    n = X_train.shape[1]
    if not (target_feature_ratio and 0 < target_feature_ratio < 1 and n > 0):
        return None, feature_names, X_train, np.ones(n, dtype=bool)
    n_sel = max(1, int(n * target_feature_ratio))
    params = dict(rfe_estimator_params) if rfe_estimator_params else {"n_estimators": 150, "max_depth": 10, "n_jobs": -1}
    params.pop("random_state", None)
    rfe = RFE(estimator=RandomForestRegressor(random_state=random_state, **params), n_features_to_select=min(n_sel, n), step=1)
    try:
        rfe.fit(X_train, y_train)
    except Exception as e:
        logger.error("RFE failed: %s", e)
        return None, feature_names, X_train, np.ones(n, dtype=bool)
    mask = rfe.support_
    if mask.sum() == 0:
        return None, feature_names, X_train, np.ones(n, dtype=bool)
    sel = [n for n, s in zip(feature_names, mask) if s] if len(feature_names) == n else [f"S_{i}" for i in range(mask.sum())]
    return rfe, sel, rfe.transform(X_train), mask


def train_random_forest(X_train, y_train, random_state=RS, rf_params=None):
    if X_train.shape[1] == 0 or len(X_train) == 0:
        raise ValueError("Cannot train with 0 features or samples.")
    params = {"n_estimators": 150, "max_depth": 10, "n_jobs": -1}
    if rf_params:
        params.update(rf_params)
    params["random_state"] = random_state
    m = RandomForestRegressor(**params)
    m.fit(X_train, y_train)
    return m


def evaluate_regression(model, X_train, y_train, X_test, y_test):
    if X_test.shape[0] == 0 or X_train.shape[1] == 0:
        return {"Train": {}, "Test": {}}
    yp_tr, yp_te = model.predict(X_train), model.predict(X_test)
    def m(d, p): return {"R²": r2_score(d, p), "MAE": mean_absolute_error(d, p), "RMSE": float(np.sqrt(mean_squared_error(d, p)))}
    return {"Train": m(y_train, yp_tr), "Test": m(y_test, yp_te)}


def analyze_residuals(model, X_test, y_test, test_original_indices, n_worst=20):
    if X_test.shape[0] == 0 or y_test.empty:
        return pd.Series(dtype=float), pd.DataFrame(), np.array([])
    yp = model.predict(X_test)
    resid = pd.Series(y_test.values - yp, index=test_original_indices, name="Residual")
    worst = pd.DataFrame()
    if n_worst > 0 and not resid.empty:
        ab = resid.abs().sort_values(ascending=False)
        idx = ab.index[:min(n_worst, len(resid))]
        worst = pd.DataFrame({"Original_Index": idx, "Actual_Target": y_test.loc[idx].values,
                               "Predicted_Target": pd.Series(yp, index=test_original_indices).loc[idx].values,
                               "Residual": resid.loc[idx].values, "Absolute_Residual": ab.loc[idx].values})
    return resid, worst, yp


def analyze_feature_importance_mdi(model, feature_names):
    if not feature_names or not hasattr(model, "feature_importances_"):
        return pd.DataFrame(), None
    imp = model.feature_importances_
    feat = list(feature_names)
    if len(imp) > len(feat):
        feat.extend([f"U_{i}" for i in range(len(feat), len(imp))])
    else:
        feat = feat[:len(imp)]
    imp_df = pd.DataFrame({"Feature": feat, "Importance_MDI": imp}).sort_values("Importance_MDI", ascending=False).reset_index(drop=True)
    vals = np.maximum(imp_df["Importance_MDI"].values, 0)
    total = np.sum(vals)
    gini = None
    if not np.isclose(total, 0):
        cum = np.cumsum(vals) / total
        gini = float(np.clip(1 - 2 * np.trapezoid(np.insert(cum, 0, 0), np.linspace(0, 1, len(vals) + 1)), 0, 1))
        imp_df["Cumulative_MDI_Importance_Normalized"] = cum.tolist()
    return imp_df, gini


def analyze_shap(model, X_test, feature_names, test_original_indices):
    if not SHAP or X_test.shape[1] == 0 or X_test.shape[0] == 0:
        return None, None, None
    X_df = pd.DataFrame(X_test, columns=feature_names, index=test_original_indices)
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X_df)
    ev = float(explainer.expected_value[0]) if isinstance(explainer.expected_value, (list, np.ndarray)) else float(explainer.expected_value)
    sv_df = pd.DataFrame(sv, columns=feature_names, index=test_original_indices)
    ss_df = pd.DataFrame({"Feature": feature_names, "Mean_Abs_SHAP": np.abs(sv).mean(axis=0)}).sort_values("Mean_Abs_SHAP", ascending=False).reset_index(drop=True)
    return sv_df, ss_df, ev


def run_regression_pipeline(df, target_col, scaler_type="Standard", pca_n_comp=None, test_size=0.2,
                            random_state=42, target_feature_ratio=0.6, rf_params=None,
                            n_worst_residuals=20, output_file=None, run_shap=True):
    logger.info("Pipeline (target='%s')", target_col)
    start = time.time()
    results = {"run_info": {"target": target_col, "test_size": test_size, "scaler": scaler_type, "pca": pca_n_comp, "rfe_ratio": target_feature_ratio}}
    try:
        X_tr, X_te, y_tr, y_te, fnames, tidx, _ = split_and_select_numeric_data(df, target_col, test_size, random_state)
        results.update({"y_train": y_tr, "y_test": y_te, "test_original_indices": tidx})
        Xp_tr, Xp_te, fp, _, _ = apply_preprocessing(X_tr, X_te, scaler_type, pca_n_comp, random_state)
        results["feature_names_after_processing"] = fp
        rfe_obj, sn, Xf_tr, _ = perform_rfe(Xp_tr, y_tr, fp, target_feature_ratio, random_state, rf_params)
        if rfe_obj is not None and Xf_tr.shape[1] < Xp_tr.shape[1]:
            Xf_te, ff = rfe_obj.transform(Xp_te), sn
        else:
            Xf_te, ff = Xp_te, fp
        model = train_random_forest(Xf_tr, y_tr, random_state, rf_params)
        results.update({"model": model, "selected_features": ff})
        results["metrics"] = evaluate_regression(model, Xf_tr, y_tr, Xf_te, y_te)
        resid, wdf, yp = analyze_residuals(model, Xf_te, y_te, tidx, n_worst_residuals)
        results.update({"residuals": resid, "worst_residuals_df": wdf, "y_test_pred": yp})
        imp_df, gini = analyze_feature_importance_mdi(model, ff)
        results["mdi_importance_df"] = imp_df
        results["run_info"]["mdi_gini"] = gini
        if run_shap and SHAP:
            sv_df, ss_df, ev = analyze_shap(model, Xf_te, ff, tidx)
            results.update({"shap_values_df": sv_df, "shap_summary_df": ss_df, "shap_expected_value": ev})
        if output_file:
            _save_results(results, output_file)
    except Exception as e:
        logger.error("Pipeline failed: %s", e)
        traceback.print_exc()
        return None
    results["run_info"]["duration"] = round(time.time() - start, 2)
    return results


def _save_results(results, output_file):
    if not output_file.endswith(".csv"):
        output_file = output_file.rsplit(".", 1)[0] + ".csv" if "." in output_file else output_file + ".csv"
    ri = results.get("run_info")
    if ri:
        pd.DataFrame(list(ri.items()), columns=["Parameter", "Value"]).to_csv(output_file.replace(".csv", "_info.csv"), index=False)
    m = results.get("metrics")
    if m:
        pd.DataFrame(m).to_csv(output_file.replace(".csv", "_metrics.csv"))
    for key, name in [("mdi_importance_df", "_importance"), ("shap_summary_df", "_shap_summary"), ("worst_residuals_df", "_worst_residuals")]:
        v = results.get(key)
        if isinstance(v, pd.DataFrame) and not v.empty:
            v.to_csv(output_file.replace(".csv", f"{name}.csv"), index=False)
    sv = results.get("shap_values_df")
    if isinstance(sv, pd.DataFrame) and not sv.empty:
        sv.to_csv(output_file.replace(".csv", "_shap_values.csv"))
    logger.info("Saved to %s", output_file)


def run_scaling_pca_experiments(X_train, X_test, y_train, y_test, scalers, pca_options, model_class, model_params, random_state=42):
    results = []
    for sname, scaler in scalers.items():
        X_tr_s, X_te_s = X_train.copy(), X_test.copy()
        if scaler is not None:
            try:
                X_tr_s, X_te_s = scaler.fit_transform(X_tr_s), scaler.transform(X_te_s)
            except Exception as e:
                for nc in pca_options:
                    results.append({"Scaler": sname, "PCA_Components": "None" if nc is None else nc, "R2_Score": np.nan, "RMSE": np.nan, "Error": str(e)})
                continue
        nf = X_tr_s.shape[1]
        for nc in pca_options:
            if nc is not None and nc > nf:
                continue
            X_tr_p, X_te_p = X_tr_s, X_te_s
            if nc is not None:
                pca = PCA(n_components=nc, random_state=random_state)
                try:
                    X_tr_p, X_te_p = pca.fit_transform(X_tr_p), pca.transform(X_te_p)
                except Exception as e:
                    results.append({"Scaler": sname, "PCA_Components": nc, "R2_Score": np.nan, "RMSE": np.nan, "Error": str(e)})
                    continue
            try:
                m = model_class(**model_params)
                m.fit(X_tr_p, y_train)
                yp = m.predict(X_te_p)
                results.append({"Scaler": sname, "PCA_Components": "None" if nc is None else nc,
                                "R2_Score": r2_score(y_test, yp), "RMSE": np.sqrt(mean_squared_error(y_test, yp)), "Error": None})
            except Exception as e:
                results.append({"Scaler": sname, "PCA_Components": "None" if nc is None else nc, "R2_Score": np.nan, "RMSE": np.nan, "Error": str(e)})
    return results


def process_and_display_results(results, output_file=None, n_top=5):
    if not results:
        logger.warning("No results.")
        return
    df = pd.DataFrame(results)
    clean = df.dropna(subset=["R2_Score", "RMSE"]).sort_values("R2_Score", ascending=False).reset_index(drop=True)
    logger.info("Top %d:\n%s", n_top, clean.head(n_top).to_string())
    if output_file:
        f = output_file if output_file.endswith(".csv") else output_file + ".csv"
        df.to_csv(f, index=False)
        logger.info("Saved to %s", f)
    return df


def run_pca_scale_pipeline(df, target, acp_min=1, test_size=0.2, random_state=42, output_file=None,
                           model_class=RandomForestRegressor, model_params=None, scalers=None, n_top=5):
    if target not in df.columns:
        raise ValueError(f"Target '{target}' not found.")
    if scalers is None:
        scalers = {"None": None, "Standard": StandardScaler(), "Robust": RobustScaler(), "MinMax": MinMaxScaler()}
    if model_params is None:
        model_params = {"n_estimators": 150, "max_depth": 10, "random_state": random_state, "n_jobs": -1}
    df_c = df.dropna(subset=[target])
    X, y = df_c.drop(target, axis=1), df_c[target]
    X_num = X.select_dtypes(include=np.number)
    if X_num.shape[1] == 0:
        raise ValueError("No numeric features.")
    X_tr, X_te, y_tr, y_te = train_test_split(X_num, y, test_size=test_size, random_state=random_state)
    max_pc = X_tr.shape[1]
    acp = max(1, min(acp_min, max_pc))
    pca_opts = [None] + list(range(acp, max_pc + 1))
    results = run_scaling_pca_experiments(X_tr, X_te, y_tr, y_te, scalers, pca_opts, model_class, model_params, random_state)
    return process_and_display_results(results, output_file, n_top)


if __name__ == "__main__":
    import pytest
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"f1": rng.normal(5, 1, 50), "f2": rng.normal(10, 2, 50), "f3": rng.uniform(0, 10, 50),
                        "target": rng.normal(50, 10, 50)})
    models = build_model_dict()
    assert "LinearRegression" in models and "RandomForest" in models
    assert hasattr(models["LinearRegression"], "fit")
    X_tr, X_te, y_tr, y_te = prepare_data(df, "target")
    assert X_tr.shape[0] > 0
    results = overview_models(df, "target")
    assert "Model" in results.columns and "R2 (Test)" in results.columns
    r = run_regression_pipeline(df, "target", run_shap=False)
    assert r is not None
    assert "metrics" in r
    X_tr2, X_te2, y_tr2, y_te2, fn, _, _ = split_and_select_numeric_data(df, "target")
    assert len(fn) > 0
    Xp_tr, Xp_te, fp, s, p = apply_preprocessing(pd.DataFrame(X_tr2), pd.DataFrame(X_te2), "standard", 2)
    assert Xp_tr.shape[1] == 2
    m = train_random_forest(X_tr2.values, y_tr2)
    assert hasattr(m, "predict")
    ev = evaluate_regression(m, X_tr2.values, y_tr2, X_te2.values, y_te2)
    assert "Train" in ev and "Test" in ev
    imp, g = analyze_feature_importance_mdi(m, fn)
    assert len(imp) > 0
    res_df = run_pca_scale_pipeline(df, "target", acp_min=1, model_class=RandomForestRegressor,
                                     model_params={"n_estimators": 50, "max_depth": 5, "random_state": 42})
    assert res_df is not None and len(res_df) > 0
