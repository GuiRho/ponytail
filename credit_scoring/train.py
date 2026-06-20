import json, os, pickle, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, confusion_matrix, make_scorer

def load_model(model_dir):
    model_path = Path(model_dir) / "model.pkl"
    if not model_path.exists(): raise FileNotFoundError(str(model_path))
    return pickle.loads(model_path.read_bytes())

def save_model(model, model_dir):
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    (Path(model_dir) / "model.pkl").write_bytes(pickle.dumps(model))

def log_metrics(run_name, metrics, params=None, log_dir="logs"):
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    entry = {"run": run_name, "params": params or {}, "metrics": metrics}
    (log_dir / f"{run_name}.json").write_text(json.dumps(entry, indent=2, default=str))

def get_resampling_strategy(technique, target_balance, initial_balance, random_state=42):
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.pipeline import Pipeline as ImbPipeline
    if technique == "oversample":
        if target_balance <= initial_balance: return None
        return SMOTE(sampling_strategy=target_balance, random_state=random_state)
    if technique == "undersample":
        if target_balance == 1.0: return None
        return RandomUnderSampler(sampling_strategy=target_balance / (1 - target_balance), random_state=random_state)
    if technique == "hybrid":
        if target_balance <= initial_balance: return None
        mid = (initial_balance + target_balance) / 2
        return ImbPipeline(steps=[("over", SMOTE(sampling_strategy=mid, random_state=random_state)),
                                   ("under", RandomUnderSampler(sampling_strategy=target_balance / (1 - target_balance), random_state=random_state))])
    raise ValueError(f"Unknown technique: {technique}")

def sanitize_dataframe(df):
    df = df.select_dtypes(include=np.number).copy()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    if df.isna().sum().sum() > 0: df.fillna(df.median(), inplace=True)
    return df

def _compute_custom_and_normalized(y_true, y_pred_bin, pos_proportion):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_bin).ravel()
    custom = (2 * tp) + tn - fp - (10 * fn)
    n = len(y_true)
    pos_prop = max(pos_proportion, 1e-9)
    return float(custom), float(custom * (1.0 / max(1, n)) * (1.0 / pos_prop))

def _find_best_threshold_custom_score(y_true, y_pred_proba, pos_proportion):
    best_t, best_score = 0.5, -np.inf
    for t in np.linspace(0.01, 0.99, 99):
        _, ns = _compute_custom_and_normalized(y_true, (y_pred_proba >= t).astype(int), pos_proportion)
        if ns > best_score: best_score, best_t = ns, t
    return best_t, best_score

def get_classifiers(random_state=42):
    clf = {"logreg": LogisticRegression(max_iter=1000, solver="liblinear", random_state=random_state),
           "random_forest": RandomForestClassifier(n_estimators=100, random_state=random_state, max_depth=8, n_jobs=-1),
           "gradient_boosting": GradientBoostingClassifier(n_estimators=100, random_state=random_state, max_depth=8)}
    try:
        from xgboost import XGBClassifier
        clf["xgboost"] = XGBClassifier(eval_metric="logloss", random_state=random_state, n_jobs=-1)
    except ImportError: pass
    try:
        from catboost import CatBoostClassifier
        clf["catboost"] = CatBoostClassifier(n_estimators=100, max_depth=8, random_state=random_state, verbose=0)
    except ImportError: pass
    return clf

def evaluate_algorithms(input_dir, target_col="TARGET", random_state=42, log_dir="logs"):
    train = pd.read_parquet(os.path.join(input_dir, "train_processed.parquet"))
    test = pd.read_parquet(os.path.join(input_dir, "test_processed.parquet"))
    X_tr, y_tr = train.drop(columns=[target_col]), train[target_col]
    X_te, y_te = test.drop(columns=[target_col]), test[target_col]
    for c in X_tr.select_dtypes(include=["int32","int64"]):
        X_tr[c] = X_tr[c].astype("float64"); X_te[c] = X_te[c].astype("float64")
    pos_prop = float(y_tr.mean())
    balances = ["init", 0.11, 0.15]
    for bal in balances:
        if bal != "init":
            from imblearn.under_sampling import RandomUnderSampler
            rus = RandomUnderSampler(sampling_strategy=bal / (1 - bal), random_state=random_state)
            X_b, y_b = rus.fit_resample(X_tr, y_tr)
        else: X_b, y_b = X_tr, y_tr
        for name, clf in get_classifiers(random_state).items():
            pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
            pipe.fit(X_b, y_b)
            p_te = pipe.predict_proba(X_te)[:, 1]
            best_t, _ = _find_best_threshold_custom_score(y_b.to_numpy(), pipe.predict_proba(X_b)[:, 1], pos_prop)
            yb = (p_te >= best_t).astype(int)
            _, tn = _compute_custom_and_normalized(y_te.to_numpy(), yb, pos_prop)
            run_name = f"{Path(input_dir).name}__{name}__bal_{bal}"
            log_metrics(run_name, {"test_auc": float(roc_auc_score(y_te, p_te)), "test_normalized_score": tn,
                        "best_threshold": float(best_t), "test_rows": len(X_te), "num_features": X_te.shape[1]},
                        params={"algorithm": name, "balance": str(bal)}, log_dir=log_dir)
            print(f"{run_name}: AUC={roc_auc_score(y_te, p_te):.4f}")

def evaluate_balancing_strategies(input_dir, target_col="TARGET", log_dir="logs", random_state=42):
    train = pd.read_parquet(os.path.join(input_dir, "train_processed.parquet"))
    test = pd.read_parquet(os.path.join(input_dir, "test_processed.parquet"))
    X_tr = sanitize_dataframe(train.drop(columns=[target_col]))
    y_tr = train[target_col]
    X_te = sanitize_dataframe(test.drop(columns=[target_col]))
    y_te = test[target_col]
    init_bal = float(y_tr.value_counts(normalize=True).get(1, 0))
    for target in np.arange(0.10, 0.51, 0.05):
        for tech in ["undersample", "oversample", "hybrid"]:
            rs = get_resampling_strategy(tech, target, init_bal, random_state)
            if rs is None: continue
            X_r, y_r = rs.fit_resample(X_tr, y_tr)
            pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000, random_state=random_state))])
            pipe.fit(X_r, y_r)
            p = pipe.predict_proba(X_te)[:, 1]
            auc = float(roc_auc_score(y_te, p))
            run_name = f"bal_{Path(input_dir).name}_{target:.2f}_{tech}"
            log_metrics(run_name, {"test_auc": auc, "resampled_rows": X_r.shape[0],
                        "final_balance": float(y_r.value_counts(normalize=True).get(1, 0))},
                        params={"technique": tech, "target_balance": float(target)}, log_dir=log_dir)
            print(f"{run_name}: AUC={auc:.4f}")

def _run_grid_search_logreg(X_train, y_train, pos_prop_global, random_state=42):
    def scorer(y_true, y_pred):
        t, _ = _find_best_threshold_custom_score(y_true, y_pred, pos_prop_global)
        _, ns = _compute_custom_and_normalized(y_true, (y_pred >= t).astype(int), pos_prop_global)
        return ns
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=3000, random_state=random_state))])
    grid = [
        {"clf__solver": ["liblinear"], "clf__penalty": ["l1","l2"], "clf__C": [0.01,0.1,1,10,100], "clf__class_weight": [None,"balanced"]},
        {"clf__solver": ["saga"], "clf__penalty": ["l1","l2","elasticnet","none"], "clf__C": [0.01,0.1,1,10,100], "clf__class_weight": [None,"balanced"], "clf__l1_ratio": np.linspace(0,1,5)},
        {"clf__solver": ["lbfgs"], "clf__penalty": ["l2","none"], "clf__C": [0.01,0.1,1,10,100], "clf__class_weight": [None,"balanced"]}]
    gs = GridSearchCV(pipe, grid, scoring=make_scorer(scorer, needs_proba=True, greater_is_better=True), cv=StratifiedKFold(3, shuffle=True, random_state=random_state), n_jobs=-1, verbose=0)
    gs.fit(X_train, y_train)
    return gs.best_estimator_, {k.replace("clf__",""): v for k, v in gs.best_params_.items()}, gs.best_score_

def get_optuna_suggest_and_model(model_name, random_state=42):
    def suggest_rf(t):
        return {"n_estimators": t.suggest_int("n_estimators",50,400), "max_depth": t.suggest_int("max_depth",3,15),
                "min_samples_split": t.suggest_int("min_samples_split",2,10), "min_samples_leaf": t.suggest_int("min_samples_leaf",1,4)}
    def suggest_xgb(t):
        return {"n_estimators": t.suggest_int("n_estimators",50,400), "learning_rate": t.suggest_float("learning_rate",1e-3,0.3,log=True),
                "max_depth": t.suggest_int("max_depth",3,10), "subsample": t.suggest_float("subsample",0.6,1.0)}
    def suggest_lgb(t):
        return {"n_estimators": t.suggest_int("n_estimators",50,400), "num_leaves": t.suggest_int("num_leaves",10,200),
                "learning_rate": t.suggest_float("learning_rate",1e-3,0.3,log=True), "subsample": t.suggest_float("subsample",0.6,1.0)}
    if model_name == "random_forest": return suggest_rf, RandomForestClassifier(random_state=random_state, n_jobs=-1)
    try:
        if model_name == "xgboost":
            from xgboost import XGBClassifier
            return suggest_xgb, XGBClassifier(random_state=random_state, eval_metric="logloss", n_jobs=-1)
    except ImportError: pass
    try:
        if model_name == "lightgbm":
            import lightgbm as lgb
            return suggest_lgb, lgb.LGBMClassifier(random_state=random_state, n_jobs=-1)
    except ImportError: pass
    raise ValueError(f"Unsupported model: {model_name}")

def tune_and_log_model(config, random_state=42, register_as=None, log_dir="logs"):
    input_dir, model_name, n_trials, target_col = config["dataset_dir"], config["model_name"], config.get("n_trials"), "TARGET"
    train = pd.read_parquet(os.path.join(input_dir, "train_processed.parquet"))
    test = pd.read_parquet(os.path.join(input_dir, "test_processed.parquet"))
    X_tr = train.drop(columns=[target_col]).select_dtypes(include=np.number)
    y_tr = train[target_col]
    X_te = test.drop(columns=[target_col]).select_dtypes(include=np.number)
    y_te = test[target_col]
    for c in X_tr.select_dtypes(include=["int32","int64"]):
        X_tr[c] = X_tr[c].astype("float64"); X_te[c] = X_te[c].astype("float64")
    pos_prop = float(y_tr.mean())
    if model_name == "logreg":
        pipe, params, cv_score = _run_grid_search_logreg(X_tr, y_tr, pos_prop, random_state)
    else:
        import optuna
        from optuna.samplers import TPESampler
        suggest_fn, base_model = get_optuna_suggest_and_model(model_name, random_state)
        def objective(t):
            p = suggest_fn(t)
            try:
                m = base_model.set_params(**p)
                scores = []
                for ti, vi in StratifiedKFold(3, shuffle=True, random_state=random_state).split(X_tr, y_tr):
                    pipe = Pipeline([("scaler", StandardScaler()), ("clf", m)])
                    pipe.fit(X_tr.iloc[ti], y_tr.iloc[ti])
                    pv = pipe.predict_proba(X_tr.iloc[vi])[:, 1]
                    _, ns = _compute_custom_and_normalized(y_tr.iloc[vi].to_numpy(), (pv >= _find_best_threshold_custom_score(y_tr.iloc[vi].to_numpy(), pv, pos_prop)[0]).astype(int), pos_prop)
                    scores.append(ns)
                return float(np.mean(scores))
            except Exception: raise optuna.exceptions.TrialPruned()
        study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=random_state))
        study.optimize(objective, n_trials=n_trials, n_jobs=1)
        params = study.best_params
        cv_score = study.best_value
        final = base_model.set_params(**params)
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", final)])
        pipe.fit(X_tr, y_tr)
    p_te = pipe.predict_proba(X_te)[:, 1]
    p_tr = pipe.predict_proba(X_tr)[:, 1]
    best_t, _ = _find_best_threshold_custom_score(y_tr.to_numpy(), p_tr, pos_prop)
    yb = (p_te >= best_t).astype(int)
    tc, tn = _compute_custom_and_normalized(y_te.to_numpy(), yb, pos_prop)
    run_name = f"tune_{model_name}_{Path(input_dir).name}"
    log_metrics(run_name, {"best_cv_score": float(cv_score), "test_normalized": tn, "test_custom": tc,
                "best_threshold": float(best_t)},
                params={"model": model_name, "best_params": params, "dataset": Path(input_dir).name}, log_dir=log_dir)
    if register_as: save_model(pipe, f"registered_models/{register_as}")
    return pipe

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    df = pd.DataFrame({"a": rng.normal(0, 1, 50), "b": rng.uniform(0, 10, 50), "TARGET": rng.integers(0, 2, 50)})
    from sklearn.model_selection import train_test_split
    import tempfile
    tmp = Path(tempfile.mkdtemp())
    model_dir = tmp / "model"; model_dir.mkdir()
    model = LogisticRegression(max_iter=100); model.fit(df[["a","b"]], df["TARGET"])
    save_model(model, model_dir); loaded = load_model(model_dir); assert hasattr(loaded, "predict")
    log_metrics("test_run", {"score": 1.0}, params={"x": 1})
    assert (Path("logs") / "test_run.json").exists()
    try:
        strat = get_resampling_strategy("oversample", 0.3, 0.1)
        assert strat is not None
        assert get_resampling_strategy("oversample", 0.05, 0.1) is None
    except ModuleNotFoundError: pass
    df_s = sanitize_dataframe(df); assert df_s.isna().sum().sum() == 0
    c, n = _compute_custom_and_normalized(np.array([0,1,0,1]), np.array([0,1,0,1]), 0.5)
    assert isinstance(c, float) and isinstance(n, float)
    t, s = _find_best_threshold_custom_score(np.array([0,1,0,1]), np.array([0.1,0.9,0.2,0.8]), 0.5)
    assert 0 < t < 1
    clfs = get_classifiers(); assert "logreg" in clfs
    import shutil; shutil.rmtree(tmp, ignore_errors=True); shutil.rmtree("logs", ignore_errors=True)
    print("All train asserts passed")
