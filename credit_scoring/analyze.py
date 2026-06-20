import json, logging
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

logger = logging.getLogger(__name__)

def load_test_data(data_dir):
    df = pd.read_parquet(Path(data_dir) / "test_processed.parquet")
    return df.drop(columns=["TARGET"]), df["TARGET"]

def sample_data(X, y, sample_size=500, random_state=42):
    n = min(sample_size, len(X))
    _, idx = next(StratifiedShuffleSplit(n_splits=1, test_size=n, random_state=random_state).split(X, y))
    return X.iloc[idx], y.iloc[idx]

def load_model_and_extract(model_dir):
    import pickle
    return pickle.loads((Path(model_dir) / "model.pkl").read_bytes())

def generate_summary_plots(shap_values, X_processed, class_names, output_dir, dpi=150):
    import shap
    import matplotlib.pyplot as plt
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    for i, name in enumerate(class_names):
        fig, _ = plt.subplots(figsize=(12, 8))
        shap.summary_plot(shap_values[i], X_processed, max_display=20, show=False)
        plt.title(f"SHAP Summary - {name}", fontsize=16)
        fig.savefig(output_dir / f"summary_{name}.png", dpi=dpi); plt.close(fig)

def generate_dependency_plots(shap_values, X_processed, features, class_names, output_dir, dpi=150):
    import shap
    import matplotlib.pyplot as plt
    base = Path(output_dir) / "dependency_plots"
    for ci, cn in enumerate(class_names):
        cd = base / cn.replace(" ", "_"); cd.mkdir(parents=True, exist_ok=True)
        for f in features:
            fig, _ = plt.subplots()
            shap.dependence_plot(f, shap_values[ci], X_processed, show=False)
            plt.title(f"Dependence: {f} ({cn})", fontsize=14)
            safe = "".join(c for c in f if c.isalnum() or c in ("_", "-")).rstrip()
            fig.savefig(cd / f"dependency_{safe}.png", dpi=dpi); plt.close(fig)

def save_global_shap_importance(shap_values, feature_names, class_names, output_path):
    pos = 1 if len(class_names) >= 2 else 0
    records = pd.DataFrame({"feature": feature_names, "importance": np.abs(shap_values[pos]).mean(axis=0)}).sort_values("importance", ascending=False).to_dict(orient="records")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(records, indent=2))

def run_shap_analysis(model_dir, dataset_dir, output_dir, sample_size=500, top_n=8, dpi=150, random_state=42):
    import shap
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    X_test, y_test = load_test_data(dataset_dir)
    X_sample, y_sample = sample_data(X_test, y_test, sample_size, random_state)
    pd.concat([X_sample, y_sample], axis=1).to_csv(out / "data_for_analysis.csv", index=False)
    pipe = load_model_and_extract(model_dir)
    scaler = pipe.named_steps.get("scaler", None) if hasattr(pipe, "named_steps") else None
    clf = pipe.named_steps["clf"] if hasattr(pipe, "named_steps") else pipe
    X_proc = scaler.transform(X_sample) if scaler else X_sample.values
    X_proc = pd.DataFrame(X_proc, index=X_sample.index, columns=X_sample.columns if not scaler else [f"f{i}" for i in range(X_proc.shape[1])])
    explainer = shap.TreeExplainer(clf)
    sv = explainer.shap_values(X_proc)
    if isinstance(sv, np.ndarray) and sv.ndim == 3:
        sv = [sv[:, :, i] for i in range(sv.shape[2])]
    class_names = [f"Class_{c}" for c in (clf.classes_ if hasattr(clf, "classes_") else [0, 1])]
    gi = np.sum([np.abs(s) for s in sv], axis=0).mean(axis=0)
    top_feats = pd.DataFrame({"f": X_proc.columns, "i": gi}).nlargest(top_n, "i")["f"].tolist()
    generate_summary_plots(sv, X_proc, class_names, out / "plots", dpi)
    generate_dependency_plots(sv, X_proc, top_feats, class_names, out / "plots", dpi)
    save_global_shap_importance(sv, X_proc.columns.tolist(), class_names, out / "global_feature_importance.json")
    return {"sample_csv": str(out / "data_for_analysis.csv"), "plots_dir": str(out / "plots"), "importance_json": str(out / "global_feature_importance.json")}

def load_datasets(data_dir):
    d = Path(data_dir)
    return pd.read_parquet(d / "train_processed.parquet"), pd.read_parquet(d / "test_processed.parquet")

def generate_drift_report(reference, current, output_path):
    from evidently.legacy.metric_preset import DataDriftPreset
    from evidently.legacy.report import Report
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Report(metrics=[DataDriftPreset()]).run(reference_data=reference, current_data=current).save_html(str(output_path))
    return output_path

def run_drift_analysis(dataset_dir, output_dir):
    ref, cur = load_datasets(dataset_dir)
    return generate_drift_report(ref, cur, Path(output_dir) / "data_drift_report.html")

if __name__ == "__main__":
    import tempfile, shutil
    rng = np.random.default_rng(42)
    data = {f"f{i}": rng.normal(0, 1, 100) for i in range(5)}
    data["TARGET"] = rng.integers(0, 2, 100)
    df = pd.DataFrame(data)
    X, y = df.drop(columns=["TARGET"]), df["TARGET"]
    Xs, ys = sample_data(X, y, sample_size=10)
    assert len(Xs) == 10
    d = Path(tempfile.mkdtemp())
    tr, te = df.iloc[:80], df.iloc[80:]
    tr.to_parquet(d / "train_processed.parquet"); te.to_parquet(d / "test_processed.parquet")
    ref, cur = load_datasets(d); assert ref.shape[0] == 80 and cur.shape[0] == 20
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", RandomForestClassifier(n_estimators=10, random_state=42))])
    pipe.fit(X, y)
    (d / "model.pkl").write_bytes(__import__("pickle").dumps(pipe))
    try:
        res = run_shap_analysis(d, d, d / "analysis_out", sample_size=10, top_n=2)
        assert "plots_dir" in res
    except (ModuleNotFoundError, ImportError): print("SHAP/evidently not available, skipping")
    try:
        dr = run_drift_analysis(d, d); assert dr.exists()
    except (ModuleNotFoundError, ImportError): print("Evidently not available, skipping")
    shutil.rmtree(d)
    print("All analyze asserts passed")
