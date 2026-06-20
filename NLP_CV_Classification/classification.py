import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from xgboost import XGBClassifier

RS = 42

def models_and_params():
    return {
        "LogisticRegression": {
            "model": LogisticRegression(random_state=RS, max_iter=2000, solver="saga"),
            "params": {"C": np.logspace(-2, 2, 5).tolist(), "penalty": ["l1", "l2"]},
        },
        "RandomForestClassifier": {
            "model": RandomForestClassifier(random_state=RS),
            "params": {"n_estimators": np.linspace(50, 175, 5, dtype=int).tolist(), "max_depth": np.linspace(1, 12, 12, dtype=int).tolist() + [None], "min_samples_split": np.linspace(2, 10, 5, dtype=int).tolist(), "min_samples_leaf": np.linspace(1, 5, 5, dtype=int).tolist(), "max_features": ["sqrt", "log2", 0.7, 0.9]},
        },
        "XGBClassifier": {
            "model": XGBClassifier(random_state=RS, use_label_encoder=False, eval_metric="mlogloss"),
            "params": {"n_estimators": np.linspace(50, 175, 5, dtype=int).tolist(), "max_depth": [3, 5, 7, 9], "learning_rate": np.logspace(-2, -0.5, 4).tolist(), "subsample": np.linspace(0.7, 1.0, 4).tolist(), "colsample_bytree": np.linspace(0.7, 1.0, 4).tolist()},
        },
        "SVC": {
            "model": SVC(random_state=RS, probability=True),
            "params": {"C": np.logspace(-2, 2, 5).tolist(), "gamma": np.logspace(-3, 0, 4).tolist(), "kernel": ["rbf"]},
        },
        "MultinomialNB": {
            "model": MultinomialNB(),
            "params": {"alpha": np.logspace(-3, 1, 5).tolist()},
        },
    }

def tune(X_train, y_train, X_val, y_val, feature_name):
    print(f"\n--- Tuning: {feature_name} ---")
    results = {}
    for name, config in models_and_params().items():
        if name == "MultinomialNB" and np.any(X_train < 0): continue
        rs = RandomizedSearchCV(config["model"], config["params"], n_iter=30, cv=3, scoring="accuracy", random_state=RS, n_jobs=-1, verbose=0).fit(X_train, y_train)
        val_score = rs.best_estimator_.score(X_val, y_val)
        results[name] = {"search_results": rs, "best_cv_score": rs.best_score_, "validation_accuracy": val_score, "best_params": rs.best_params_}
        print(f"  {name}: CV={rs.best_score_:.4f}, Val={val_score:.4f}")
    return results

def svc_grid_search(X_train, y_train, X_val, y_val, param_grid):
    gs = GridSearchCV(SVC(probability=True, random_state=RS), param_grid, scoring="accuracy", cv=3, n_jobs=-1, verbose=0).fit(X_train, y_train)
    print(f"  SVC Grid - Val: {gs.best_estimator_.score(X_val, y_val):.4f}")
    return gs

def plot_tuning(search_results, model_name):
    results_df = pd.DataFrame(search_results.cv_results_)
    param_cols = [c for c in results_df.columns if c.startswith("param_")]
    if not param_cols: return
    n = len(param_cols)
    fig, axes = plt.subplots(max(1,(n+1)//2), 2 if n>1 else 1, figsize=(12,4*max(1,(n+1)//2)), sharey=True)
    axes_list = axes.flatten() if n>1 else [axes]
    fig.suptitle(f"Tuning: {model_name}")
    for i, p in enumerate(param_cols):
        if i >= len(axes_list): break
        name = p.replace("param_","")
        results_df[p+"_str"] = results_df[p].apply(lambda x: str(x) if not isinstance(x,(list,tuple,np.ndarray)) else str(tuple(x)))
        sns.boxplot(x=p+"_str", y="mean_test_score", data=results_df, ax=axes_list[i])
        axes_list[i].set_title(f"Score vs {name}"); axes_list[i].tick_params(axis="x", rotation=45)
    for j in range(i+1, len(axes_list)): fig.delaxes(axes_list[j])
    plt.tight_layout(); plt.show()

def evaluate(best_model, tuning_results, X_train, y_train, X_val, y_val, X_test, y_test, class_names):
    X_full = np.vstack([X_train, X_val])
    y_full = np.concatenate([y_train, y_val])
    if "SVC_fine_tuned" in tuning_results:
        model = tuning_results["SVC_fine_tuned"]
        if isinstance(model, dict): model = model.get("grid_search_object", list(model.values())[0])
    else:
        model = models_and_params()[best_model]["model"].set_params(**tuning_results[best_model]["best_params"])
    model.fit(X_full, y_full)
    y_pred = model.predict(X_test)
    print(f"  Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred, target_names=class_names))
    plt.figure(figsize=(10,8))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Confusion Matrix ({best_model})"); plt.ylabel("True"); plt.xlabel("Predicted"); plt.show()
    return model

if __name__ == "__main__":
    import matplotlib; matplotlib.use('Agg')
    assert "LogisticRegression" in models_and_params()
    X = np.random.randn(30, 4)
    y = np.random.randint(0, 3, 30)
    results = tune(X, y, X, y, "test")
    assert len(results) > 0
    model = evaluate("LogisticRegression", results, X, y, X, y, X, y, ["a","b","c"])
    assert model is not None
    print("classification OK")
