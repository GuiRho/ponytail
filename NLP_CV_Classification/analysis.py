from typing import Any
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE
import umap

RS = 42

def analyze_features(feature_df: pd.DataFrame, name: str) -> dict[str, Any]:
    if feature_df.empty: return {}
    stats = {"name": name, "shape": feature_df.shape, "num_features": feature_df.shape[1], "num_samples": feature_df.shape[0], "nan_count": int(feature_df.isnull().sum().sum())}
    if feature_df.shape[1] > 0:
        vals = feature_df.values
        for k in ["mean","std","min","max"]: stats[f"{k}_of_features"] = float(getattr(vals, k)())
    if "BoW" in name or "TF-IDF" in name:
        total = feature_df.shape[0] * feature_df.shape[1]
        if total > 0:
            stats["sparsity_ratio"] = int((feature_df == 0).sum().sum()) / total
            stats["avg_non_zero_per_doc"] = float((feature_df != 0).sum(axis=1).mean())
    return stats

def plot_2d(X, labels, class_names, title):
    plt.figure(figsize=(12,10))
    scatter = plt.scatter(X[:,0], X[:,1], c=labels, cmap="viridis", s=10, alpha=0.7)
    plt.legend(scatter.legend_elements()[0], [class_names[int(l)] for l in np.unique(labels)], title="Categories", bbox_to_anchor=(1.05,1))
    plt.title(title); plt.grid(True); plt.tight_layout(); plt.show()

def plot_3d(X, labels, class_names, title):
    fig = plt.figure(figsize=(12,10))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(X[:,0], X[:,1], X[:,2], c=labels, cmap="viridis", s=10, alpha=0.6)
    ax.set_title(title); plt.tight_layout(); plt.show()

def plot_cm(cm, class_names, title):
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title(title); plt.ylabel("True"); plt.xlabel("Predicted"); plt.show()

def plot_tsne_annotated(feature_df, target_labels, category_names, feature_set_name):
    if feature_df.empty or feature_df.shape[0] < 2: return
    p = min(30, feature_df.shape[0]-1)
    if p < 5: return
    aligned = target_labels.loc[feature_df.index]
    try:
        reduced = TSNE(n_components=2, random_state=RS, perplexity=p, n_iter=300).fit_transform(feature_df)
    except ValueError:
        return
    fig, ax = plt.subplots(figsize=(20,20))
    sns.scatterplot(x=reduced[:,0], y=reduced[:,1], hue=aligned, palette=sns.color_palette("viridis", n_colors=len(category_names)), alpha=0.7, s=100, ax=ax)
    ax.set_title(f"t-SNE of {feature_set_name}", fontsize=24); ax.set_aspect("equal")
    colors = sns.color_palette("viridis", n_colors=len(category_names))
    for label in sorted(aligned.unique()):
        pts = reduced[aligned == label]
        if len(pts) > 0:
            name = category_names[label] if label < len(category_names) else str(label)
            ax.text(pts[:,0].mean(), pts[:,1].mean(), name, color="white", fontsize=16, weight="bold", ha="center", va="center", bbox=dict(facecolor=colors[label%len(colors)], alpha=0.7, edgecolor="none", boxstyle="round,pad=0.4"), path_effects=[pe.Stroke(linewidth=3, foreground="black"), pe.Normal()])
    plt.show()

def run_visualization(features, labels, class_names, feature_name, dataset_type):
    print(f"\n--- Viz: {feature_name} ({dataset_type}) ---")
    suffix = f"- {feature_name} ({dataset_type})"
    n = features.shape[0]
    reducer = umap.UMAP(n_components=2, random_state=RS)
    plot_2d(reducer.fit_transform(features), labels, class_names, f"2D UMAP {suffix}")
    p = min(30, n-1)
    if p > 0:
        plot_2d(TSNE(n_components=2, random_state=RS, perplexity=p, n_iter=300).fit_transform(features), labels, class_names, f"2D t-SNE {suffix}")
    reducer_3d = umap.UMAP(n_components=3, random_state=RS)
    plot_3d(reducer_3d.fit_transform(features), labels, class_names, f"3D UMAP {suffix}")
    if p > 0:
        plot_3d(TSNE(n_components=3, random_state=RS, perplexity=p, n_iter=400).fit_transform(features), labels, class_names, f"3D t-SNE {suffix}")

def analysis_pipeline(features_dataframes, processed_df, processing_info):
    print("\n===== ANALYSIS =====")
    if not features_dataframes: return {}
    target_labels = processed_df.set_index("uniq_id")["target"]
    category_names = processing_info.get("category_names", [])
    results = {}
    for name, df in features_dataframes.items():
        if not df.empty:
            results[name] = analyze_features(df, name)
            if not target_labels.empty and df.index.isin(target_labels.index).all():
                plot_tsne_annotated(df, target_labels, category_names, name)
    return results

if __name__ == "__main__":
    import matplotlib; matplotlib.use('Agg')
    df = pd.DataFrame(np.random.randn(20, 4), columns=[f"f{i}" for i in range(4)])
    stats = analyze_features(df, "test")
    assert stats["num_features"] == 4 and stats["num_samples"] == 20
    cm = np.array([[5,1],[2,4]])
    plot_cm(cm, ["a","b"], "test")
    print("analysis OK")
