import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from sklearn.mixture import GaussianMixture
from scipy.optimize import linear_sum_assignment

RS = 42

def align_clusters(cluster_labels, true_labels):
    cm = confusion_matrix(true_labels, cluster_labels)
    row_ind, col_ind = linear_sum_assignment(-cm)
    mapping = {col_ind[i]: row_ind[i] for i in range(len(row_ind))}
    return np.array([mapping.get(c, -1) for c in cluster_labels])

def kmeans_metrics(data_matrix, k_range, dataset_name="features"):
    if isinstance(data_matrix, pd.DataFrame): data_matrix = data_matrix.values
    if data_matrix.shape[0] < 2: return
    ks = sorted(k_range)
    curves = {"inertia": [], "silhouette": [], "calinski_harabasz": [], "davies_bouldin": []}
    for k in ks:
        if k < 1 or k > data_matrix.shape[0]:
            for v in curves.values(): v.append(np.nan); continue
        try:
            km = KMeans(n_clusters=k, random_state=RS, n_init="auto", algorithm="lloyd").fit(data_matrix)
            curves["inertia"].append(km.inertia_)
            if k > 1 and len(np.unique(km.labels_)) > 1:
                curves["silhouette"].append(metrics.silhouette_score(data_matrix, km.labels_))
                curves["calinski_harabasz"].append(metrics.calinski_harabasz_score(data_matrix, km.labels_))
                curves["davies_bouldin"].append(metrics.davies_bouldin_score(data_matrix, km.labels_))
            else:
                curves["silhouette"].append(np.nan); curves["calinski_harabasz"].append(np.nan); curves["davies_bouldin"].append(np.nan)
        except:
            for v in curves.values(): v.append(np.nan)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"K-Means Metrics for {dataset_name}")
    for i, (name, scores) in enumerate(curves.items()):
        ax = axes[i//2, i%2]
        ax.plot(ks, scores, marker="o"); ax.set_title(name); ax.set_xlabel("k"); ax.set_xticks(ks)
    plt.tight_layout(rect=[0,0,1,0.95]); plt.show()

def evaluate_clustering(features, labels, class_names, feature_name, dataset_type, pca_ratios=None):
    if pca_ratios is None: pca_ratios = [0.99, 0.95, 0.5]
    n_classes = len(class_names)
    all_results, best = [], {"score": -1.0}
    spaces = [("Full", features)]
    for ratio in pca_ratios:
        X_pca = PCA(n_components=ratio, random_state=RS).fit_transform(features)
        spaces.append((f"PCA ({ratio:.0%})", X_pca))
    for space_name, X in spaces:
        for model_name, model in [("KMeans", KMeans(n_clusters=n_classes, random_state=RS, n_init=10)), ("GMM", GaussianMixture(n_components=n_classes, random_state=RS, n_init=5))]:
            cl = model.fit_predict(X)
            aligned = align_clusters(cl, labels)
            ari = metrics.adjusted_rand_score(labels, cl)
            nmi = metrics.normalized_mutual_info_score(labels, cl)
            all_results.append({"Dataset": dataset_type, "Space": space_name, "Dim": X.shape[1], "Method": model_name, "ARI": ari, "NMI": nmi})
            if ari > best["score"]: best = {"score": ari, "title": f"Best CM: {model_name} on {space_name} (ARI: {ari:.3f})", "true_labels": labels, "aligned_labels": aligned}
    if best["score"] > -1:
        cm = confusion_matrix(best["true_labels"], best["aligned_labels"])
        plt.figure(figsize=(10,8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
        plt.title(best["title"]); plt.ylabel("True"); plt.xlabel("Predicted"); plt.show()
    return pd.DataFrame(all_results)

def kmeans_nlp(feature_df, true_labels, category_names, feature_name, n_clusters):
    if feature_df.empty or feature_df.shape[0] < n_clusters or n_clusters < 2: return {}
    aligned = true_labels.loc[feature_df.index]
    try:
        km = KMeans(n_clusters=n_clusters, random_state=RS, n_init="auto", algorithm="lloyd").fit(feature_df.values)
    except Exception as e:
        print(f"  KMeans error: {e}"); return {}
    cl = km.labels_
    ari = metrics.adjusted_rand_score(aligned, cl)
    nmi = metrics.normalized_mutual_info_score(aligned, cl)
    print(f"  {feature_name}: ARI={ari:.4f}, NMI={nmi:.4f}")
    return {"ARI": ari, "NMI": nmi}

if __name__ == "__main__":
    import matplotlib; matplotlib.use('Agg')
    X = np.random.randn(50, 4)
    y = np.random.randint(0, 3, 50)
    result = evaluate_clustering(X, y, ["a","b","c"], "test", "test")
    assert isinstance(result, pd.DataFrame) and len(result) > 0
    aligned = align_clusters(np.array([0,0,1,1]), np.array([1,1,0,0]))
    assert len(aligned) == 4
    print("clustering OK")
