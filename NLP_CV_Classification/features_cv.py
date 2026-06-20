import os, random
import cv2
import numpy as np
import pandas as pd
import albumentations as A
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

TARGET_SIZE = 224

def get_extractor(method="sift"):
    return cv2.SIFT_create() if method == "sift" else cv2.ORB_create(nfeatures=2000)

def extract_descriptors(path, extractor, aug_config=None, target_size=TARGET_SIZE):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None: return []
    gray = A.Compose([A.Resize(target_size,target_size), A.ToGray(p=1.0)])(image=img)["image"]
    _, des = extractor.detectAndCompute(gray, None)
    results = [des] if des is not None else []
    if aug_config:
        for _ in range(aug_config["num_augs_per_image"]):
            aug = random.choice(aug_config["pipelines"])(image=img)["image"]
            _, d = extractor.detectAndCompute(aug, None)
            if d is not None: results.append(d)
    return results

def bovw_histograms(des_list, vocab):
    n = vocab.n_clusters
    h = np.zeros((len(des_list), n), dtype=np.float32)
    for i, des in enumerate(des_list):
        if des is not None and len(des) > 0:
            h[i] = np.histogram(vocab.predict(des), bins=np.arange(n+1))[0]
    return h

def run_feature_experiments(train_df, val_df, image_dir, methods, vocab_sizes, aug_config):
    results, all_feat = [], {}
    for method in methods:
        ext = get_extractor(method)
        train_des, y_train = [], []
        for _, row in tqdm(train_df.iterrows(), total=len(train_df)):
            for des in extract_descriptors(os.path.join(image_dir, row["image"]), ext, aug_config):
                if des is not None and des.shape[0] > 0: train_des.append(des); y_train.append(row["target"])
        valid = [d for d in train_des if d is not None and d.shape[0] > 0]
        if not valid: raise ValueError(f"No valid features from {method}")
        all_train = np.vstack(valid); y_train = np.array(y_train)
        val_des, y_val = [], []
        for _, row in tqdm(val_df.iterrows(), total=len(val_df)):
            for des in extract_descriptors(os.path.join(image_dir, row["image"]), ext):
                if des is not None and des.shape[0] > 0: val_des.append(des); y_val.append(row["target"])
        if not val_des: raise ValueError(f"No valid val features from {method}")
        y_val = np.array(y_val)
        for n in vocab_sizes:
            key = f"{method}_{n}"
            km = MiniBatchKMeans(n_clusters=n, random_state=42, batch_size=256, n_init=10).fit(all_train)
            X_t = bovw_histograms(train_des, km); X_v = bovw_histograms(val_des, km)
            tfidf = TfidfTransformer(sublinear_tf=True).fit(X_t)
            X_t = tfidf.transform(X_t).toarray(); X_v = tfidf.transform(X_v).toarray()
            acc = LogisticRegression(random_state=42, solver="liblinear", max_iter=1000).fit(X_t, y_train).score(X_v, y_val)
            results.append({"Method": method.upper(), "Vocab Size": n, "Val. Accuracy": acc})
            all_feat[key] = {"X_train": X_t, "y_train": y_train, "X_val": X_v, "y_val": y_val}
    return pd.DataFrame(results), all_feat

if __name__ == "__main__":
    assert get_extractor("sift") is not None
    assert get_extractor("orb") is not None
    print("features_cv OK")
