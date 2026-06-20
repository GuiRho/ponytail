import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

BERT_MODEL = "all-MiniLM-L6-v2"
_bert = None

def _load_bert():
    global _bert
    if _bert is None:
        try: _bert = SentenceTransformer(BERT_MODEL)
        except Exception as e: print(f"Warning: Could not load BERT model: {e}")
    return _bert

def _df(features, vectorizer, method, col, ids):
    if features.size == 0: return pd.DataFrame()
    if method in ("BoW","TF-IDF"):
        data = features.toarray() if hasattr(features,"toarray") else features
        return pd.DataFrame(data, columns=vectorizer.get_feature_names_out(), index=ids)
    cols = [f"{method}_{col}_feature_{i}" for i in range(features.shape[1])]
    return pd.DataFrame(features, columns=cols, index=ids)

def get_bow(df, text_col, max_features=5000):
    corpus, ids = df[text_col].tolist(), df["uniq_id"].tolist()
    if not corpus or all(not str(s).strip() for s in corpus): return pd.DataFrame(), {}
    v = CountVectorizer(max_features=max_features)
    X = v.fit_transform(corpus)
    return _df(X, v, "BoW", text_col, ids), {"method":"BoW","shape":X.shape,"vocab":len(v.vocabulary_)}

def get_tfidf(df, text_col, max_features=5000):
    corpus, ids = df[text_col].tolist(), df["uniq_id"].tolist()
    if not corpus or all(not str(s).strip() for s in corpus): return pd.DataFrame(), {}
    v = TfidfVectorizer(max_features=max_features)
    X = v.fit_transform(corpus)
    return _df(X, v, "TF-IDF", text_col, ids), {"method":"TF-IDF","shape":X.shape,"vocab":len(v.vocabulary_)}

def get_w2v(df, tokens_col, vector_size=100, window=5, min_count=2, workers=4, epochs=10):
    corpus, ids = df[tokens_col].tolist(), df["uniq_id"].tolist()
    train = [t for t in corpus if len(t) >= min_count]
    if not train: return pd.DataFrame(), {}
    m = Word2Vec(sentences=train, vector_size=vector_size, window=window, min_count=min_count, workers=workers, sg=1, epochs=epochs)
    emb = np.array([np.mean([m.wv[w] for w in t if w in m.wv], axis=0) if any(w in m.wv for w in t) else np.zeros(vector_size) for t in corpus])
    return _df(emb, m, "Word2Vec", tokens_col, ids), {"method":"Word2Vec","shape":emb.shape,"vocab":len(m.wv)}

def get_bert(df, text_col):
    m = _load_bert()
    if m is None: return pd.DataFrame(), {}
    corpus, ids = df[text_col].tolist(), df["uniq_id"].tolist()
    if not corpus or all(not str(s).strip() for s in corpus): return pd.DataFrame(), {}
    emb = m.encode(corpus, show_progress_bar=False, convert_to_numpy=True)
    return _df(emb, None, "BERT", text_col, ids), {"method":"BERT","shape":emb.shape,"dim":emb.shape[1]}

def extract_features(df):
    features, info = {}, {}
    if df.empty or "uniq_id" not in df.columns: return features, info
    for method, col, _ in [("BoW","processed_text_lemm",""),("BoW","processed_text_stem",""),("TF-IDF","processed_text_lemm",""),("TF-IDF","processed_text_stem",""),("Word2Vec","processed_tokens_lemm","tokens"),("Word2Vec","processed_tokens_stem","tokens"),("BERT","cleaned_description","")]:
        key = f"{method}_{col.replace('processed_','').replace('text_','').replace('tokens_','')}"
        if method == "Word2Vec": f, i = get_w2v(df, col)
        elif method == "BERT": f, i = get_bert(df, col)
        elif method == "BoW": f, i = get_bow(df, col)
        elif method == "TF-IDF": f, i = get_tfidf(df, col)
        else: continue
        if not f.empty: features[key] = f; info[key] = i
    return features, info

if __name__ == "__main__":
    df = pd.DataFrame({"uniq_id": ["a","b"], "processed_text_lemm": ["hello world","foo bar"], "processed_text_stem": ["hello world","foo bar"], "processed_tokens_lemm": [["hello","world"],["foo","bar"]], "processed_tokens_stem": [["hello","world"],["foo","bar"]], "cleaned_description": ["hello world","foo bar"]})
    feat, inf = extract_features(df)
    assert "BoW_processed_text_lemm" in feat
    bow, _ = get_bow(df, "processed_text_lemm")
    assert not bow.empty
    tfidf, _ = get_tfidf(df, "processed_text_lemm")
    assert not tfidf.empty
    print("features OK")
