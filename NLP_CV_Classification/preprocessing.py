import re, string
from typing import Any
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

RS = 42
_lemmatizer = WordNetLemmatizer()
_stemmer = PorterStemmer()
_stop_words = set(stopwords.words("english"))

def extract_category(text: str, level: int = 0) -> str:
    try:
        match = re.findall(r'"([^"]*)"', text)
        if not match: return "Unknown"
        segments = [s.strip() for s in match[0].split(">>")]
        return segments[level] if level < len(segments) else "Unknown"
    except (IndexError, TypeError):
        return "Unknown"

def _get_wordnet_pos(word: str) -> str:
    tag = nltk.pos_tag([word])[0][1][0].upper()
    return {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}.get(tag, wordnet.NOUN)

def clean_text(text: str) -> str:
    text = re.sub(r"http\S+|www\S+|<.*?>+|\d+", "", str(text).lower())
    text = text.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", text).strip()

def tokenize(text: str, method: str = "lemm") -> list[str]:
    text = re.sub(r"\[.*?\]|http\S+|www\S+|<.*?>+|\d+", "", str(text).lower())
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = word_tokenize(re.sub(r"\s+", " ", text).strip())
    result = []
    for t in tokens:
        if t not in _stop_words and len(t) > 2:
            if method == "lemm": result.append(_lemmatizer.lemmatize(t, _get_wordnet_pos(t)))
            elif method == "stem": result.append(_stemmer.stem(t))
            else: result.append(t)
    return result

def text_stats(df, text_col, tokens_col=None, prefix=""):
    stats = {}
    cl = df[text_col].apply(lambda x: len(str(x)))
    for k in ["mean","min","max"]: stats[f"{prefix}char_length_{k}"] = float(getattr(cl, k)())
    if tokens_col and tokens_col in df.columns:
        tc = df[tokens_col].apply(len)
        all_t = [t for s in df[tokens_col] for t in s]
    else:
        temp = df[text_col].apply(lambda x: word_tokenize(str(x).lower().translate(str.maketrans("","",string.punctuation)).strip()))
        tc = temp.apply(len)
        all_t = [t for s in temp for t in s]
    for k in ["mean","min","max"]: stats[f"{prefix}token_count_{k}"] = float(getattr(tc, k)())
    stats[f"{prefix}vocabulary_size"] = float(len(set(all_t)))
    return stats

def preprocess_text(df, desc_col, cat_col, cat_level=0):
    info = dict(initial_rows=len(df))
    df = df[["uniq_id", desc_col, cat_col]].copy().dropna(subset=[desc_col, cat_col])
    info["rows_after_nan_drop"] = len(df)
    if df.empty: return df, info
    df["main_category"] = df[cat_col].apply(lambda x: extract_category(x, cat_level))
    df = df[df["main_category"] != "Unknown"]
    info["rows_after_category_filter"] = len(df)
    if df.empty: return df, info
    le = LabelEncoder(); df["target"] = le.fit_transform(df["main_category"])
    info.update(unique_categories_count=len(le.classes_), category_names=le.classes_.tolist())
    df["cleaned_description"] = df[desc_col].apply(clean_text)
    for m in ("lemm","stem"):
        df[f"processed_tokens_{m}"] = df[desc_col].apply(lambda x, mm=m: tokenize(x, mm))
        df[f"processed_text_{m}"] = df[f"processed_tokens_{m}"].apply(lambda t: " ".join(t))
    df = df[(df["processed_tokens_lemm"].apply(len)>0)&(df["processed_tokens_stem"].apply(len)>0)]
    info["rows_after_empty_filter"] = len(df)
    return df, info

def load_preprocess_cv(data_file, image_col, cat_col):
    le = LabelEncoder()
    df = pd.read_csv(data_file)[[image_col, cat_col]].copy().dropna()
    df["main_category"] = df[cat_col].apply(lambda x: extract_category(x))
    df = df[df["main_category"] != "Unknown"]
    df["target"] = le.fit_transform(df["main_category"])
    return df[["image","main_category","target"]].copy().reset_index(drop=True), le.classes_, le

def train_val_test_split(df, stratify_col="target", test_size=0.15, val_size=0.15):
    tv, test = train_test_split(df, test_size=test_size, random_state=RS, stratify=df[stratify_col])
    train, val = train_test_split(tv, test_size=val_size/(1-test_size), random_state=RS, stratify=tv[stratify_col])
    return train, val, test

if __name__ == "__main__":
    assert extract_category('["Food >> Beverages >> Wine"]', 0) == "Food"
    assert extract_category('["Food >> Beverages >> Wine"]', 2) == "Wine"
    assert extract_category("", 0) == "Unknown"
    assert clean_text("HELLO World") == "hello world"
    assert clean_text("check http://example.com") == "check"
    assert clean_text("<p>hello</p>") == "hello"
    assert clean_text("hello! 123 world?") == "hello world"
    assert clean_text("hello   world") == "hello world"
    assert clean_text("") == ""
    assert clean_text(12345) == ""
    df = pd.DataFrame({"uniq_id": [1,2], "desc": ["foo bar","baz"], "cat": ['["A >> B"]','["C >> D"]']})
    result, info = preprocess_text(df, "desc", "cat")
    assert not result.empty
    assert info["unique_categories_count"] == 2
    df_cv = pd.DataFrame({"image": [f"i_{i}.jpg" for i in range(100)], "target": [i%3 for i in range(100)]})
    tr, vl, te = train_val_test_split(df_cv)
    assert len(tr)+len(vl)+len(te) == 100
    print("preprocessing OK")
