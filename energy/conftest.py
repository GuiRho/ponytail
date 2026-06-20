import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "num_feat_1": [10.0, 20.0, 30.0, 40.0, 1000.0],
        "num_feat_2": [1.0, 2.0, 3.0, 4.0, 5.0],
        "cat_feat": ["A", "B", "A", "C", "B"],
        "target": [100, 200, 300, 400, 500],
    })


@pytest.fixture
def wide_df():
    rng = np.random.default_rng(42)
    n = 50
    return pd.DataFrame({
        "x1": rng.normal(0, 1, n), "x2": rng.normal(0, 1, n),
        "x3": rng.normal(0, 1, n), "x4": rng.uniform(0, 1, n),
        "cat": ["Z" if i % 3 == 0 else ("Y" if i % 3 == 1 else "X") for i in range(n)],
        "y": rng.normal(10, 2, n),
    })


@pytest.fixture
def missing_df():
    return pd.DataFrame({
        "a": [1.0, 2.0, np.nan, 4.0, 5.0],
        "b": [np.nan, 2.0, 3.0, np.nan, 5.0],
        "cat": ["X", "Y", "X", "Y", "X"],
        "target": [10, 20, 30, 40, 50],
    })


@pytest.fixture
def cat_correlated_df():
    return pd.DataFrame({
        "cat1": ["a", "a", "a", "b", "b", "b"],
        "cat2": ["x", "x", "x", "y", "y", "y"],
        "cat3": ["p", "q", "p", "p", "q", "q"],
    })


@pytest.fixture
def model_features():
    rng = np.random.default_rng(0)
    n = 50
    return pd.DataFrame({
        "f1": rng.normal(5, 1, n), "f2": rng.normal(10, 2, n),
        "f3": rng.uniform(0, 10, n),
        "target": rng.normal(50, 10, n),
    })
