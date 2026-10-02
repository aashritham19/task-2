import json, joblib, numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from .features import CAT_COLS, NUM_COLS, FEATURES, build_features

# Policy numbers (ops-policy.pdf s4, s7)
RETURN_COST = 1150.0
CALL_COST = 45.0
CALL_PREVENTS = 0.35
HOLD_CANCEL = 0.12
BREAK_EVEN = CALL_COST / (CALL_PREVENTS * RETURN_COST)   # ~= 0.112


def make_model(C=0.1):
    ct = ColumnTransformer([
        ("c", OneHotEncoder(handle_unknown="ignore", min_frequency=30), CAT_COLS),
        ("n", Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler())]), NUM_COLS)])
    return make_pipeline(ct, LogisticRegression(C=C, max_iter=3000))


def load_ref(data_dir="data"):
    return pd.read_csv(f"{data_dir}/customers.csv"), pd.read_csv(f"{data_dir}/products.csv")



def mean_design(model, feats: pd.DataFrame):
    X = model[0].transform(feats[FEATURES])
    X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    return X.mean(axis=0)


def contributions(model, mean_x, row_features: pd.DataFrame):
    """Log-odds contribution of every original column vs an average order (sums one-hot pieces)."""
    X = model[0].transform(row_features[FEATURES][:1])
    X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    names = model[0].get_feature_names_out()
    contrib = (X[0] - mean_x) * model[1].coef_[0]
    out = {}
    for n, c in zip(names, contrib):
        col = n.split("__", 1)[1]
        for f in sorted(FEATURES, key=len, reverse=True):
            if col == f or col.startswith(f + "_"):
                out[f] = out.get(f, 0.0) + float(c); break
    return out
