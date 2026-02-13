from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def build_preprocessors(X):
    num_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    try:
        ohe_sparse = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except:
        ohe_sparse = OneHotEncoder(handle_unknown="ignore", sparse=True)

    categorical_sparse = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", ohe_sparse)
    ])

    pre_sparse = ColumnTransformer([
        ("num", numeric, num_cols),
        ("cat", categorical_sparse, cat_cols)
    ])

    try:
        ohe_dense = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except:
        ohe_dense = OneHotEncoder(handle_unknown="ignore", sparse=False)

    categorical_dense = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", ohe_dense)
    ])

    pre_dense = ColumnTransformer([
        ("num", numeric, num_cols),
        ("cat", categorical_dense, cat_cols)
    ])

    return pre_sparse, pre_dense