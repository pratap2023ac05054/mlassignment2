# app.py
# Streamlit app to train/evaluate models on global_cars_enhanced.csv and show metrics on button click.
#
# Expected project structure:
# project/
# ├─ data/
# │   └─ global_cars_enhanced.csv
# ├─ src/                (optional)
# └─ app.py              (this file)

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

# -----------------------
# Safe XGBoost Import
# -----------------------
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False


# ===============================
# Helpers
# ===============================
def project_root() -> str:
    # app.py in project root
    return os.path.dirname(os.path.abspath(__file__))


def default_csv_path() -> str:
    return os.path.join(project_root(), "data", "global_cars_enhanced.csv")


@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def build_preprocessors(X: pd.DataFrame):
    num_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    numeric = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # Sparse one-hot (efficient)
    try:
        ohe_sparse = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        ohe_sparse = OneHotEncoder(handle_unknown="ignore", sparse=True)

    categorical_sparse = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", ohe_sparse)
    ])

    pre_sparse = ColumnTransformer(
        transformers=[
            ("num", numeric, num_cols),
            ("cat", categorical_sparse, cat_cols),
        ],
        remainder="drop"
    )

    # Dense one-hot (GaussianNB needs dense)
    try:
        ohe_dense = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe_dense = OneHotEncoder(handle_unknown="ignore", sparse=False)

    categorical_dense = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", ohe_dense)
    ])

    pre_dense = ColumnTransformer(
        transformers=[
            ("num", numeric, num_cols),
            ("cat", categorical_dense, cat_cols),
        ],
        remainder="drop"
    )

    return pre_sparse, pre_dense


def prepare_target(y_raw: pd.Series, binarize_continuous: bool, threshold: float):
    """
    Returns: y_encoded, n_classes, label_encoder_or_none
    """
    if binarize_continuous:
        y_num = pd.to_numeric(y_raw, errors="coerce")
        if y_num.isna().any():
            raise ValueError("Selected target has non-numeric values; cannot binarize as continuous.")
        y_bin = (y_num >= threshold).astype(int).to_numpy()
        return y_bin, 2, None

    if pd.api.types.is_numeric_dtype(y_raw):
        y_vals = y_raw.to_numpy()
        classes = np.unique(y_vals[~pd.isna(y_vals)])
        return y_vals, len(classes), None

    le = LabelEncoder()
    y = le.fit_transform(y_raw.astype(str))
    return y, len(le.classes_), le


def evaluate_model(model, X_test, y_test, n_classes: int):
    y_pred = model.predict(X_test)

    auc = np.nan
    if hasattr(model, "predict_proba"):
        try:
            y_prob = model.predict_proba(X_test)
            if n_classes == 2:
                auc = roc_auc_score(y_test, y_prob[:, 1])
            else:
                auc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro")
        except Exception:
            auc = np.nan

    avg = "binary" if n_classes == 2 else "weighted"

    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": auc,
        "Precision": precision_score(y_test, y_pred, average=avg, zero_division=0),
        "Recall": recall_score(y_test, y_pred, average=avg, zero_division=0),
        "F1": f1_score(y_test, y_pred, average=avg, zero_division=0),
        "MCC": matthews_corrcoef(y_test, y_pred),
    }


def train_and_compare(df: pd.DataFrame, target_col: str, test_size: float, random_state: int,
                      include_xgb: bool, binarize_continuous: bool, threshold: float):
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")

    X = df.drop(columns=[target_col])
    y_raw = df[target_col]

    y, n_classes, le = prepare_target(y_raw, binarize_continuous=binarize_continuous, threshold=threshold)
    if n_classes < 2:
        raise ValueError("Target has < 2 classes; not suitable for classification.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if n_classes > 1 else None
    )

    pre_sparse, pre_dense = build_preprocessors(X_train)

    models = {
        "Logistic Regression": Pipeline([
            ("prep", pre_sparse),
            ("model", LogisticRegression(max_iter=1000))
        ]),
        "Decision Tree": Pipeline([
            ("prep", pre_sparse),
            ("model", DecisionTreeClassifier(random_state=random_state))
        ]),
        "KNN": Pipeline([
            ("prep", pre_sparse),
            ("model", KNeighborsClassifier(n_neighbors=5))
        ]),
        "Naive Bayes (Gaussian)": Pipeline([
            ("prep", pre_dense),
            ("model", GaussianNB())
        ]),
        "Random Forest": Pipeline([
            ("prep", pre_sparse),
            ("model", RandomForestClassifier(
                n_estimators=300,
                random_state=random_state,
                n_jobs=-1
            ))
        ]),
    }

    if include_xgb and XGB_AVAILABLE:
        models["XGBoost"] = Pipeline([
            ("prep", pre_sparse),
            ("model", XGBClassifier(
                n_estimators=400,
                learning_rate=0.05,
                max_depth=5,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=random_state,
                eval_metric="mlogloss" if n_classes > 2 else "logloss",
                n_jobs=-1,
                tree_method="hist"
            ))
        ])

    rows = []
    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        m = evaluate_model(pipe, X_test, y_test, n_classes=n_classes)
        m["Model"] = name
        rows.append(m)

    results = pd.DataFrame(rows)[["Model", "Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]]
    results = results.sort_values(by="AUC", ascending=False)

    class_map = None
    if le is not None:
        class_map = {i: c for i, c in enumerate(le.classes_)}

    return results, n_classes, class_map


# ===============================
# Streamlit UI
# ===============================
st.set_page_config(page_title="Car Dataset Model Comparison", layout="wide")

st.title("Model Evaluation Dashboard (LogReg / DT / KNN / Naive Bayes / RF / XGBoost)")
st.caption("Click the button to train models and display evaluation metrics on the UI.")

# Load data
st.sidebar.header("Data & Settings")

csv_path = default_csv_path()
if not os.path.exists(csv_path):
    st.error(f"Dataset not found at: {csv_path}\n\nPlace it at: data/global_cars_enhanced.csv")
    st.stop()

df = load_data(csv_path)

st.sidebar.success(f"Loaded: {os.path.basename(csv_path)}")
st.sidebar.write("Rows:", df.shape[0], " | Columns:", df.shape[1])

# Target selection
default_targets = [c for c in ["Price_Category", "Age_Category"] if c in df.columns]
fallback_default = default_targets[0] if default_targets else df.columns[-1]
target_col = st.sidebar.selectbox("Select Target Column", options=list(df.columns), index=list(df.columns).index(fallback_default))

# Optional binarization controls (useful if user selects continuous target like Efficiency_Score)
is_numeric_target = pd.api.types.is_numeric_dtype(df[target_col])
binarize_continuous = False
threshold = 0.50
if is_numeric_target:
    st.sidebar.subheader("Numeric target options")
    binarize_continuous = st.sidebar.checkbox("Binarize numeric target (>= threshold → 1 else 0)", value=False)
    threshold = st.sidebar.slider("Threshold", min_value=0.0, max_value=1.0, value=0.50, step=0.05)

test_size = st.sidebar.slider("Test size", min_value=0.1, max_value=0.4, value=0.2, step=0.05)
random_state = st.sidebar.number_input("Random state", min_value=0, max_value=9999, value=42, step=1)

include_xgb = st.sidebar.checkbox("Include XGBoost (if installed)", value=True)
if include_xgb and not XGB_AVAILABLE:
    st.sidebar.warning("XGBoost is not installed → it will be skipped.")

st.subheader("Dataset Preview")
st.dataframe(df.head(10), use_container_width=True)

colA, colB = st.columns([1, 2])
with colA:
    run = st.button("Train & Evaluate Models", type="primary")

with colB:
    st.info(
        "Tip: For this dataset, try **Price_Category** or **Age_Category** (multi-class). "
        "If you choose a numeric target like **Efficiency_Score**, enable binarization."
    )

if run:
    with st.spinner("Training models and computing metrics..."):
        try:
            results_df, n_classes, class_map = train_and_compare(
                df=df,
                target_col=target_col,
                test_size=test_size,
                random_state=int(random_state),
                include_xgb=include_xgb,
                binarize_continuous=binarize_continuous,
                threshold=float(threshold),
            )
        except Exception as e:
            st.error(f"Failed: {e}")
            st.stop()

    st.success(f"Done. Target '{target_col}' has {n_classes} class(es).")

    st.subheader("Model Comparison (sorted by AUC)")
    st.dataframe(results_df, use_container_width=True)

    # Download results
    csv_bytes = results_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download results as CSV",
        data=csv_bytes,
        file_name="model_comparison_results.csv",
        mime="text/csv",
    )

    if class_map and n_classes > 2:
        st.subheader("Class Mapping (encoded → original)")
        st.json(class_map)

st.markdown("---")
st.caption("Run:  streamlit run app.py")