import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from src.preprocessing import build_preprocessors
from src.evaluation import evaluate_model

# Safe XGBoost
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except:
    XGB_AVAILABLE = False


def train_and_evaluate(df, target_col, test_size=0.2, include_xgb=True):

    X = df.drop(columns=[target_col])
    y_raw = df[target_col]

    # Encode target
    if y_raw.dtype == "object":
        le = LabelEncoder()
        y = le.fit_transform(y_raw.astype(str))
    else:
        y = y_raw.values

    n_classes = len(set(y))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    pre_sparse, pre_dense = build_preprocessors(X_train)

    models = {
        "Logistic Regression": Pipeline([
            ("prep", pre_sparse),
            ("model", LogisticRegression(max_iter=1000))
        ]),
        "Decision Tree": Pipeline([
            ("prep", pre_sparse),
            ("model", DecisionTreeClassifier())
        ]),
        "KNN": Pipeline([
            ("prep", pre_sparse),
            ("model", KNeighborsClassifier())
        ]),
        "Naive Bayes": Pipeline([
            ("prep", pre_dense),
            ("model", GaussianNB())
        ]),
        "Random Forest": Pipeline([
            ("prep", pre_sparse),
            ("model", RandomForestClassifier(n_estimators=300))
        ])
    }

    if include_xgb and XGB_AVAILABLE:
        models["XGBoost"] = Pipeline([
            ("prep", pre_sparse),
            ("model", XGBClassifier(eval_metric="mlogloss"))
        ])

    results = []

    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        metrics = evaluate_model(pipe, X_test, y_test, n_classes)
        metrics["Model"] = name
        results.append(metrics)

    results_df = pd.DataFrame(results)
    results_df = results_df[
        ["Model", "Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
    ].sort_values(by="AUC", ascending=False)

    return results_df, n_classes