import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef
)

def evaluate_model(model, X_test, y_test, n_classes):
    y_pred = model.predict(X_test)

    auc = np.nan
    if hasattr(model, "predict_proba"):
        try:
            y_prob = model.predict_proba(X_test)
            if n_classes  == 2:
                auc = roc_auc_score(y_test, y_prob[:, 1])
            else:
                auc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro")
        except:
            pass

    avg = "binary" if n_classes  == 2 else "weighted"

    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": auc,
        "Precision": precision_score(y_test, y_pred, average=avg, zero_division=0),
        "Recall": recall_score(y_test, y_pred, average=avg, zero_division=0),
        "F1": f1_score(y_test, y_pred, average=avg, zero_division=0),
        "MCC": matthews_corrcoef(y_test, y_pred)
    }