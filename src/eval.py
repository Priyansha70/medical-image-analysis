from typing import Dict, Tuple
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
)


@torch.no_grad()
def predict_probs(model, loader, device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_probs = []
    all_y = []

    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[:, 1]
        all_probs.append(probs.detach().cpu().numpy())
        all_y.append(y.numpy())

    return np.concatenate(all_probs), np.concatenate(all_y)


def compute_metrics(probs: np.ndarray, y_true: np.ndarray, threshold: float = 0.5) -> Dict:
    y_pred = (probs >= threshold).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probs)) if len(np.unique(y_true)) > 1 else None,
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "threshold": float(threshold),
    }

    return metrics
