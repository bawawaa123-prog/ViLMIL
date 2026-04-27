import math

import numpy as np
from sklearn.metrics import average_precision_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score


def compute_classification_metrics(labels, probs, preds, n_classes):
    labels = np.asarray(labels).astype(int).reshape(-1)
    preds = np.asarray(preds).astype(int).reshape(-1)
    probs = np.asarray(probs)

    acc = float(np.mean(labels == preds)) if len(labels) > 0 else float("nan")
    f1 = float(f1_score(labels, preds, average="macro", zero_division=0)) if len(labels) > 0 else float("nan")

    unique_labels = np.unique(labels)
    if len(unique_labels) <= 1:
        auc = -1.0
    elif n_classes == 2:
        auc = float(roc_auc_score(labels, probs[:, 1]))
    else:
        auc = float(roc_auc_score(labels, probs, multi_class="ovr"))

    metrics = {
        "auc": auc,
        "acc": acc,
        "f1": f1,
        "balanced_acc": -1.0,
        "sensitivity": -1.0,
        "specificity": -1.0,
        "pr_auc": -1.0,
    }

    if n_classes != 2:
        return metrics

    if len(unique_labels) > 1:
        metrics["balanced_acc"] = float(balanced_accuracy_score(labels, preds))
        metrics["pr_auc"] = float(average_precision_score(labels, probs[:, 1]))

    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    sensitivity_den = tp + fn
    specificity_den = tn + fp

    if sensitivity_den > 0:
        metrics["sensitivity"] = float(tp / sensitivity_den)
    if specificity_den > 0:
        metrics["specificity"] = float(tn / specificity_den)

    if metrics["sensitivity"] >= 0 and metrics["specificity"] >= 0:
        metrics["balanced_acc"] = float((metrics["sensitivity"] + metrics["specificity"]) / 2.0)

    return metrics


def summarize_metric_list(values):
    clean_values = [float(v) for v in values if not math.isnan(float(v))]
    if not clean_values:
        return float("nan"), float("nan")
    arr = np.asarray(clean_values, dtype=float)
    return float(np.mean(arr)), float(np.std(arr))
