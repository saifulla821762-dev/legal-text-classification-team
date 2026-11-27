"""Metrics and output helpers."""
from pathlib import Path
import json
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, ConfusionMatrixDisplay


def metrics(y_true, y_pred):
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    return {"accuracy": float(accuracy_score(y_true, y_pred)),
            "weighted_precision": float(p), "weighted_recall": float(r),
            "weighted_f1": float(f)}


def save_results(results, y_true, predictions, labels, names):
    out = Path("results")
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    cm = confusion_matrix(y_true, predictions, labels=labels)
    display_names = [names.get(int(x), str(x)) for x in labels]
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_names)
    fig, ax = plt.subplots(figsize=(13, 11))
    disp.plot(ax=ax, xticks_rotation=90, colorbar=False)
    fig.tight_layout()
    fig.savefig(out / "confusion_matrix.png", dpi=160)
    plt.close(fig)
