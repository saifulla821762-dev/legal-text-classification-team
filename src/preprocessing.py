"""Dataset loading and text/label preparation."""
from pathlib import Path
import re
import textacy.datasets

DATA_DIR = Path("data")


def load_dataset():
    dataset = textacy.datasets.SupremeCourt(data_dir=DATA_DIR)
    dataset.download()
    records = []
    for text, meta in dataset.records():
        label = meta.get("issue_area")
        if label is None or int(label) < 1 or not text or not text.strip():
            continue
        records.append({
            "text": clean_text(text),
            "label": int(label),
            "case_name": meta.get("case_name", ""),
            "decision_date": meta.get("decision_date", ""),
        })
    return records, dataset.issue_area_codes


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text.strip()


def label_names(code_map):
    return {int(k): str(v) for k, v in code_map.items() if v is not None and int(k) > 0}
