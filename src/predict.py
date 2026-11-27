"""Load a saved model and predict a case's issue area."""
from pathlib import Path
import joblib
from .preprocessing import clean_text


def predict_text(text):
    path = Path("models/tfidf_lda_logreg.joblib")
    if not path.exists():
        raise FileNotFoundError("Train first: python main.py train")
    bundle = joblib.load(path)
    prediction = int(bundle["model"].predict([clean_text(text)])[0])
    return bundle["label_names"].get(prediction, str(prediction))
