"""Project CLI: train model comparison or predict from text."""
import argparse
from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split
from src.preprocessing import load_dataset, label_names
from src.tfidf_lda_model import build_models
from src.doc2vec_model import fit_doc2vec
from src.evaluation import metrics, save_results


def train():
    print("Downloading/loading dataset...")
    records, code_map = load_dataset()
    texts = [r["text"] for r in records]
    labels = [r["label"] for r in records]
    names = label_names(code_map)
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.20, random_state=42, stratify=labels)
    results, all_predictions = {}, {}

    for name, model in build_models().items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        results[name] = metrics(y_test, preds)
        all_predictions[name] = preds
        Path("models").mkdir(exist_ok=True)
        safe = "tfidf_lda_logreg" if "Class-weighted" not in name else "tfidf_lda_weighted"
        joblib.dump({"model": model, "label_names": names}, f"models/{safe}.joblib")

    print("Training Doc2Vec (train documents only)...")
    d2v, clf, vectors = fit_doc2vec(X_train, X_test, y_train)
    d2v_preds = clf.predict(vectors)
    name = "Doc2Vec + Logistic Regression"
    results[name] = metrics(y_test, d2v_preds)
    all_predictions[name] = d2v_preds
    Path("models").mkdir(exist_ok=True)
    joblib.dump({"doc2vec": d2v, "classifier": clf, "label_names": names},
                "models/doc2vec_logreg.joblib")

    print("\nMODEL COMPARISON (weighted metrics)")
    for model_name, score in results.items():
        print(f"\n{model_name}")
        for metric_name, value in score.items():
            print(f"  {metric_name}: {value:.4f}")
    best_name = max(results, key=lambda k: results[k]["weighted_f1"])
    save_results(results, y_test, all_predictions[best_name], sorted(set(labels)), names)
    print(f"\nBest weighted F1: {best_name} ({results[best_name]['weighted_f1']:.4f})")
    print("Saved metrics to results/metrics.json and confusion matrix to results/confusion_matrix.png")


def predict(text):
    from src.predict import predict_text
    print("Predicted issue area:", predict_text(text))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Supreme Court issue-area text classifier")
    parser.add_argument("command", choices=["train", "predict"])
    parser.add_argument("text", nargs="?", default="")
    args = parser.parse_args()
    if args.command == "train":
        train()
    else:
        if not args.text.strip():
            parser.error("For predict, provide the case text in quotes.")
        predict(args.text)
