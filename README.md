# Legal Text Classification – Supreme Court Decisions

A team machine-learning mini project that predicts the broad issue-area category of a U.S. Supreme Court decision from its text.

## Dataset
Textacy's Supreme Court corpus: approximately 8,400 decisions (1946–2016). We use the `issue_area` label and exclude missing labels. The project compares:
1. TF-IDF features + LDA topic features + Logistic Regression baseline
2. Class-weighted Logistic Regression on the same topic features
3. Doc2Vec document embeddings + Logistic Regression

The dataset is downloaded automatically on first run. Dataset records are split into train/test **before model training**; Doc2Vec is trained on training text only to avoid test-data leakage.

## Setup
Python 3.10+ recommended.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Run
Train and compare all models:
```bash
python main.py train
``

Predict a category for a text:
```bash
python main.py predict "Paste a Supreme Court decision excerpt here"
``

Results are saved under `results/` (metrics JSON and confusion-matrix PNG). Models are saved under `models/`.

## Team work
See [TEAM.md](TEAM.md) for task division.

## Important
This is an educational classification prototype, not legal advice or a legal decision-making tool. Predictions may be incorrect, especially for rare categories.
