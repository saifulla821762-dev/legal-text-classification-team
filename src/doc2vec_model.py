"""Leakage-safe Doc2Vec model: fit embeddings on training documents only."""
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.linear_model import LogisticRegression
import re


def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def fit_doc2vec(train_texts, test_texts, train_labels, vector_size=150, epochs=30, seed=42):
    tagged = [TaggedDocument(words=tokenize(t), tags=[str(i)]) for i, t in enumerate(train_texts)]
    model = Doc2Vec(vector_size=vector_size, window=8, min_count=2, workers=1,
                    epochs=epochs, seed=seed, dm=1)
    model.build_vocab(tagged)
    model.train(tagged, total_examples=model.corpus_count, epochs=model.epochs)
    X_train = [model.dv[str(i)] for i in range(len(train_texts))]
    X_test = [model.infer_vector(tokenize(t), epochs=20) for t in test_texts]
    clf = LogisticRegression(max_iter=2000, random_state=seed)
    clf.fit(X_train, train_labels)
    return model, clf, X_test
