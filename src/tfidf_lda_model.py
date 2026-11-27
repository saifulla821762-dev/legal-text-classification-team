"""TF-IDF -> LDA topic vectors -> Logistic Regression."""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def build_models(num_topics=100, random_state=42):
    base = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=30000, stop_words="english", ngram_range=(1, 2))),
        ("lda", LatentDirichletAllocation(n_components=num_topics, max_iter=10, learning_method="batch", random_state=random_state)),
        ("classifier", LogisticRegression(max_iter=2000, random_state=random_state)),
    ])
    weighted = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=30000, stop_words="english", ngram_range=(1, 2))),
        ("lda", LatentDirichletAllocation(n_components=num_topics, max_iter=10, learning_method="batch", random_state=random_state)),
        ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state)),
    ])
    return {"TF-IDF + LDA + Logistic Regression": base,
            "Class-weighted TF-IDF + LDA + Logistic Regression": weighted}
