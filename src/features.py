import joblib
from sklearn.feature_extraction.text import TfidfVectorizer


def build_tfidf_vectorizer(max_features: int = 5000) -> TfidfVectorizer:
    return TfidfVectorizer(max_features=max_features, ngram_range=(1, 1))


def fit_transform(vectorizer: TfidfVectorizer, texts: list):
    return vectorizer.fit_transform(texts)


def transform(vectorizer: TfidfVectorizer, texts: list):
    return vectorizer.transform(texts)


def save_vectorizer(vectorizer: TfidfVectorizer, path: str):
    joblib.dump(vectorizer, path)


def load_vectorizer(path: str) -> TfidfVectorizer:
    return joblib.load(path)
