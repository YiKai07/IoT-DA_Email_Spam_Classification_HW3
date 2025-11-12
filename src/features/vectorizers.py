from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from typing import Tuple


def build_vectorizer(config: dict):
    vec = config.get('vectorizer', 'tfidf')
    ngram = tuple(config.get('ngram_range', (1, 1)))
    maxf = config.get('max_features', None)
    if vec == 'count':
        return CountVectorizer(ngram_range=ngram, max_features=maxf)
    return TfidfVectorizer(ngram_range=ngram, max_features=maxf)


def fit_transform_vectorizer(vec, texts):
    X = vec.fit_transform(texts)
    return X
