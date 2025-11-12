import sys
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.preprocessing.text_pipeline import TextPipeline
from src.features.vectorizers import build_vectorizer, fit_transform_vectorizer


def test_stopword_removal_and_stemming():
    texts = ["This is a simple test", "Testing stemming running runs"]
    tp = TextPipeline(use_stopwords=True, stemmer='snowball', language='english')
    out = tp.process(texts)
    assert isinstance(out, list)
    assert len(out) == 2
    # ensure common stopword 'is' removed
    assert 'is' not in out[0].split()


def test_tfidf_vectorizer_shapes():
    texts = ["spam message free money", "hello friend this is ham"]
    vec = build_vectorizer({'vectorizer': 'tfidf', 'ngram_range': (1,1), 'max_features': 10})
    X = fit_transform_vectorizer(vec, texts)
    assert X.shape[0] == 2
    assert X.shape[1] <= 10
