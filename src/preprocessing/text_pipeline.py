import re
from typing import List, Iterable
try:
    from nltk.corpus import stopwords
    from nltk.stem import SnowballStemmer
except Exception:
    stopwords = None
    SnowballStemmer = None

# Minimal fallback stopword list when NLTK data is not available
DEFAULT_STOPWORDS = {
    'a', 'an', 'the', 'and', 'or', 'but', 'if', 'while', 'is', 'are', 'was', 'were',
    'this', 'that', 'these', 'those', 'in', 'on', 'for', 'to', 'of', 'with', 'as',
    'I', 'you', 'he', 'she', 'it', 'we', 'they', 'be', 'by', 'at', 'from'
}


def simple_tokenize(text: str) -> List[str]:
    tokens = re.findall(r"\b[a-zA-Z]{2,}\b", text.lower())
    return tokens


class TextPipeline:
    def __init__(self, use_stopwords: bool = True, stemmer: str = 'snowball', language: str = 'english'):
        self.use_stopwords = use_stopwords
        self.stemmer_name = stemmer
        self.language = language
        if use_stopwords:
            if stopwords:
                try:
                    self.stopwords = set(stopwords.words(language))
                except Exception:
                    self.stopwords = set(w.lower() for w in DEFAULT_STOPWORDS)
            else:
                self.stopwords = set(w.lower() for w in DEFAULT_STOPWORDS)
        else:
            self.stopwords = set()
        if SnowballStemmer and stemmer in ('snowball', 'porter'):
            self.stemmer = SnowballStemmer(language)
        else:
            self.stemmer = None

    def process(self, texts: Iterable[str]) -> List[str]:
        out = []
        for t in texts:
            toks = simple_tokenize(t)
            if self.use_stopwords:
                toks = [tok for tok in toks if tok not in self.stopwords]
            if self.stemmer:
                toks = [self.stemmer.stem(tok) for tok in toks]
            out.append(' '.join(toks))
        return out
