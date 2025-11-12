import argparse
import yaml
from pathlib import Path
import joblib
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import classification_report

from src.data.load_data import load_sms_spam
from src.preprocessing.text_pipeline import TextPipeline


def build_vectorizer(cfg):
    vec = cfg.get('vectorizer', 'tfidf')
    ngram = tuple(cfg.get('ngram_range', (1,1)))
    maxf = cfg.get('max_features', None)
    if vec == 'count':
        return CountVectorizer(ngram_range=ngram, max_features=maxf)
    return TfidfVectorizer(ngram_range=ngram, max_features=maxf)


def train(config_path: str):
    cfg = yaml.safe_load(Path(config_path).read_text())
    data_path = cfg['data']['csv']
    df = load_sms_spam(data_path)
    X = df['text'].astype(str).values
    y = (df['label'].str.lower() == 'spam').astype(int).values
    # optional preprocessing pipeline
    preproc_cfg = cfg.get('preprocessing', {})
    if preproc_cfg.get('enabled', False):
        tp = TextPipeline(use_stopwords=preproc_cfg.get('use_stopwords', True),
                          stemmer=preproc_cfg.get('stemmer', 'snowball'),
                          language=preproc_cfg.get('language', 'english'))
        X = tp.process(X)

    vect = build_vectorizer(cfg.get('features', {}))
    pipe = Pipeline([
        ('vect', vect),
        ('clf', LogisticRegression(max_iter=1000))
    ])

    param_grid = cfg.get('training', {}).get('param_grid', {'clf__C': [1.0]})
    cv = cfg.get('training', {}).get('cv', 3)
    gs = GridSearchCV(pipe, param_grid=param_grid, cv=cv, n_jobs=1)
    gs.fit(X, y)

    out_dir = Path(cfg.get('output', {}).get('dir', 'models'))
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / cfg.get('output', {}).get('model_name', 'logistic_best.joblib')
    joblib.dump(gs.best_estimator_, model_path)

    # save basic report
    rep = classification_report(y, gs.predict(X), output_dict=False)
    report_path = out_dir / 'training_report.txt'
    report_path.write_text(str(rep))
    return str(model_path)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', '-c', required=True)
    args = p.parse_args()
    model_file = train(args.config)
    print('Saved model to', model_file)
