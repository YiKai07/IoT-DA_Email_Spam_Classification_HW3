import os
import sys
from pathlib import Path
import yaml
# ensure project root is on sys.path for tests
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.train.logistic import train


def make_small_csv(tmp_path):
    p = tmp_path / 'sample.csv'
    p.write_text('spam,Free money now\nham,Hello friend\nspam,Win prize\n')
    return str(p)


def test_train_smoke(tmp_path):
    csv = make_small_csv(tmp_path)
    cfg = {
        'data': {'csv': csv},
        'features': {'vectorizer': 'count', 'ngram_range': [1,1]},
        'training': {'cv': 2, 'param_grid': {'clf__C': [1.0]}},
        'output': {'dir': str(tmp_path / 'models'), 'model_name': 'm.joblib'}
    }
    cfg_path = tmp_path / 'cfg.yaml'
    cfg_path.write_text(yaml.dump(cfg))
    model_path = train(str(cfg_path))
    assert Path(model_path).exists()
