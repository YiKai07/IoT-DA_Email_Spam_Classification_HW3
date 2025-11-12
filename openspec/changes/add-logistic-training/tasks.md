## 1. Implementation
- [ ] 1.1 Create `src/train/logistic.py` implementing the training pipeline and CLI interface.
- [ ] 1.2 Integrate preprocessing pipeline and vectorizer options (`vectorizer: count|tfidf`) into training config.
- [ ] 1.3 Implement hyperparameter tuning (GridSearchCV or RandomizedSearchCV) with cross-validation and `random_state` control.
- [ ] 1.4 Serialize best model to `models/logistic_best.joblib` and save metrics to `reports/training_logistic.md`.
- [ ] 1.5 Add end-to-end smoke test `tests/test_train_logistic.py` using a small sample dataset fixture.

## 2. Validation
- [ ] 2.1 Run training on small sample and ensure model file and report are produced.
- [ ] 2.2 Validate that hyperparameter search respects time budget (configurable `n_iter` or grid size).

## 3. Documentation
- [ ] 3.1 Add `config/train_logistic.yaml` example and document CLI usage in README.
