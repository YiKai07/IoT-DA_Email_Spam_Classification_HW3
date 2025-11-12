---
title: Add Logistic Regression training pipeline with hyperparameter tuning
change-id: add-logistic-training
---

# Change: Add Logistic Regression training pipeline with hyperparameter tuning

## Why
A clear, reproducible core training pipeline is required to establish a baseline classifier for spam detection. Logistic Regression is a robust, interpretable baseline that works well on sparse text features (Count/TF-IDF). Adding hyperparameter tuning (e.g., regularization C, penalty, solver, class_weight) via scikit-learn's GridSearchCV or RandomizedSearchCV will improve model performance and allow reproducible experiments.

## What Changes
- ADDED: Training module `src/train/logistic.py` that encapsulates data loading, preprocessing pipeline integration, vectorization selection (count or tfidf), model training, hyperparameter search, cross-validation, and model serialization.
- ADDED: Task to save best model and training artifacts (metrics, confusion matrix, classification report) under `models/` and `reports/`.
- ADDED: Unit tests and an end-to-end smoke test verifying training runs on a small sample.
- ADDED: Example CLI entrypoint `python -m src.train.logistic --config config/train_logistic.yaml`.

## Impact
- Affected specs: `specs/training/spec.md`
- Affected code: new `src/train/` module, small changes to `src/preprocessing/` and `src/features/` if needed.

## Rollout
- Keep hyperparameter search bounded (reasonable grid or randomized search) and provide sensible defaults to allow CI smoke tests to run quickly.
