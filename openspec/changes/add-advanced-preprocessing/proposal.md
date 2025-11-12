---
title: Add advanced preprocessing (stopword removal + stemming)
change-id: add-advanced-preprocessing
---

# Change: Add advanced preprocessing (stopword removal + stemming)

## Why
Current baseline preprocessing may be limited to simple tokenization and lowercasing. Adding stopword removal and stemming/lemmatization will reduce noise, decrease vocabulary size, and likely improve classifier generalization on SMS/email spam detection tasks. This is a small, low-risk enhancement that benefits feature extraction and makes downstream models more stable.

## What Changes
- ADDED: Preprocessing module that supports configurable stopword removal and stemming (or lemmatization) as pipeline steps.
- ADDED: TF-IDF vectorization support as a configurable feature-extraction step (with options for n-grams and max_features).
- ADDED: Unit tests that validate stopword removal, stemming, and TF-IDF behavior on representative samples.
- ADDED: Documentation and an example notebook demonstrating effect on model performance (baseline vs TF-IDF + advanced preprocessing).
- UPDATED: Data processing tasks to reference dataset `datasets/sms_spam_no_header.csv` in repo (copy/link if needed).

## Impact
- Affected specs: `specs/preprocessing/spec.md`
- Affected code: new module under `src/preprocessing/` (e.g., `tokenize.py`, `normalize.py`), tests in `tests/` and example notebook in `notebooks/`.

## Rollout
- Implement feature behind a config flag (e.g., `use_stopwords: true`, `stemmer: porter|snowball|none`, `vectorizer: tfidf|count|none`) so experiments can toggle behavior.

