## 1. Implementation
- [ ] 1.1 Create `src/preprocessing/` with configurable pipeline: tokenization, stopword removal, stemming/lemmatization.
- [ ] 1.2 Add helper to load dataset at `datasets/sms_spam_no_header.csv` (or point to repo path).
- [ ] 1.3 Add unit tests for preprocessing functions in `tests/test_preprocessing.py`.
- [ ] 1.4 Add example notebook `notebooks/experiments/advanced_preprocessing.ipynb` showing baseline vs with-stopwords+stemming.

## TF-IDF Tasks
- [ ] 1.5 Implement TF-IDF vectorization option in `src/features/vectorizers.py` with configurable n-grams and max_features.
- [ ] 1.6 Add experiments in `notebooks/experiments/advanced_preprocessing.ipynb` comparing CountVectorizer vs TfidfVectorizer and report results.


## 2. Validation
- [ ] 2.1 Run smoke end-to-end experiment ensuring training pipeline still runs and outputs model + metrics.
- [ ] 2.2 Review metrics delta and document findings in `reports/advanced_preprocessing.md`.

## 3. Documentation
- [ ] 3.1 Update `README.md` with new preprocessing options and how to run experiments.
