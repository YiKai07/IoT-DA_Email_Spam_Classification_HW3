## ADDED Requirements

### Requirement: Advanced Text Preprocessing
The system SHALL provide a configurable text preprocessing pipeline that supports the following steps: tokenization, stopword removal, and stemming or lemmatization.

#### Scenario: Stopword removal enabled
- **WHEN** preprocessing is run with `use_stopwords: true`
- **THEN** tokens matching the project's stopword list are removed from the token stream

#### Scenario: Stemming enabled
- **WHEN** preprocessing is run with `stemmer: porter` (or other configured stemmer)
- **THEN** tokens are normalized to their stems consistently across runs

#### Scenario: Toggleable pipeline
- **WHEN** preprocessing is executed with `use_stopwords: false` and `stemmer: none`
- **THEN** pipeline only performs tokenization and lowercasing (no removal or stemming)

### Requirement: TF-IDF Feature Extraction
The system SHALL provide TF-IDF vectorization as a configurable feature extraction option. Configuration options SHALL include n-gram range and maximum feature cardinality (max_features).

#### Scenario: TF-IDF enabled
- **WHEN** preprocessing is run with `vectorizer: tfidf` and `ngram_range: (1,2)` and `max_features: 10000`
- **THEN** the pipeline SHALL produce a TF-IDF feature matrix suitable for scikit-learn classifiers

#### Scenario: Compare vectorizers
- **WHEN** an experiment toggles between `vectorizer: count` and `vectorizer: tfidf`
- **THEN** the experiment notebook SHALL record and report metrics for both configurations

