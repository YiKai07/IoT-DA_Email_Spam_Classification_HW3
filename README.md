# Email/SMS Spam Classification — Phase 4

A comprehensive spam classification pipeline with interactive Streamlit UI for data exploration, model training, and real-time inference.

## Project Overview

This project implements a multi-phase spam classification system that:
1. **Loads and preprocesses** SMS/email datasets
2. **Trains models** using Logistic Regression with hyperparameter tuning
3. **Visualizes performance** with confusion matrices, ROC/Precision-Recall curves
4. **Enables interactive inference** via a Streamlit web interface

## Directory Structure

```
HW3_email spam classification/
├── app/
│   └── streamlit_app.py           # Main Streamlit UI application
├── src/
│   ├── data/
│   │   └── load_data.py           # Data loading utilities
│   ├── preprocessing/
│   │   └── text_pipeline.py       # Text preprocessing (stopwords, stemming)
│   ├── features/
│   │   └── vectorizers.py         # TF-IDF and Count vectorizers
│   └── train/
│       └── logistic.py            # Model training pipeline
├── datasets/
│   └── sms_spam_no_header.csv     # Training dataset (label, text)
├── models/
│   ├── logistic_best.joblib       # Trained Logistic Regression model
│   └── training_report.txt        # Classification report from training
├── tests/
│   ├── test_preprocessing.py      # Unit tests for text preprocessing
│   └── test_train_logistic.py     # Unit tests for model training
├── config/
│   └── train_config.yaml          # Training configuration
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## Setup Instructions

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- Virtual environment (recommended)

### 1. Clone/Download the Project

```bash
cd path/to/HW3_email\ spam\ classification
```

### 2. Create Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- `pandas` — Data manipulation
- `scikit-learn` — Machine learning models and metrics
- `joblib` — Model serialization
- `nltk` — NLP utilities (stopwords, stemming)
- `pytest` — Unit testing
- `pyyaml` — Configuration files
- `streamlit` — Web UI framework
- `matplotlib`, `seaborn` — Visualization

### 4. Download NLTK Data (Optional but Recommended)

```bash
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
```

This enables stopword removal in the preprocessing pipeline.

## Usage

### Option 1: Interactive Streamlit UI (Recommended)

Start the web interface:

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run app/streamlit_app.py
```

Open your browser to: **http://localhost:8501**

#### Features:

**Left Sidebar (Inputs):**
- `Dataset CSV` — Select CSV file from `datasets/` folder
- `Label column` — Column name for spam/ham labels
- `Text column` — Column name for message text
- `Models dir` — Directory path containing trained `.joblib` models
- `Test size` — Fraction of data for testing (0.1–0.5)
- `Seed` — Random seed for reproducibility
- `Decision threshold` — Adjust spam probability threshold (0.0–1.0)

**Main Area (Visualizations):**

1. **Data Overview**
   - Class distribution bar chart
   - Token replacements table (original vs. cleaned text)

2. **Top Tokens by Class**
   - Interactive slider to select top N tokens
   - Horizontal bar charts for Ham and Spam most frequent words

3. **Model Performance (Test)**
   - Model selector dropdown
   - Confusion matrix (heatmap + table)
   - ROC curve (with AUC score)
   - Precision-Recall curve
   - Threshold sweep table (precision, recall, F1 at different thresholds)

4. **Live Inference**
   - Quick buttons: "Use spam example" / "Use ham example"
   - Text input for custom messages
   - Predict button for real-time classification
   - Display: prediction label + spam probability

### Option 2: Command-Line Training

Train a new model with the configuration file:

```powershell
.\.venv\Scripts\Activate.ps1
python -m src.train.logistic --config config/train_config.yaml
```

Output:
- Model saved to `models/logistic_best.joblib`
- Report saved to `models/training_report.txt`

### Option 3: Run Tests

```powershell
.\.venv\Scripts\Activate.ps1
pytest tests/ -v
```

Tests included:
- `test_preprocessing.py` — Stopword removal, stemming, TF-IDF vectorization
- `test_train_logistic.py` — Model training pipeline

## Configuration

Edit `config/train_config.yaml` to customize training:

```yaml
data:
  csv: datasets/sms_spam_no_header.csv

preprocessing:
  enabled: true
  use_stopwords: true
  stemmer: snowball
  language: english

features:
  vectorizer: tfidf
  ngram_range: [1, 1]
  max_features: 5000

training:
  cv: 5
  param_grid:
    clf__C: [0.1, 1.0, 10.0]

output:
  dir: models
  model_name: logistic_best.joblib
```

## Dataset Format

The CSV file should have **no header** and **two columns**:
1. Column 0: Label (`spam` or `ham`)
2. Column 1: Message text

**Example:**
```
spam,Free money! Click here now!!!
ham,Hey, are we still on for tonight?
spam,Congratulations! You won a prize.
ham,Meeting at 3pm tomorrow.
```

## Key Features

### Text Preprocessing Pipeline
- **Tokenization**: Extract words (2+ characters, lowercase)
- **Stopword Removal**: Remove common English words (optional)
- **Stemming**: Reduce words to root form using Snowball stemmer (optional)

### Model Training
- **Algorithm**: Logistic Regression with L2 regularization
- **Vectorization**: TF-IDF with configurable n-grams and max features
- **Hyperparameter Tuning**: GridSearchCV with cross-validation
- **Evaluation**: Confusion matrix, ROC-AUC, Precision-Recall

### Interactive UI
- **Wide layout** for better visualization
- **Dynamic retraining**: Changes to test size, seed, or threshold automatically recalculate metrics
- **Live inference**: Real-time spam classification with probability display
- **Token analysis**: Visualize most important words per class

## Troubleshooting

### Issue: "Streamlit command not found"
**Solution**: Ensure virtual environment is activated (`Activate.ps1` on Windows)

### Issue: "No module named 'src'"
**Solution**: Run Streamlit from the project root directory:
```powershell
cd path/to/HW3_email\ spam\ classification
streamlit run app/streamlit_app.py
```

### Issue: "matplotlib/seaborn import failed"
**Solution**: Reinstall plotting libraries:
```powershell
pip install --upgrade matplotlib seaborn
```

### Issue: "No models found in directory"
**Solution**: Train a model first using the training script:
```powershell
python -m src.train.logistic --config config/train_config.yaml
```

## Example Workflow

1. **Start Streamlit UI**
   ```powershell
   streamlit run app/streamlit_app.py
   ```

2. **Select Dataset**
   - Choose `sms_spam_no_header.csv` from the Dataset CSV dropdown

3. **Explore Data**
   - View class distribution and token examples

4. **Select Model**
   - Choose a trained model from the dropdown

5. **Adjust Parameters**
   - Modify test size, seed, or decision threshold
   - Visualizations update automatically

6. **Live Inference**
   - Enter a custom message or use example buttons
   - Click "Predict (live)" to classify

## Performance

On the SMS Spam dataset (~5,500 messages):
- **Training time**: ~2–5 seconds (with cross-validation)
- **Inference time**: <100ms per message
- **Typical accuracy**: 95%+ (depends on preprocessing and threshold)

## Dependencies & Versions

See `requirements.txt` for exact versions. Key dependencies:
- `pandas>=1.0`
- `scikit-learn>=1.0`
- `matplotlib` (for plotting)
- `seaborn` (for styled plots)
- `streamlit` (for UI)
- `nltk` (for NLP preprocessing)

## Contributing

To add new features or fix issues:
1. Create a new branch
2. Make changes and test (`pytest tests/`)
3. Ensure no lint errors
4. Submit a pull request

## License

[Specify your license, e.g., MIT, GPL, etc.]

## Contact

For questions or issues, please open a GitHub issue or contact the maintainers.

---

**Last Updated**: November 12, 2025  
**Project Type**: Educational (Homework Assignment)  
**Status**: Phase 4 (Interactive Visualizations & Inference)
