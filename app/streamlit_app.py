"""Streamlit interface for Phase 4 visualizations and live inference.

Left sidebar provides Inputs. Main area shows data overview, model
performance and live inference controls.
"""

import sys
from pathlib import Path
from typing import List, Tuple

import streamlit as st

st.set_page_config(layout='wide')
import joblib
import pandas as pd
import numpy as np
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA_DIR = Path('datasets')
MODELS_DIR = Path('models')

# plotting libs
try:
    import matplotlib.pyplot as plt  # type: ignore
    import seaborn as sns  # type: ignore
    HAS_PLOTTING = True
except Exception as e:
    HAS_PLOTTING = False
    # Debug: print the error
    import sys
    print(f"Plotting import failed: {e}", file=sys.stderr)

try:
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, precision_recall_fscore_support
    from src.preprocessing.text_pipeline import TextPipeline
except Exception:
    train_test_split = None
    confusion_matrix = None
    roc_curve = None
    auc = None
    precision_recall_curve = None
    precision_recall_fscore_support = None
    TextPipeline = None


def list_csvs() -> List[str]:
    if not DATA_DIR.exists():
        return []
    return [p.name for p in DATA_DIR.glob('*.csv')]


@st.cache_data
def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, header=None)
    # try to coerce to two-column if extra headers
    if df.shape[1] >= 2:
        df = df.iloc[:, :2]
    df = df.rename(columns={0: 'label', 1: 'text'})
    df['label'] = df['label'].astype(str)
    df['text'] = df['text'].astype(str)
    return df


def load_model(path: Path):
    if not path.exists():
        return None
    return joblib.load(path)


def class_distribution(df: pd.DataFrame) -> pd.Series:
    return df['label'].value_counts()


def token_counts_by_class(df: pd.DataFrame, tp=None):
    ham_ctr = Counter()
    spam_ctr = Counter()
    replacements = []
    for _, row in df.iterrows():
        text = row['text']
        label = row['label'].lower()
        cleaned = text
        if tp is not None:
            cleaned = tp.process([text])[0]
        # record example pair
        replacements.append((text[:120], cleaned[:120]))
        toks = cleaned.split()
        if label == 'spam':
            spam_ctr.update(toks)
        else:
            ham_ctr.update(toks)
    return ham_ctr, spam_ctr, replacements


def compute_threshold_metrics(y_true, y_scores, thresholds: np.ndarray):
    rows = []
    for t in thresholds:
        y_pred = (np.array(y_scores) >= t).astype(int)
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
        rows.append({'threshold': float(t), 'precision': float(prec), 'recall': float(rec), 'f1': float(f1)})
    return pd.DataFrame(rows)


def plot_confusion_cm(y_true, y_pred):
    if not HAS_PLOTTING:
        return None
    fig, ax = plt.subplots(figsize=(4, 3))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    plt.tight_layout()
    return fig, cm


def plot_roc_pr(y_true, y_scores):
    if not HAS_PLOTTING:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    axes[0].plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
    axes[0].plot([0, 1], [0, 1], 'k--')
    axes[0].set_title('ROC')
    axes[0].set_xlabel('FPR')
    axes[0].set_ylabel('TPR')
    axes[0].legend(fontsize=8)

    prec, rec, _ = precision_recall_curve(y_true, y_scores)
    axes[1].plot(rec, prec)
    axes[1].set_title('Precision-Recall')
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    plt.tight_layout()
    return fig


def main():
    st.title('Spam/Ham Classifier — Phase 4 Visualizations')

    # Sidebar inputs
    st.sidebar.header('Inputs')
    csv_list = list_csvs()
    dataset_choice = st.sidebar.selectbox('Dataset CSV', options=csv_list if csv_list else ['(no CSV found)'])
    label_col = st.sidebar.selectbox('Label column', options=['label'])
    text_col = st.sidebar.selectbox('Text column', options=['text'])
    model_dir = st.sidebar.text_input('Models dir', value=str(MODELS_DIR))
    test_size = st.sidebar.slider('Test size', 0.1, 0.5, 0.25, step=0.05)
    seed = st.sidebar.number_input('Seed', value=42, step=1)
    decision_threshold = st.sidebar.slider('Decision threshold', 0.0, 1.0, 0.5, step=0.01)

    # Load dataset if selected
    df = None
    if dataset_choice and dataset_choice != '(no CSV found)':
        path = DATA_DIR / dataset_choice
        try:
            df = load_dataset(path)
        except Exception as e:
            st.sidebar.error(f'Failed to load dataset: {e}')

    # Initialize counters at top level so they're accessible everywhere
    ham_ctr = Counter()
    spam_ctr = Counter()
    replacements = []
    tp = None
    
    # Precompute token counts if dataset is loaded
    if df is not None:
        if TextPipeline is not None:
            tp = TextPipeline(use_stopwords=True, stemmer='snowball')
        ham_ctr, spam_ctr, replacements = token_counts_by_class(df, tp)

    # Main: Data Overview
    st.header('Data Overview')
    if df is None:
        st.info('No dataset selected.')
    else:
        cols = st.columns([1, 2])
        with cols[0]:
            st.subheader('Class distribution')
            dist = class_distribution(df)
            st.bar_chart(dist)

        with cols[1]:
            st.subheader('Token replacements in cleaned text (approx)')
            # show a few replacement examples as table
            if replacements:
                df_replacements = pd.DataFrame(replacements[:5], columns=['Original', 'Cleaned'])
                st.dataframe(df_replacements, width='stretch')

    # Top Tokens by Class
    st.subheader('Top Tokens by Class')
    top_n = st.slider('Top N tokens', 5, 50, 15)
    if df is not None:
        ham_top = ham_ctr.most_common(top_n)
        spam_top = spam_ctr.most_common(top_n)
        if HAS_PLOTTING:
            col1, col2 = st.columns(2)
            with col1:
                st.write('Ham top tokens')
                if ham_top:
                    toks, vals = zip(*ham_top)
                    fig, ax = plt.subplots()
                    ax.barh(list(toks)[::-1], list(vals)[::-1])
                    st.pyplot(fig)
            with col2:
                st.write('Spam top tokens')
                if spam_top:
                    toks, vals = zip(*spam_top)
                    fig, ax = plt.subplots()
                    ax.barh(list(toks)[::-1], list(vals)[::-1], color='orange')
                    st.pyplot(fig)
        else:
            st.info('Plotting libraries not available; tokens cannot be visualized.')
    else:
        st.info('Select a dataset to see top tokens.')

    # Model performance
    st.header('Model Performance (Test)')
    # choose model file from models dir if exists
    try:
        model_files = [p.name for p in Path(model_dir).glob('*.joblib')] if Path(model_dir).exists() else []
    except Exception:
        model_files = []
    
    model = None
    if model_files:
        # Default to first model if available
        default_model = model_files[0] if model_files else ''
        model_choice = st.selectbox('Select model', options=model_files, index=0)
        if model_choice:
            model_path = Path(model_dir) / model_choice
            model = load_model(model_path)
    else:
        st.info('No models found in directory.')

    if df is None or model is None:
        st.info('Model or dataset missing — cannot compute test performance.')
    else:
        # Use session state to detect changes in parameters
        cache_key = (len(df), test_size, int(seed), decision_threshold)
        if 'last_cache_key' not in st.session_state or st.session_state['last_cache_key'] != cache_key:
            st.session_state['last_cache_key'] = cache_key
            st.session_state['recompute'] = True
        
        X = df['text'].tolist()
        y = (df['label'].str.lower() == 'spam').astype(int).values
        if train_test_split is None:
            st.error('sklearn not available in environment.')
        else:
            # split and compute metrics
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=int(seed), stratify=y if len(np.unique(y))>1 else None)
            # preprocess if needed
            if TextPipeline is not None:
                tp_local = TextPipeline(use_stopwords=True, stemmer='snowball')
                X_test_proc = tp_local.process(X_test)
            else:
                X_test_proc = X_test

            try:
                y_score = [p[1] for p in model.predict_proba(X_test_proc)]
            except Exception:
                # fallback to predict
                y_score = model.predict(X_test_proc)
            y_pred = (np.array(y_score) >= decision_threshold).astype(int)

            # confusion
            if HAS_PLOTTING:
                result_cm = plot_confusion_cm(y_test, y_pred)
                if result_cm is not None:
                    fig_cm, cm = result_cm
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        st.pyplot(fig_cm)
                    with col2:
                        # Display confusion matrix as table
                        cm_df = pd.DataFrame(cm, 
                                           index=['Actual Ham', 'Actual Spam'], 
                                           columns=['Predicted Ham', 'Predicted Spam'])
                        st.write('**Confusion Matrix (Table)**')
                        st.dataframe(cm_df, width='stretch')
                else:
                    st.info('Confusion matrix plot unavailable (plotting libs disabled).')

                fig_rocpr = plot_roc_pr(y_test, y_score)
                if fig_rocpr is not None:
                    st.pyplot(fig_rocpr)
                else:
                    st.info('ROC/Precision-Recall plots unavailable (plotting libs disabled).')
            else:
                st.info('Plotting libraries not available; performance plots cannot be visualized.')

            # threshold sweep
            st.header('Threshold sweep (precision/recall/f1)')
            thresholds = np.linspace(0, 1, 21)
            df_thresh = compute_threshold_metrics(y_test, y_score, thresholds)
            st.dataframe(df_thresh, width='stretch')

    # Live Inference
    st.header('Live Inference')
    
    # Ensure model is loaded for live inference
    if model is None:
        st.info('No model selected for live inference. Please select a model in "Model Performance (Test)" section above.')
    else:
        colA, colB = st.columns(2)
        sample_spam = 'Congratulations! You have won a free ticket. Click to claim.'
        sample_ham = 'Hey, are we still meeting tomorrow at 10am?'
        with colA:
            if st.button('Use spam example'):
                st.session_state['live_text'] = sample_spam
        with colB:
            if st.button('Use ham example'):
                st.session_state['live_text'] = sample_ham

        live_text = st.text_area('Enter a message to classify', value=st.session_state.get('live_text', ''), height=150)
        if st.button('Predict (live)'):
            if not live_text.strip():
                st.error('Enter a message')
            else:
                processed = live_text
                if TextPipeline is not None:
                    tp_live = TextPipeline(use_stopwords=True, stemmer='snowball')
                    processed = tp_live.process([live_text])[0]
                    st.write('Preprocessed input:', processed)
                try:
                    pred = model.predict([processed])[0]
                    prob = None
                    try:
                        prob = model.predict_proba([processed])[0][1]
                    except Exception:
                        prob = None
                    label = 'Spam' if int(pred) == 1 else 'Not Spam'
                    st.success(f'Prediction: {label}')
                    if prob is not None:
                        st.info(f'Probability (Spam): {prob:.3f}')
                except Exception as e:
                    st.error(f'Live prediction failed: {e}')


if __name__ == '__main__':
    main()
