# Conversation Log — 2025-11-12 (Detailed)

Repository: IoT-DA_Email_Spam_Classification_HW3

This file is a full, chronological record of the interactive session performed on 2025-11-12 while developing and debugging the Streamlit UI and associated ML pipelines for the Email/SMS Spam Classification homework project. It includes commands executed, files created/edited, errors encountered, root causes and fixes, verification steps, and follow-up notes.

---

## High-level summary

- Implemented Phase 4 Streamlit UI with full layout and components: dataset selection, model selection, test_size/seed/threshold controls, data overview (class distribution, token replacements table), top tokens by class, model performance (confusion matrix plotted + table, ROC and PR curves), threshold sweep table, and Live Inference area (text input + prediction with probability).
- Fixed numerous UI and plotting issues that caused blank screens and runtime errors. Key fixes included:
	- Moving token-counter initialization so it is available to all UI sections.
	- Adding `st.session_state` `cache_key` to trigger recomputation when `test_size`, `seed`, or `decision_threshold` change.
	- Installing and verifying plotting libraries (matplotlib, seaborn) and adding `HAS_PLOTTING` guards to degrade gracefully.
	- Fixing `selectbox` defaults to avoid empty model selection and improving model loading order.
	- Reducing confusion matrix figure size and adding a side-by-side DataFrame view of the confusion matrix.
- Trained model saved as `models/logistic_best.joblib` via the existing training pipeline; Streamlit app wired to load and use this model for live inference.
- Added a comprehensive `README.md` with installation and usage instructions.

---

## Chronological detailed trace

1) Initial problem reported by user:
	 - Streamlit app launched but the UI was blank. The user requested debugging and to build a full Phase 4 UI with specific left/right sections and controls.

2) Investigation and edits performed:
	 - Reviewed current codebase for `app/streamlit_app.py`, `src/preprocessing/text_pipeline.py`, `src/features/vectorizers.py`, and training pipeline. (Searches and file reads were performed during interactive debugging.)

3) Early fixes applied:
	 - Set Streamlit page config to wide layout via `st.set_page_config(layout='wide')` to ensure full-width rendering.
	 - Removed the empty-string default option in dataset/model selectboxes to avoid None/empty selections that caused downstream errors.
	 - Moved the token counters (ham/spam counters and token replacements) to be computed immediately after dataset load so they are available to both Data Overview and Top Tokens sections.

4) Plotting errors discovered:
	 - Errors like "name 'plt' is not defined" and "Plotting libraries not available" occurred in the Top Tokens by Class and Model Performance sections.
	 - Diagnosis: The project's virtual environment lacked `matplotlib` and `seaborn`. Some plotting calls were unguarded and variables referenced outside their scope.

5) Fixes for plotting and imports:
	 - Installed plotting libraries in the venv: matplotlib 3.10.7 and seaborn 0.13.2.
	 - Added a `HAS_PLOTTING` guard (try/except import) to the Streamlit app so that the UI shows a helpful message if plotting libs are missing instead of crashing.
	 - Ensured functions that create plots (`plot_confusion_cm`, `plot_roc_pr`) return plot objects and/or DataFrames to allow flexible rendering.

6) Confusion matrix display improvements:
	 - Reduced the confusion matrix figure size to improve layout (e.g., `figsize=(4,3)` for heatmap).
	 - Modified `plot_confusion_cm(y_true, y_pred)` to return a tuple `(fig, cm_df)` where `cm_df` is a pandas DataFrame representation of the confusion matrix. The Streamlit UI shows the plot and DataFrame side-by-side using `st.columns(2)`.

7) Dynamic recomputation implementation:
	 - Implemented a `cache_key = (len(df), test_size, int(seed), decision_threshold)` stored in `st.session_state`. When the key changes, the app re-splits the dataset and recomputes model predictions and plots. This made test_size/seed/threshold responsive without needing a manual refresh button.

8) Model loading and Live Inference issues:
	 - The Live Inference area sometimes showed "No model loaded" because model selection defaulted to empty or model loading happened after the inference section.
	 - Fixed by ensuring the model selection runs early and by auto-selecting the first model in the models directory if available.
	 - Live inference reuses the same preprocessing pipeline instance (if available) to ensure text processed in the same manner as training.

9) Training verification:
	 - The training pipeline (`src/train/logistic.py`) was executed previously to produce `models/logistic_best.joblib` using GridSearchCV and the configured vectorizer. That artifact is used by the Streamlit app.

10) README and documentation:
	 - Created a comprehensive `README.md` at the project root with setup steps (venv creation, pip install requirements), usage (how to run training and Streamlit app), project layout, and troubleshooting tips.

---

## Files created or modified (detailed)

- `app/streamlit_app.py`
	- Rewritten/extended to implement Phase 4 UI. Key functions and behaviors:
		- `load_dataset(path)` (@st.cache_data) — loads CSV into pandas DataFrame and allows column selection.
		- `token_counts_by_class(df, text_col, label_col)` — computes token counters per class and approximate replacements.
		- `plot_confusion_cm(y_true, y_pred)` — returns `(fig, cm_df)` for side-by-side display.
		- `plot_roc_pr(y_true, y_score)` — returns a matplotlib figure with ROC and PR plots.
		- Session state `cache_key` logic — recompute metrics/plots when `test_size`, `seed`, or `decision_threshold` change.
		- Live inference UI — text input, example buttons, model loading, preprocessing, prediction, and probability display.

- `src/preprocessing/text_pipeline.py`
	- Existing preprocessing pipeline (tokenize, lowercasing, optional stopword removal, optional Snowball stemming). Used by Streamlit app to ensure consistent preprocessing for training and inference.

- `src/features/vectorizers.py`
	- Vectorizer factory and helper functions to build CountVectorizer or TF-IDF vectorizer used in training.

- `src/train/logistic.py`
	- Training pipeline leveraging scikit-learn (`LogisticRegression`, `GridSearchCV`) and saving best model with `joblib.dump()` to `models/logistic_best.joblib`.

- `models/logistic_best.joblib`
	- Trained pipeline serialized to disk (vectorizer + classifier). This file is referenced by Streamlit for live inference.

- `README.md`
	- Created or updated with full setup and usage instructions.

- `conversations/conversation_log_2025-11-12.md`
	- This file (now replaced with this extended detailed log).

---

## Exact commands run during debugging (representative)

Note: these were run in the project venv and development terminal on Windows PowerShell. Some were executed by the assistant in the user's environment during the session; others are recorded edits.

- Activate venv and run Streamlit (verification):
```powershell
.\\.venv\\Scripts\\Activate.ps1
streamlit run app/streamlit_app.py
```

- Install plotting libs:
```powershell
.\\.venv\\Scripts\\Activate.ps1
pip install matplotlib seaborn
```

- Basic import test (verification):
```powershell
python -c "import matplotlib.pyplot as plt; import seaborn as sns; print('Plotting libs OK')"
```

- Git commands used to create conversation file locally (assistant created the file via edit operations):
```powershell
git add .\\conversations\\conversation_log_2025-11-12.md
git commit -m "Add detailed conversation log (2025-11-12)"
```

---

## Errors encountered and fixes (detailed)

- Error: Streamlit showed blank page / nothing rendered.
	- Cause: multiple small issues, including selectbox default empty option and variable scope problems.
	- Fixes: wide layout config, remove empty options, move variable initialization earlier, add safe guards.

- Error: "name 'plt' is not defined" in Top Tokens block.
	- Cause: plot code referenced `plt` but imports were missing or placed under try/except where scope differed.
	- Fix: centralized try/except import and ensure plotting functions import properly when HAS_PLOTTING True.

- Error: "ModuleNotFoundError: No module named 'matplotlib'" when running plots in venv.
	- Fix: pip install matplotlib seaborn in the venv.

---

## Verification performed

- Streamlit started successfully and UI rendered at `http://localhost:8501` during testing.
- Plots render correctly when plotting libs available.
- Confusion matrix renders as a smaller heatmap with side-by-side DataFrame.
- Model predictions return expected labels and probabilities for sample inputs.

---

## Reproducibility and how to push this log to GitHub

I cannot push changes to your GitHub repository directly due to lack of credentials. Below are the exact PowerShell commands to run from the project root to push this file.

1) Set remote (if not set):
```powershell
git remote add origin https://github.com/YiKai07/IoT-DA_Email_Spam_Classification_HW3.git
```

2) Stage and commit the conversation log:
```powershell
git add .\\conversations\\conversation_log_2025-11-12.md
git commit -m "Add detailed conversation log (2025-11-12): Streamlit UI work and fixes"
```

3) Push to your remote main branch (replace `main` with `master` if needed):
```powershell
git push -u origin main
```

If authentication issues occur, create a GitHub Personal Access Token (PAT) with `repo` scope and use it when prompted for a password. Alternatively, use `gh auth login` (GitHub CLI) or Git Credential Manager.

---

## Follow-ups and optional improvements

- Add feature-importance visualization (coefficients from LogisticRegression) to the Streamlit UI.
- Expose vectorizer options (n-grams, max_features) as sidebar controls.
- Add a "retrain" button that triggers retraining with the selected config and stores the model back into `models/`.
- Add unit tests for the Streamlit helper functions (plotting outputs as DataFrames/figures).

---

Timestamp: 2025-11-12

Recorded by: assistant

