# Streamlit App

Run locally:

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run app/streamlit_app.py
```

The app expects a trained model at `models/logistic_best.joblib`. If missing, run the training script first:

```powershell
python -m src.train.logistic --config config\train_logistic.yaml
```
