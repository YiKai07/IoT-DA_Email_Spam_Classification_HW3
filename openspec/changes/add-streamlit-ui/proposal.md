---
title: Add Streamlit UI for interactive model testing and results display
change-id: add-streamlit-ui
---

# Change: Add Streamlit UI for interactive model testing and results display

## Why
Providing a lightweight Streamlit frontend enables interactive testing of the spam classifier by non-technical users (e.g., TAs or graders), helps demonstrating model behavior, and simplifies qualitative inspection of predictions. Streamlit is low-effort to integrate and supports quick deployment or local execution.

## What Changes
- ADDED: `app/streamlit_app.py` frontend that loads the trained model and allows users to input text, run preprocessing, and view predicted label, probability, and relevant feature highlights (e.g., top TF-IDF tokens or important features).
- ADDED: Documentation detailing how to run Streamlit locally and how the app loads models and config.
- ADDED: A basic end-to-end test that the Streamlit app can start and serve a health endpoint (smoke test).

## Impact
- Affected specs: `specs/ui/streamlit/spec.md`
- Affected code: new `app/streamlit_app.py`, lightweight templates or helper functions in `src/ui/`.

## Rollout
- Implement as optional tooling (not required for core ML pipeline). The app should load model from `models/` and respect preprocessing config flags.
