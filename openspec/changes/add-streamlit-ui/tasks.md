## 1. Implementation
- [ ] 1.1 Create `app/streamlit_app.py` that loads `models/latest_model.joblib` (or configurable path) and exposes a minimal interactive UI: text input, predict button, result display.
- [ ] 1.2 Add helper functions in `src/ui/` to format model outputs and compute top contributing tokens (if model supports feature importance).
- [ ] 1.3 Add a smoke test that the Streamlit app process starts and responds to `/health` or the main page.

## 2. Documentation
- [ ] 2.1 Add `docs/streamlit.md` describing how to run the app locally: `streamlit run app/streamlit_app.py` and config options.

## 3. Accessibility and safety
- [ ] 3.1 Ensure UI clearly notes model limitations and that uploaded text should not contain PII.
