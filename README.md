# Function Explorer – Quadratic MVP

## Getting Started (Dash)

1. **Python 3.11** is required. Create a virtual environment in the repo root:
   ```
   python -m venv .venv
   ```
2. Activate it (PowerShell example):
   ```
   .\.venv\Scripts\activate
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Run the Dash app:
   ```
   python dash_app.py
   ```
5. Interaction logs are written to `function_explorer/data/dash_events.jsonl`.

## Legacy (Streamlit)

The original Streamlit UI is still available but no longer the primary experience.

```
streamlit run streamlit_legacy/app.py
```

Expect reduced performance and limited functionality compared with the Dash proof-of-concept.
