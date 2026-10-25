# Static Website Alignment

The Streamlit application is the supported product surface for the SRS workflow. It is the reference implementation for signal loading, preprocessing, task/model selection, inference, evidence, warnings, and limitations.

The files under `website/dist/` are a generated browser demo. They contain embedded CWRU examples and exported Random Forest trees so the workflow can be shown without a Python server. The demo intentionally does not claim full parity with the dashboard: it does not support arbitrary MAT/NPY input, registry-driven model selection, SVM/XGBoost execution, or the complete evidence panels.

## Source of truth

- Python feature and inference behavior: `src/features.py`, `src/config.py`, and `src/inference.py`
- Supported dashboard: `app/dashboard.py`
- Model availability: `results/model_registry.json`
- Browser-demo export: `website/export_assets.py`
- Browser-demo contract check: `website/check.mjs`

## Regenerate and verify

Run after model artifacts or the inference contract change:

```powershell
conda activate bearing
python website/export_assets.py
node website/check.mjs
```

The export writes `website/dist/site-data.js` with the feature version, preprocessing metadata, supported demo model family, and explicit limitations. Do not hand-edit that generated data file.

