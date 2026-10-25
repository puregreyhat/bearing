# Phase 0 Baseline and Environment Stabilization

## Date

24 October 2026

## Result

Phase 0 is substantially complete. The repository has a usable Anaconda
scientific environment, all 40 CWRU recordings are present, extraction and
dataset validation pass, and the existing browser smoke check and demo
inference path pass. The local `.venv` is not usable as the project environment
because it is Python 3.14, contains only pip, and cannot install the current
scientific stack without source builds.

## Checks Performed

| Check | Result | Evidence |
|---|---|---|
| Repository state captured | PASS | Existing user changes preserved; no reset or cleanup performed |
| Conda environment | PASS | `bearing`, Python 3.12.14 |
| Required scientific stack | PASS | NumPy, Pandas, SciPy, scikit-learn, XGBoost, Joblib, Matplotlib, Streamlit import successfully in Conda `bearing` |
| Browser demo smoke check | PASS | `node website/check.mjs` verifies four samples, both forests, and `/demo` |
| Demo inference | PASS | `demo_csv_files/097_normal.csv`: 4096 samples, two windows, Healthy/Normal result |
| Saved deployment artifacts | PASS | `models/binary_model.joblib`, `models/multiclass_model.joblib` present |
| Processed data and reports | PASS | Feature CSVs and evaluation outputs present |
| Raw CWRU recordings | PASS | 40 `.mat` files present and the downloader regenerated a 40-row manifest |
| Existing `.venv` | BLOCKED | Python 3.14, no project packages; `pip install -r requirements.txt` fails while building CMake/Ninja dependencies |
| Feature extraction | PASS | 2,331 windows written for 2048 samples and 1,146 windows written for 4096 samples |
| Dataset validation | PASS in base Anaconda | `src/validate.py` completed with 0 missing values, 0 infinite values, unique window IDs, and manifest label checks passing |
| Dataset validation in Conda `bearing` | BLOCKED | The same script exits with native Windows code `-1066598273` and no Python traceback; use base Anaconda for validation until the environment is rebuilt |
| Python syntax compilation | PASS | All 8 Python files under `src/` and `app/` compile successfully |

## Environment Decision

Use the Conda environment named `bearing` with Python 3.12 for development and
execution. The README now documents this as the recommended environment. The
direct `joblib` dependency has also been added because the runtime imports it
explicitly for saved model loading.

The existing `.venv` was not deleted or recreated because it is an existing
workspace artifact and may be needed for separate user workflows. It should not
be used as the project environment until a Python 3.12-compatible environment
is created and verified.

## Data Decision

The current `data/raw/` directory now contains all 40 selected recordings. The
manifest-driven downloader successfully inspected every Drive-End channel and
wrote the 40-row manifest. Extraction and validation were then rerun against
the complete raw set.

Do not overwrite the existing model or results artifacts until the complete raw
dataset is available and the regeneration run is intentionally approved.

## Phase 0 Exit Conditions

- [x] Supported development environment identified.
- [x] README run instructions normalized.
- [x] Direct runtime dependency declared.
- [x] Existing artifacts and smoke checks recorded.
- [x] Complete 40-file CWRU raw dataset restored.
- [x] Full validation script completes successfully in the base Anaconda environment.
- [x] Feature extraction rerun from the complete raw data.
- [ ] Resolve or retire the crashing Conda `bearing` environment.
- [ ] Train and evaluate a new model baseline after intentional approval.

## Next Phase 0 Actions

1. Rebuild the Conda `bearing` environment with Python 3.12, or use the verified base Anaconda environment for current training and validation commands.
2. Decide whether the regenerated feature CSVs should become the frozen baseline.
3. Run training and evaluation only after intentional approval because those commands overwrite model and result artifacts.
4. Freeze the resulting artifact/results baseline before starting Phase 1.
