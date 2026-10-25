# Developer Guide

## Environment

Use the verified Python 3.12 Conda environment:

```powershell
conda activate bearing
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
```

The repository `.venv` uses Python 3.14 and is not the supported environment for the current scientific dependency set.

## Reproduce the pipeline

Run these commands from the repository root:

```powershell
python src/download.py
python src/extract.py
python src/validate.py
python src/train.py
python src/evaluate.py
python src/model_registry.py
```

Training and evaluation overwrite model and result artifacts. Run them intentionally after confirming that `data/raw/` contains the complete dataset. `src/validate.py` is also verified with base Anaconda if the Conda environment experiences the documented native Windows validation crash.

## Run verification

```powershell
python -m pytest -q --basetemp .pytest-tmp
node website/check.mjs
python -m compileall -q src app
```

The acceptance mapping is recorded in `results/acceptance_report.md`.

## Artifact and registry workflow

`src/train.py` creates task/model candidate artifacts in `models/`, including SVM, Random Forest, and XGBoost candidates, and keeps `binary_model.joblib` and `multiclass_model.joblib` as deployment aliases for the selected LOLO winner. Each artifact includes the feature order, feature version, preprocessing contract, classes, task, model family, validation summary, dataset hashes, and creation time.

After training or replacing artifacts, refresh the registry:

```powershell
python src/model_registry.py
```

The dashboard and Python inference select only artifacts accepted by the registry. If an artifact is rejected, inspect `rejected_artifacts` in `results/model_registry.json` before changing inference code.

## Static demo export

The static website is generated from the deployment aliases and is intentionally limited to embedded CWRU examples plus Random Forest browser inference:

```powershell
python website/export_assets.py
node website/check.mjs
```

Do not hand-edit `website/dist/site-data.js`. The supported inference source remains `src/inference.py` and the Streamlit dashboard.

## Change discipline

- Preserve `FEATURE_NAMES`, `FEATURE_VERSION`, `WINDOW_SIZE`, and preprocessing settings unless the SRS contract is intentionally revised.
- Add or update tests when changing input validation, feature formulas, artifact compatibility, or dashboard result fields.
- Keep metadata columns out of the model feature matrix.
- Keep grouped validation and Leave-One-Load-Out evaluation leakage-safe.
- Regenerate reports and the model registry after model changes.

