# CWRU Bearing Fault Diagnosis

End-to-end bearing-fault benchmark using real CWRU drive-end vibration recordings only.

## Project structure

```text
bearing/
├── app/                 # supported Streamlit dashboard
├── src/                 # data, feature, training, registry, and inference code
├── tests/               # automated verification and SRS acceptance tests
├── data/                # raw and processed CWRU data
├── models/              # self-describing trained artifacts
├── results/             # evaluation reports, registry, and plots
├── docs/                # SRS, guides, plans, and handover documentation
├── demo_csv_files/      # small example inputs for demonstrations
└── website/             # separate generated browser demo
```

The shadcn/React UI is in `frontend/`; the current Streamlit dashboard remains
in `app/`, and the browser demo remains isolated in `website/`.

Run the new frontend with:

```powershell
cd frontend
npm install
npm run dev
```

The frontend is connected to the FastAPI backend and renders the existing
`InferenceResult` contract. It shows a preview state until the API returns a
diagnosis.

Run the live API and frontend together in separate terminals:

```powershell
conda activate bearing
uvicorn backend.main:app --reload --port 8000
cd frontend
npm run dev
```

Recommended environment: Python 3.12. The repository has been verified with the
Conda environment named `bearing`.

```powershell
conda activate bearing
python -m pip install -r requirements.txt
python src/download.py
python src/extract.py
python src/validate.py
python src/train.py
python src/evaluate.py
python src/model_registry.py
streamlit run app/dashboard.py
```

For development and verification, install the test dependencies as well:

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest -q --basetemp .pytest-tmp
```

Raw `.mat` files remain untouched in `data/raw`. Normal recordings (97–100) are 48 kHz and are FIR-decimated by four during extraction; all feature rows therefore represent 12 kHz signals. Segmentation is non-overlapping with incomplete tails dropped. Outputs are `cwru_features.csv` (2048 samples) and `cwru_features_4096.csv`.

Training evaluates binary and four-class tasks with nested, file-grouped cross-validation and Leave-One-Load-Out testing. The dashboard accepts numeric `.csv`, `.npy`, or Drive-End `.mat` signals at 12 or 48 kHz. Results are CWRU laboratory benchmarks and must not be treated as validated ceiling-fan performance.

# WSL commands
cd /mnt/c/Users/IDEAPAD/Documents/GitHub/bearing
conda activate bearing
streamlit run app/dashboard.py

The repository `.venv` currently points to Python 3.14 and is not the supported
environment for this project. Use a Python 3.12 environment until the
dependency set has compatible wheels for Python 3.14.

Training saves Binary and Multiclass artifacts for SVM, Random Forest, and
XGBoost under `models/`. The deployment alias (`binary_model.joblib` or
`multiclass_model.joblib`) remains the model selected by Leave-One-Load-Out
macro-F1. Run `python src/model_registry.py` after training to refresh the
dashboard's compatible-model registry.

The Streamlit dashboard is the supported SRS product surface. The files in
`website/dist/` are a generated browser demo using embedded CWRU examples and
exported Random Forests. Regenerate and verify that demo with:

```powershell
conda activate bearing
python website/export_assets.py
node website/check.mjs
```

Further handover documentation:

- [`docs/dashboard_user_guide.md`](docs/dashboard_user_guide.md) — operator workflow and result interpretation.
- [`docs/developer_guide.md`](docs/developer_guide.md) — environment, regeneration, testing, and maintenance.
- [`docs/artifact_schema.md`](docs/artifact_schema.md) — model artifact, registry, and inference contracts.
- [`docs/static_demo_guide.md`](docs/static_demo_guide.md) — static website role and source of truth.

Run the Phase 6 automated verification suite from the `bearing` environment:

```powershell
python -m pip install -r requirements-dev.txt
pytest -q
```
