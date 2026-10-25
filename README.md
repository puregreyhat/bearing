# Bearing Health

End-to-end bearing fault diagnosis using CWRU drive-end vibration recordings.
The project includes a Python ML pipeline, a FastAPI inference service, and a
React/shadcn dashboard for interactive diagnosis.

## What the project provides

- Binary classification: `Healthy` or `Faulty`.
- Multiclass classification: `Normal`, `Ball`, `Inner_Race`, or `Outer_Race`.
- Random Forest, SVM, and XGBoost model artifacts.
- 25-feature vibration-signal contract with 2,048-sample non-overlapping windows.
- Support for CWRU `.mat` files and numeric `.csv` / `.npy` uploads.
- Live diagnosis through FastAPI and the React frontend.
- Streamlit dashboard for the legacy/reference workflow.
- Evaluation reports, model registry metadata, and automated tests.

## Repository structure

```text
bearing/
|-- backend/              # FastAPI application and response schemas
|-- frontend/             # React + TypeScript + Vite + shadcn UI
|-- app/                  # Streamlit dashboard
|-- src/                  # Data preparation, features, training, registry, inference
|-- tests/                # Automated tests and SRS acceptance checks
|-- data/                 # Raw and processed CWRU data
|-- models/               # Trained model artifacts
|-- results/              # Metrics, reports, plots, and registry metadata
|-- docs/                 # SRS, guides, plans, and handover documentation
|-- demo_csv_files/       # Small example inputs for API demonstrations
`-- website/              # Separate generated static browser demo
```

## Environment setup

Use the Conda environment named `bearing` with Python 3.12. The repository's
`.venv` uses Python 3.14 and is not the verified environment for the current
scientific dependency set.

```powershell
conda create -n bearing python=3.12 -y
conda activate bearing
python -m pip install -r requirements.txt
```

Install test tooling when developing:

```powershell
python -m pip install -r requirements-dev.txt
```

## Run the application

Start the FastAPI backend from the repository root:

```powershell
conda activate bearing
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

In a second terminal, start the React dashboard:

```powershell
cd frontend
npm install
npm run dev
```

On PowerShell systems where `npm` scripts are blocked by execution policy, use
`npm.cmd` instead:

```powershell
npm.cmd install
npm.cmd run dev
```

Open the Vite URL shown in the terminal, normally `http://localhost:5173` or
`http://localhost:5174` if port 5173 is already in use.
The dashboard starts with the **01 Input & configuration** section. After a
diagnosis is submitted, it displays the diagnosis, signal evidence, data
quality, interpretation workspace, and processing details.

### Admin panel

Open `http://localhost:5173/admin` or `http://localhost:5174/admin` to view the separate SQLite-backed admin
panel. Every completed diagnosis is stored in `data/bearing_health.db` with
these five fields: test ID, creation time, input filename, model family, and
predicted result. The database file is intentionally ignored by Git.

### Frontend API configuration

The frontend uses `http://127.0.0.1:8000` by default. To use another backend,
create `frontend/.env.local`:

```text
VITE_API_URL=http://127.0.0.1:8000
```

### Backend endpoints

```text
GET  /api/health
GET  /api/models?task=binary
POST /api/diagnose
GET  /api/admin/tests
GET  /api/admin/tests/{test_id}
```

The backend allows local frontend origins on ports 5173 and 5174. If the UI
shows `Failed to fetch`, restart both services so the current backend code is
running:

```powershell
# Terminal 1
conda activate bearing
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000

# Terminal 2
cd frontend
npm.cmd run dev
```

Example diagnosis request using a bundled sample:

```powershell
curl.exe -X POST `
  -F "task=binary" `
  -F "model_family=rf" `
  -F "sampling_rate=12000" `
  -F "sample_name=097_normal.csv" `
  http://127.0.0.1:8000/api/diagnose
```

For an uploaded file, replace `sample_name` with:

```powershell
-F "signal_file=@C:\path\to\signal.csv"
```

## Streamlit dashboard

The Streamlit dashboard remains available as a reference and fallback product
surface:

```powershell
conda activate bearing
streamlit run app/dashboard.py
```

## Rebuild the ML pipeline

Run these commands from the repository root after activating the `bearing`
environment:

```powershell
python src/download.py
python src/extract.py
python src/validate.py
python src/train.py
python src/evaluate.py
python src/model_registry.py
```

Raw `.mat` files in `data/raw` are preserved. Normal recordings are converted
from 48 kHz to 12 kHz during extraction. Feature rows use non-overlapping
2,048-sample windows, and incomplete tails are discarded.

Training produces binary and multiclass artifacts for Random Forest, SVM, and
XGBoost under `models/`. The registry in `results/model_registry.json` records
the compatible artifacts and validation metadata.

## Verification

Run the Python test suite:

```powershell
conda activate bearing
pytest -q
```

Build the frontend:

```powershell
cd frontend
npm.cmd run build
```

Verify the generated static demo:

```powershell
conda activate bearing
python website/export_assets.py
node website/check.mjs
```

## Important scope limitation

The models are CWRU laboratory benchmarks. Their outputs must not be treated as
validated performance for ceiling-fan motors or other machines outside the
training and validation scope.

## Documentation

- [`docs/IEEE_SRS_Bearing_Fault_Diagnosis_v2_0_UserFlow_Dashboard.md`](docs/IEEE_SRS_Bearing_Fault_Diagnosis_v2_0_UserFlow_Dashboard.md) — SRS and user flow.
- [`docs/SRS_gap_analysis_and_implementation_plan.md`](docs/SRS_gap_analysis_and_implementation_plan.md) — gap analysis and implementation plan.
- [`docs/dashboard_user_guide.md`](docs/dashboard_user_guide.md) — dashboard operation and interpretation.
- [`docs/developer_guide.md`](docs/developer_guide.md) — environment, regeneration, testing, and maintenance.
- [`docs/artifact_schema.md`](docs/artifact_schema.md) — model artifact, registry, and inference contracts.
- [`docs/ML_guide.md`](docs/ML_guide.md) — machine-learning workflow.
- [`docs/static_demo_guide.md`](docs/static_demo_guide.md) — role and maintenance of the static demo.
