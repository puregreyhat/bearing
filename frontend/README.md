# Bearing Health Frontend

React + TypeScript dashboard UI using the requested shadcn preset `b3lWZLedl`.

## Run

```powershell
npm install
npm run dev
```

Build for production:

```powershell
npm run build
npm run preview
```

The interface includes the Streamlit workflow: signal source selection, sampling rate, task and model selection, diagnosis result, probabilities, Health Index, waveform/FFT evidence, processing metadata, feature values, artifact metadata, feature importance, validation performance, window decisions, quality warnings, and limitations.

The frontend calls the FastAPI endpoint and renders the existing `InferenceResult.to_dict()` contract. Before the API response arrives, the screen shows a representative preview state.

Run the live backend from the repository root:

```powershell
conda activate bearing
uvicorn backend.main:app --reload --port 8000
```

Set `VITE_API_URL` only when the API is not running at `http://127.0.0.1:8000`.
