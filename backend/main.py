"""HTTP API used by the React/shadcn dashboard."""

import io
import sys
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import SUPPORTED_INPUT_SAMPLING_RATES  # noqa: E402
from inference import analyze_signal, read_signal  # noqa: E402
from model_registry import discover_registry  # noqa: E402
from .schemas import DiagnosisResponse, ModelOption  # noqa: E402

app = FastAPI(title="Bearing Health API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SAMPLES = {
    "097_normal.csv": ROOT / "demo_csv_files/healthy/097_normal.csv",
    "105_inner_race.csv": ROOT / "demo_csv_files/faulty/inner_race/105_inner_race.csv",
    "118_ball.csv": ROOT / "demo_csv_files/faulty/ball/118_ball.csv",
    "130_outer_race.csv": ROOT / "demo_csv_files/faulty/outer_race/130_outer_race.csv",
}


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "bearing-health-api"}


@app.get("/api/models", response_model=list[ModelOption])
def models(task: str):
    if task not in {"binary", "multiclass"}:
        raise HTTPException(status_code=400, detail="task must be binary or multiclass")
    entries, rejected = discover_registry()
    return [
        ModelOption(
            task=entry["task"],
            model_family=entry["model_family"],
            artifact=entry["artifact"],
            classes=entry["classes"],
            validation=entry.get("validation", {}),
        )
        for entry in entries
        if entry["task"] == task
    ]


@app.post("/api/diagnose", response_model=DiagnosisResponse)
async def diagnose(
    task: str = Form(...),
    model_family: str = Form(...),
    sampling_rate: int = Form(...),
    sample_name: str = Form(""),
    signal_file: UploadFile | None = File(None),
):
    if sampling_rate not in SUPPORTED_INPUT_SAMPLING_RATES:
        raise HTTPException(status_code=400, detail="sampling rate must be 12,000 or 48,000 Hz")
    if task not in {"binary", "multiclass"}:
        raise HTTPException(status_code=400, detail="task must be binary or multiclass")

    if signal_file is not None:
        filename = signal_file.filename or "uploaded_signal"
        source = io.BytesIO(await signal_file.read())
    elif sample_name in SAMPLES:
        filename = sample_name
        source = SAMPLES[sample_name]
    else:
        raise HTTPException(status_code=400, detail="provide an upload or a supported CWRU sample")

    try:
        signal = read_signal(source, filename)
        result = analyze_signal(signal, sampling_rate, task, model_family, filename)
    except Exception as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return result.to_dict()

