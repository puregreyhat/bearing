"""HTTP API used by the React/shadcn dashboard."""

import io
import sys
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import SUPPORTED_INPUT_SAMPLING_RATES  # noqa: E402
from inference import analyze_signal, read_signal  # noqa: E402
from model_registry import discover_registry  # noqa: E402
from .database import create_test_log, get_test_log, init_db, list_test_logs  # noqa: E402
from .schemas import AdminTestLog, AdminTestLogList, DiagnosisResponse, ModelOption  # noqa: E402

app = FastAPI(title="Bearing Health API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
init_db()

SAMPLES = {
    "097_normal.csv": ROOT / "demo_csv_files/healthy/097_normal.csv",
    "105_inner_race.csv": ROOT / "demo_csv_files/faulty/inner_race/105_inner_race.csv",
    "118_ball.csv": ROOT / "demo_csv_files/faulty/ball/118_ball.csv",
    "130_outer_race.csv": ROOT / "demo_csv_files/faulty/outer_race/130_outer_race.csv",
}


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "bearing-health-api"}


@app.get("/api/admin/tests", response_model=AdminTestLogList)
def admin_tests(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    search: str = Query("", max_length=120),
):
    items, total = list_test_logs(limit=limit, offset=offset, search=search)
    return AdminTestLogList(items=items, total=total, limit=limit, offset=offset)


@app.get("/api/admin/tests/{test_id}", response_model=AdminTestLog)
def admin_test(test_id: str):
    record = get_test_log(test_id)
    if record is None:
        raise HTTPException(status_code=404, detail="test log not found")
    return record


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
    response = result.to_dict()
    response["test_id"] = create_test_log(
        input_filename=filename,
        model_family=model_family,
        predicted_result=result.predicted_class,
    )["test_id"]
    return response

