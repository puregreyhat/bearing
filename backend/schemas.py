from typing import Any, Optional

from pydantic import BaseModel


class ModelOption(BaseModel):
    task: str
    model_family: str
    artifact: str
    classes: list[str]
    validation: dict[str, Any]


class DiagnosisResponse(BaseModel):
    task: str
    model_family: str
    classes: list[str]
    input_filename: Optional[str]
    source_sampling_rate: int
    processed_sampling_rate: int
    sample_count: int
    duration_seconds: float
    window_size: int
    valid_window_count: int
    discarded_tail_samples: int
    probabilities: dict[str, float]
    window_probabilities: list[dict[str, float]]
    window_predictions: list[str]
    predicted_class: str
    confidence: float
    health_index: Optional[float]
    first_window: list[float]
    first_window_features: dict[str, float]
    warnings: list[str]

