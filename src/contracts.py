"""Structured contracts returned by the inference layer."""

from dataclasses import asdict, dataclass, field
from typing import Optional


@dataclass
class InferenceResult:
    """Recording-level diagnosis plus traceability and evidence metadata."""

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
    warnings: list[str] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)
