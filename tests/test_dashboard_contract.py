from pathlib import Path

from inference import analyze_signal


ROOT = Path(__file__).resolve().parents[1]


def test_dashboard_module_compiles_and_exposes_run_contract():
    source = (ROOT / "app/dashboard.py").read_text(encoding="utf-8")
    compile(source, str(ROOT / "app/dashboard.py"), "exec")
    assert "st.session_state.diagnosis_result" in source
    assert "render_model_evidence(result)" in source


def test_uploaded_signal_contract_matches_dashboard_fields():
    result = analyze_signal([0] * 2048, 12000, "binary", "rf", "upload.csv")
    dashboard_fields = {"predicted_class", "confidence", "probabilities", "warnings"}
    assert dashboard_fields <= set(result.__dataclass_fields__)

