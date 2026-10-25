# SRS Acceptance Test Report

Phase 6 automated verification is implemented in `tests/` and was run on 2026-10-25 with `pytest` from the verified `bearing` environment.

## Result

| Item | Status |
|---|---|
| Automated test suite | PASS — 25 passed |
| Acceptance criteria mapped | AC-001 through AC-021 |
| Manual UI checks | Streamlit health and static smoke checks remain required |
| Phase 8 handover documentation | PASS — user, developer, artifact, and static-demo guides added |

## Acceptance mapping

| Criterion | Automated coverage |
|---|---|
| AC-001 | `test_signal_loading.py`: CSV, NPY, MAT loading |
| AC-002 | `test_signal_loading.py`: unsupported and malformed input |
| AC-003 | `test_signal_loading.py`: empty, NaN, infinity input |
| AC-004 | `test_preprocessing.py`: 12 kHz pass-through |
| AC-005 | `test_preprocessing.py`: 48 kHz decimation |
| AC-006 | `test_windowing.py`: 2048-sample windows and tail handling |
| AC-007 | `test_features.py`: approved feature formulas and finite output |
| AC-008 | `test_training_validation.py`: processed labels and file groups |
| AC-009 | `test_artifacts.py`: feature contract and artifact compatibility |
| AC-010 | `test_artifacts.py`: class/task compatibility |
| AC-011 | `test_inference.py`: selected model inference |
| AC-012 | `test_inference.py`: structured dashboard result |
| AC-013 | `test_inference.py`: window-level results and aggregation |
| AC-014 | `test_dashboard_contract.py`: dashboard source contract |
| AC-015 | `test_dashboard_contract.py`: result evidence fields |
| AC-016 | `test_dashboard_contract.py`: upload-to-result contract |
| AC-017 | `test_training_validation.py`: LOLO validation metadata |
| AC-018 | `test_artifacts.py`: incompatible artifact rejection |
| AC-019 | `test_preprocessing.py`: unsupported configuration rejection |
| AC-020 | `test_inference.py`: processing warnings |
| AC-021 | `test_acceptance.py`: complete SRS criterion mapping |

## Manual checks

- Run `streamlit run app/dashboard.py` and verify the upload/sample, task, model, Run Diagnosis, result, evidence, and limitations flow.
- Run `node website/check.mjs` to verify the static demo smoke contract.
- Confirm the displayed model and task match the selected registry entry.

