# Development Plan for Bearing Fault Diagnosis

## Purpose

This plan converts the IEEE-style Software Requirements Specification version 2.0 and the repository gap analysis into an implementation sequence for the CWRU bearing-fault diagnosis project. It is written for the development team responsible for the machine-learning pipeline, inference layer, Streamlit dashboard, static demo, testing, and documentation.

The plan preserves the current CWRU benchmark and classical-ML pipeline. The first release target is an SRS-compliant diagnostic dashboard, not a prognostics system, a remaining-useful-life estimator, or a field safety product.

## Product Outcome

The completed baseline product shall allow a maintenance or engineering user to:

1. Upload a supported `.mat`, `.csv`, or `.npy` vibration signal.
2. Select the input sampling rate when it cannot be inferred from metadata.
3. Select Binary or Multiclass Classification.
4. Select only a model that is trained, saved, and compatible with that task.
5. Run the approved preprocessing, 2048-sample windowing, and 25-feature pipeline.
6. View the prediction, every supported class probability, confidence, and Health Index where applicable.
7. Inspect the analyzed waveform, FFT, processing details, key features, feature importance, validation performance, data quality, and limitations.
8. Receive clear errors for invalid inputs or unavailable/incompatible artifacts.

## Guiding Decisions

| Decision | Plan |
|---|---|
| Primary interface | Streamlit dashboard is the reference implementation. |
| Static website | Keep as a generated demonstration unless it is deliberately brought into parity with the model registry and inference contract. |
| Primary window | 2048 samples, non-overlapping, incomplete tail discarded. |
| Processed sampling rate | 12 kHz. 48 kHz CWRU normal data is FIR-decimated by four. |
| Features | The existing approved 25-feature set remains the model boundary. |
| Baseline models | SVM, Random Forest, and XGBoost remain training candidates. Only compatible saved artifacts are selectable at runtime. |
| Deployment selection | Mean Leave-One-Load-Out macro-F1 remains the primary model-selection criterion. |
| Health Index | `100 × Healthy probability`; explicitly not RUL or a safety guarantee. |
| Field claims | No field performance claim until naturally collected fan-motor data is separately validated. |

## Target Architecture

```text
Input file and sampling metadata
            |
            v
     Signal loader and validator
            |
            v
   Preprocessing configuration
            |
            v
  12 kHz, 2048-sample windows
            |
            v
        25 features
            |
            v
   Model registry compatibility check
            |
            v
      Selected task and model
            |
            v
 Window predictions and aggregation
            |
            v
 Structured inference result
            |
            v
 Dashboard result and evidence panels
```

The model registry and structured inference result are the central contracts. The dashboard should not reconstruct model metadata, preprocessing assumptions, or aggregation behavior independently.

## Delivery Structure

Development is organized into nine implementation phases. Each phase has an outcome and an exit gate. A phase is complete only when its code, tests, documentation, and generated artifacts are consistent.

### Phase 0 — Baseline and Environment Stabilization

**Objective:** Make the current project reproducible before changing behavior.

Tasks:

- Confirm Python environment and dependency installation from `requirements.txt`.
- Confirm whether `data/raw/` is populated; document the download prerequisite.
- Run the current static-site smoke check.
- Run the current extraction, validation, training, and evaluation path when raw data and dependencies are available.
- Record current model artifacts, feature list, metric outputs, and working-tree state.
- Normalize the README run instructions for Windows/WSL.
- Add a project version or artifact-generation identifier without modifying raw data.

Primary files:

- `README.md`
- `requirements.txt`
- `src/download.py`
- `website/check.mjs`

Exit gate:

- A new developer can install the environment, understand whether raw data is present, run the existing checks, and identify the current deployment artifacts.

### Phase 1 — Configuration and Artifact Contract

**Objective:** Make every saved model self-describing and safely reusable.

Tasks:

- Add a shared configuration module containing the target sampling rate, supported input rates, window size, resampling method, and feature-version identifier.
- Add an artifact schema with task, model family, classes, feature names/order, preprocessing configuration, window size, training dataset identity, selection criterion, validation scheme, metrics, and creation metadata.
- Update `src/train.py` to write the expanded schema into both model artifacts.
- Add `src/model_registry.py` to discover artifacts and report availability and compatibility.
- Generate `results/model_registry.json` from trained artifacts and evaluation outputs.
- Define clear compatibility errors for missing, malformed, task-mismatched, class-mismatched, feature-mismatched, or preprocessing-mismatched artifacts.

Primary files:

- New: `src/config.py`
- New: `src/model_registry.py`
- Update: `src/train.py`
- Update: `src/evaluate.py`
- Update: `src/inference.py`

Traceability: FR-018, FR-023, FR-031, FR-045, FR-046, NFR-007.

Exit gate:

- The registry exposes only artifacts that can actually run, and an artifact can be audited without reading training code.

### Phase 2 — Inference Contract and Input Validation

**Objective:** Refactor inference into a testable, task-aware pipeline without changing approved feature formulas.

Tasks:

- Split `src/inference.py` into explicit functions for loading, validation, resampling, windowing, feature extraction, model execution, and aggregation.
- Introduce a structured `InferenceResult` representation containing input metadata, processing metadata, warnings, windows, features, probabilities, predictions, selected class, confidence, and Health Index.
- Add explicit error classes or stable error messages for empty, malformed, non-numeric, NaN, infinite, unsupported, multidimensional, and too-short inputs.
- Return both window-level and recording-level probabilities.
- Make task and model explicit arguments to inference.
- Validate the selected artifact through the registry before prediction.
- Preserve the exact existing feature order from the artifact.
- Add a quality-warning path for discarded tails and user-supplied sampling-rate assumptions.

Primary files:

- Update: `src/inference.py`
- Update: `src/features.py` only if shared constants or validation helpers are needed
- New: `src/contracts.py` or equivalent result definitions

Traceability: FR-004 to FR-010, FR-032 to FR-039, NFR-001 to NFR-003, NFR-009.

Exit gate:

- Existing valid sample predictions remain consistent, and invalid inputs produce deterministic human-readable errors.

### Phase 3 — Training and Evaluation Evidence

**Objective:** Make the ML outputs usable by the dashboard and traceable to the SRS.

Tasks:

- Preserve grouped outer validation and nested inner tuning.
- Keep the File_ID and load disjointness assertions.
- Standardize metric names, especially the current `accuracy_not_headline` field.
- Export a compact task/model summary containing balanced accuracy, macro-F1, accuracy, per-class precision/recall, binary PR-AUC/ROC-AUC, training time, and inference time.
- Export model-selection results and the selected winner for each task.
- Export confusion-matrix paths and feature-importance paths.
- Add a data identity record including manifest version or hash, feature list, window size, and sampling assumptions.
- Keep model performance claims labeled as CWRU benchmark results.

Primary files:

- Update: `src/train.py`
- Update: `src/evaluate.py`
- Outputs: `results/model_registry.json`, `results/metrics_summary.csv`, `results/model_evaluation.md`, confusion matrices, importance files

Traceability: FR-024 to FR-031, DF-010, DF-011, NFR-007.

Exit gate:

- The dashboard can retrieve all evidence for a selected task/model without parsing human-oriented report text.

### Phase 4 — Dashboard Workflow

**Objective:** Implement the finalized SRS user flow in Streamlit.

Tasks:

- Add an input panel with supported formats, minimum samples, and sampling-rate guidance.
- Add a Binary/Multiclass task selector.
- Populate the model selector from the registry for the selected task.
- Hide unavailable models instead of displaying nonfunctional choices.
- Add a clear Predict action and a reset/re-run behavior.
- Display the selected task and model in the result header.
- Show the task-appropriate classes and prediction summary.
- Display every class probability numerically and visually.
- Show Health Index only when the selected task and class mapping support Healthy probability.
- Add human-readable error handling and preserve the last valid result where appropriate.

Primary files:

- Update: `app/dashboard.py`
- Update: `src/model_registry.py`
- Update: `src/inference.py`

Traceability: FR-040 to FR-043, DF-001 to DF-005, AC-011 to AC-021, NFR-001, NFR-002, NFR-008.

Exit gate:

- A non-ML user can complete upload, task selection, compatible model selection, prediction, and result interpretation without code.

### Phase 5 — Dashboard Evidence and Interpretation

**Objective:** Add the secondary panels required for trustworthy interpretation.

Tasks:

- Add waveform visualization and label it as the first analyzed window or an overview, depending on the data shown.
- Add FFT visualization with frequency units and the 12 kHz processing assumption.
- Display source rate, processed rate, window size, signal duration, valid-window count, and discarded-tail count.
- Add a key-feature table for the first window and/or aggregate values.
- Add model/task-specific feature importance from generated results.
- Add model performance metrics and validation scheme.
- Add a confusion-matrix image or link for the selected task/model.
- Add data-quality status and warnings.
- Add a limitations panel covering CWRU laboratory scope, domain shift, uncalibrated probability limitations, and non-RUL Health Index meaning.

Primary files:

- Update: `app/dashboard.py`
- Update: `results` generation in `src/evaluate.py`

Traceability: DF-006 to DF-012, AC-012, AC-015, AC-020, AC-021.

Exit gate:

- All 12 finalized dashboard feature groups from the SRS are available, with primary diagnosis information more prominent than secondary evidence.

### Phase 6 — Automated Verification

**Objective:** Convert the SRS acceptance criteria into repeatable tests.

Test structure:

```text
tests/
  test_signal_loading.py
  test_preprocessing.py
  test_windowing.py
  test_features.py
  test_artifacts.py
  test_inference.py
  test_training_validation.py
  test_dashboard_contract.py
  test_acceptance.py
```

Unit tests:

- MAT, CSV, and NPY loading.
- Empty, short, malformed, NaN, infinity, and unsupported inputs.
- 48 kHz decimation and 12 kHz pass-through.
- 2048-sample windowing, tail handling, and duration calculations.
- RMS, statistical, FFT, entropy, and band-energy calculations.
- Feature order and metadata exclusion.
- Label encoding and decoding.
- Model loading and probability-to-class mapping.

Integration tests:

- Signal to preprocessing to windows to features.
- Feature matrix to selected model to probability aggregation.
- Uploaded file to dashboard result contract.
- Registry to compatible task/model selection.

Negative tests:

- Missing artifact.
- Feature-name mismatch.
- Task mismatch.
- Class mapping mismatch.
- Unsupported sampling-rate configuration.
- Invalid model configuration.

Acceptance tests:

- Map AC-001 through AC-021 to automated checks or documented manual checks.
- Keep an acceptance report in `results/acceptance_report.md`.

Traceability: SRS section 10, AC-001 to AC-021, NFR-002, NFR-004, NFR-007.

Exit gate:

- A clean environment can run the test suite and produce a requirement-linked result.

### Phase 7 — Static Website Alignment

**Objective:** Decide and document the role of the static website.

Preferred option:

- Treat Streamlit as the supported dashboard and static website as a generated demo.
- Clearly label the static site as a browser demo if it does not support the full SRS workflow.
- Regenerate `website/dist/site-data.js` only through `website/export_assets.py` after model artifacts change.

Alternative option:

- Bring the static website into parity by exporting the model registry, task/model choices, preprocessing metadata, and all required evidence panels.
- Add a browser-side contract test that matches the Python inference outputs for representative inputs.

Do not maintain two independent sources of truth for feature formulas, model availability, or prediction aggregation.

Traceability: NFR-004, NFR-005, AC-019, AC-020.

Exit gate:

- The project documentation clearly identifies the supported product surface and the source of truth for inference.

### Phase 8 — Documentation and Handover

**Objective:** Make the project maintainable by another developer or evaluator.

Tasks:

- Update `README.md` with setup, raw-data download, training, dashboard, and static-demo commands.
- Add `docs/dashboard_user_guide.md`.
- Add `docs/developer_guide.md` covering artifact regeneration and model registry updates.
- Add `docs/artifact_schema.md`.
- Add `results/acceptance_report.md`.
- Document the SRS version and resolve its version-table inconsistency.
- Document that current models are CWRU benchmark models and not field-validated fan diagnostics.
- Record known limitations and future enhancements separately from baseline requirements.

Traceability: SRS deliverables 38 to 47, NFR-004, NFR-007, NFR-008.

Exit gate:

- A new developer can reproduce the pipeline, run the dashboard, understand model availability, and interpret results without reverse-engineering source code.

## Module-Level Change Plan

| Module | Planned changes | Priority |
|---|---|---|
| `src/features.py` | Preserve formulas; add constants/versioning and tests | P0 |
| `src/inference.py` | Refactor into structured, task-aware inference contract | P0 |
| `src/train.py` | Save expanded artifact metadata and registry inputs | P0 |
| `src/evaluate.py` | Export machine-readable model evidence and selection summaries | P0 |
| `app/dashboard.py` | Implement SRS user flow and evidence panels | P0 |
| `src/model_registry.py` | New artifact discovery and compatibility layer | P0 |
| `src/config.py` | New shared preprocessing and feature configuration | P0 |
| `tests/` | New unit, integration, negative, and acceptance suite | P0 |
| `website/export_assets.py` | Make export metadata explicit and deterministic | P1 |
| `website/dist/` | Align or label as demo | P1 |
| `README.md` | Update operational instructions and limitations | P1 |
| `docs/` | Add user, developer, schema, and acceptance documentation | P1 |

## Dependency Order

```text
Configuration and artifact schema
          |
          v
Model registry and compatibility checks
          |
          v
Structured inference result
          |
          +------------------+
          v                  v
Dashboard workflow     Evidence/report exports
          |                  |
          +--------+---------+
                   v
             Automated tests
                   |
                   v
             Documentation and handover
```

The dashboard should not be implemented first against ad hoc model files. The registry and inference contract are dependencies for reliable task/model selection and correct result interpretation.

## Definition of Done

The baseline implementation is complete when:

- All baseline functional requirements FR-001 through FR-046 are either implemented or have an explicitly documented approved exception.
- The dashboard follows the SRS sequence and exposes only compatible models.
- The 25-feature vector and preprocessing configuration are shared between training and inference.
- Model artifacts are self-describing and validated before use.
- All required dashboard information groups are available.
- Invalid inputs produce clear errors without an application crash.
- Tests cover the SRS unit, integration, negative, and acceptance requirements.
- Model evaluation remains grouped and leakage-safe.
- The Health Index is not described as RUL or a safety guarantee.
- CWRU benchmark performance is not represented as field validation.
- README, user, developer, artifact, and acceptance documentation are updated.

## Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Dashboard offers models without compatible artifacts | Incorrect or failed predictions | Registry-driven model selector and artifact compatibility tests |
| Training and inference feature drift | Silent prediction errors | Shared configuration, feature-version identifier, and feature-order tests |
| Static and Python inference diverge | Inconsistent user results | One canonical inference path or browser parity tests |
| High CWRU scores are overinterpreted | Unsafe field decisions | Persistent benchmark limitation and no-safety-guarantee messaging |
| Raw data is missing from a new environment | Incomplete reproduction | Startup data check and documented download step |
| Probability values are treated as calibrated confidence | Misleading diagnosis | Use model-probability wording and track calibration as future work |
| New field data is incompatible with CWRU assumptions | False predictions | Input contract, quality warnings, and future domain-validation phase |

## Deferred Work

The following SRS future enhancements are intentionally outside the baseline plan:

- Naturally degraded ceiling-fan dataset collection.
- Multi-sensor fusion with temperature, current, RPM, or operating state.
- Probability calibration.
- Unknown/anomaly detection.
- Envelope-spectrum features.
- Transfer learning.
- 1D-CNN comparison.
- RUL/prognostics.

These should begin only after the baseline dashboard, artifact contract, tests, and acceptance report are complete.

## Requirement Traceability Summary

| SRS area | Planned phases | Evidence of completion |
|---|---|---|
| Dataset and input FR-001 to FR-004 | 0, 2, 6 | Input tests and data-quality result |
| Preprocessing FR-005 to FR-010 | 1, 2, 6 | Shared config, processing metadata, preprocessing tests |
| Features FR-011 to FR-014 | 1, 2, 6 | Feature version/order tests and traceable rows |
| Classification FR-015 to FR-018 | 1, 2, 4, 6 | Task selector, class mapping, artifact tests |
| Candidate models FR-019 to FR-023 | 0, 1, 3 | Training outputs, registry, reproducibility metadata |
| Evaluation FR-024 to FR-031 | 3, 6 | Metrics, confusion matrices, leakage and acceptance reports |
| Inference FR-032 to FR-039 | 2, 4, 5, 6 | Structured result and end-to-end tests |
| Dashboard FR-040 to FR-044 and DF-001 to DF-012 | 4, 5, 6 | Streamlit workflow and manual/automated acceptance checks |
| Artifacts FR-045 to FR-046 | 1, 3, 6 | Self-describing artifacts and compatibility tests |
| Non-functional requirements | 0, 1, 2, 4, 6, 8 | Reproducibility, usability, performance, security, and documentation evidence |

## Immediate Next Actions

1. Confirm and document the working Python environment and raw-data availability.
2. Add shared configuration and model artifact schema.
3. Implement the model registry.
4. Refactor `src/inference.py` to return a structured task/model-aware result.
5. Add unit tests for loading, preprocessing, windowing, and feature order.
6. Implement the dashboard task/model selection flow.
7. Add model evidence and data-quality panels.
8. Complete acceptance tests and update the README and developer documentation.

