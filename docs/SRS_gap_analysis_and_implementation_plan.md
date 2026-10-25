# Executive Summary

The repository already implements most of the CWRU benchmark and core machine-learning pipeline described by the supplied Software Requirements Specification (SRS) version 2.0. The strongest existing coverage is the data flow from CWRU recordings through resampling, fixed windowing, 25 approved signal features, grouped validation, model selection, saved artifacts, and basic inference.

The project is not yet fully compliant with the SRS dashboard scope. The largest work is to make the dashboard follow the finalized user flow — upload signal, select classification task, select a compatible available model, predict, and inspect results — and to expose the evidence needed to interpret a diagnosis. This includes model/task selection, model metadata, feature importance, validation performance, signal-quality status, processing details, and explicit limitations.

The recommended implementation strategy is to preserve the existing CWRU and feature pipeline, strengthen the model-artifact contract, add a small model registry, refactor inference around an explicit task/model selection, and then build the dashboard panels and tests on top of those stable interfaces.

# Scope and Source Basis

This document compares the repository at `C:\Users\IDEAPAD\Documents\GitHub\bearing` with `IEEE_SRS_Bearing_Fault_Diagnosis_v2_0_UserFlow_Dashboard.docx` supplied by the user. The SRS is treated as the product specification. The repository is treated as the current implementation baseline.

The SRS contains a minor document-control inconsistency: its title says version 2.0, while some metadata tables still say version 1.0. This does not change the implementation requirements, but the SRS should be normalized before it is used as a controlled baseline.

# Current Implementation Baseline

The current system contains these working areas:

| Area | Current implementation | Main location |
|---|---|---|
| CWRU dataset | Manifest-driven CWRU recordings, Drive-End channel selection, raw-file preservation | `docs/data_manifest.csv`, `src/download.py` |
| Preprocessing | 48 kHz to 12 kHz zero-phase FIR decimation; 12 kHz accepted directly | `src/extract.py`, `src/inference.py` |
| Windowing | Non-overlapping 2048-sample windows; incomplete tails dropped | `src/extract.py`, `src/inference.py` |
| Features | 25 time- and frequency-domain features | `src/features.py` |
| Validation | Label/data checks, finite-value checks, unique window IDs, signal plots | `src/validate.py` |
| Training | SVM, Random Forest, XGBoost; nested grouped tuning; LOLO and grouped 4-fold evaluation | `src/train.py` |
| Evaluation | Metrics, confusion matrices, feature importance, leakage audit | `src/evaluate.py` |
| Deployment models | Binary and multiclass joblib artifacts; current deployment artifacts are Random Forest-based | `models/` |
| Inference | CSV, NPY, and MAT loading; window-level inference; probability averaging | `src/inference.py` |
| Streamlit dashboard | Upload/CWRU sample, condition, confidence, fault type, health index, class chart, first-window waveform and FFT | `app/dashboard.py` |
| Static website | Browser demo with embedded samples and exported Random Forest trees | `website/dist/`, `website/export_assets.py` |
| Verification | Static-site smoke check; no dedicated Python unit/integration/negative test suite | `website/check.mjs` |

# Gap Status Legend

| Status | Meaning |
|---|---|
| Implemented | The requirement is materially covered and should be regression-tested. |
| Partial | Some behavior exists, but the SRS behavior, evidence, or interface is incomplete. |
| Gap | Required behavior is missing. |
| Future | Explicitly identified by the SRS as an enhancement rather than a baseline requirement. |

# Requirements Gap Analysis

## Dataset, Input, and Preprocessing

| SRS requirement | Status | Evidence in repository | Required action |
|---|---|---|---|
| FR-001 to FR-003: CWRU and MAT/CSV/NPY input | Implemented | Manifest, `src/download.py`, `src/inference.py` | Add tests and user-facing input guidance. |
| FR-004: empty, short, non-numeric, missing, and infinite input detection | Partial | `read_signal` rejects non-finite and non-1D input; `prepare_windows` rejects too-short input | Add explicit error categories and negative tests for empty files, malformed CSV, NaN, infinity, unsupported extension, and short signals. |
| FR-005: identify sampling rate | Partial | Upload flow asks the user to select 12 kHz or 48 kHz; MAT samples use manifest metadata | Store sampling-rate metadata in the input result and show it in the dashboard. Do not claim automatic identification for arbitrary CSV/NPY files unless a metadata convention is added. |
| FR-006: CWRU downsampling | Implemented | FIR decimation by four is used in extraction and inference | Centralize this configuration and test numerical consistency between training and inference. |
| FR-007: preprocessing consistency | Partial | Both paths use the same 12 kHz feature logic, but configuration is implicit in code | Save preprocessing configuration with artifacts and validate it at inference time. |
| FR-008 to FR-010: 2048, non-overlap, tail handling | Implemented | `WINDOW_SIZE = 2048`; reshape/range logic drops tails | Expose window size, valid-window count, discarded-tail count, and signal duration in the dashboard. |

## Features and Classification

| SRS requirement | Status | Evidence in repository | Required action |
|---|---|---|---|
| FR-011 to FR-013: approved features and traceable rows | Implemented | `src/features.py`, `src/extract.py` | Add unit tests that compare known feature outputs and verify feature order. |
| FR-014: metadata excluded from model features | Implemented | `EXCLUDED` set in `src/train.py` | Preserve the assertion and add a regression test against accidental metadata inclusion. |
| FR-015 to FR-018: binary/multiclass tasks and label mapping | Implemented | `train.py` creates both tasks; artifacts store classes | Make task selection explicit in the dashboard and expose the selected target/classes in results. |
| FR-019 to FR-021: SVM, RF, XGBoost baselines | Implemented for training | `model_specs` and grid search exist | Retain task-specific candidate artifacts or explicitly expose only the saved deployment model. Add an artifact inventory. |
| FR-022: optional 1D-CNN | Future | Not implemented | Do not add to the baseline dashboard. Track as a later experiment requiring raw-window data, separate validation, and a saved compatible artifact. |
| FR-023: reproducibility | Partial | Fixed seed is used in training | Save training configuration, package versions, dataset identity, feature list, and artifact creation time. |

## Evaluation and Model Selection

| SRS requirement | Status | Evidence in repository | Required action |
|---|---|---|---|
| FR-024 to FR-026: leakage-safe grouped and nested validation | Implemented | `StratifiedGroupKFold`, `LeaveOneGroupOut`, inner grouped grid search | Add automated tests for disjoint File_ID groups and held-out loads. |
| FR-027: required metrics | Implemented | Balanced accuracy, macro-F1, accuracy, per-class precision/recall, binary PR-AUC/ROC-AUC | Normalize the metric naming and expose the selected model’s summary through a machine-readable registry. |
| FR-028: confusion matrices | Implemented | `src/evaluate.py` writes result images | Add links or thumbnails in the dashboard’s model-performance panel. |
| FR-029: inference time | Implemented in evaluation | `inference_ms_per_window` is recorded | Surface it as model benchmark information, not as a guarantee for every uploaded file. |
| FR-030: model comparison | Implemented in reports | `metrics_summary.csv`, `model_evaluation.md` | Provide a dashboard comparison table or an expandable model-performance view. |
| FR-031: LOLO-based deployment selection | Implemented in training | Winner selected by mean LOLO macro-F1 and refit | Record winner, selection metric, validation scheme, and candidate scores in the artifact/registry. |

## Inference and Model Artifacts

| SRS requirement | Status | Evidence in repository | Required action |
|---|---|---|---|
| FR-032 to FR-037: new-signal and window-level inference | Implemented | `src/inference.py` | Refactor `predict` to accept an explicit task/model and return window-level plus recording-level results. |
| FR-034: recording aggregation | Implemented | Mean probabilities across windows | Also return per-window predictions and aggregation count for transparency. |
| FR-035 to FR-037: probabilities, condition, fault type | Partial | Basic dashboard shows both tasks simultaneously | Make output conditional on selected task and always show all supported classes for that task. |
| FR-038 to FR-039: health index interpretation | Implemented | Health index is healthy probability and is labeled as model score | Repeat the non-RUL/non-safety limitation in the main result and limitations panel. |
| FR-045: reusable model saving | Implemented | Joblib artifacts exist | Add artifact validation and a clear missing/incompatible-model error. |
| FR-046: configuration saving | Partial | Features, classes, task, and selection are saved | Add window size, target sampling rate, resampling method, feature-version identifier, model family, training dataset identity, and validation summary. |

## Dashboard and User Flow

| SRS requirement | Status | Evidence in repository | Required action |
|---|---|---|---|
| FR-040 to FR-043: dashboard, upload, results, clear errors | Partial | Streamlit app works for upload and basic results | Add explicit staged workflow, friendly validation messages, and a reset/re-run path. |
| DF-001: prediction summary | Partial | Condition/fault type are shown | Make the summary task-aware and show the selected task/model beside it. |
| DF-002: every class probability | Implemented | Multiclass bar chart is shown | Add binary probabilities as well and show numeric values for all bars. |
| DF-003: health index | Implemented | Health index is shown | Keep current formula and improve interpretation text. |
| DF-004: selected task | Gap | No task selector; both tasks always run | Add Binary/Multiclass selector and task-specific output. |
| DF-005: selected model | Gap | No model selector or model inventory | Add compatible model selector populated from a registry; initially expose only available RF artifacts. |
| DF-006: input signal visualization | Partial | Only the first window is plotted | Add uploaded-signal overview or clearly label the plot as the first analyzed window; show the selected window count. |
| DF-007: processing information | Partial | Sampling rate and tail warning are partly shown | Add source rate, processed rate, window size, duration, valid windows, discarded samples, and preprocessing method. |
| DF-008: frequency spectrum | Partial | First-window FFT exists | Match feature-processing assumptions exactly, show frequency axis and units, and label the analyzed window. |
| DF-009: key features | Gap | No feature table in Streamlit | Add a readable selected-window/aggregate feature table using the approved 25-feature set. |
| DF-010: feature importance | Gap | Files exist in `results/`, not dashboard | Add model/task-specific feature-importance view and explain association is not causality. |
| DF-011: model performance | Gap | Reports exist, not dashboard | Add selected-model metrics, validation scheme, fold variability, and confusion-matrix link/image. |
| DF-012: data quality and limitations | Partial | Exceptions and benchmark warning exist | Add explicit data-quality status, input assumptions, unsupported-domain warning, and a limitations expander. |
| AC-016 to AC-021 | Partial | Basic upload and results work | Complete the selection flow and evidence panels, then write acceptance tests for each criterion. |

# What to Add

## 1. Model Registry and Artifact Contract

Add a small registry module, for example `src/model_registry.py`, that discovers artifacts and exposes:

- task: binary or multiclass;
- model family: SVM, Random Forest, or XGBoost;
- artifact path and availability;
- supported classes;
- feature names and order;
- window size and processed sampling rate;
- preprocessing method and feature-version identifier;
- validation scheme and summary metrics;
- training timestamp and dataset identity.

The registry must reject an artifact when the task, feature list, class list, window size, or sampling rate is incompatible. This is the control that prevents the dashboard from offering a model that cannot actually run.

## 2. Explicit Inference Result Contract

Change inference to return a structured result containing:

- selected task and model;
- input filename and format;
- source and processed sampling rates;
- source sample count and duration;
- window size, valid-window count, and discarded-tail count;
- window-level probabilities/predictions;
- recording-level aggregated probabilities;
- selected class and confidence;
- health index when the binary Healthy class exists;
- first-window features and plot data;
- quality warnings.

This avoids duplicating calculations in the dashboard and makes the same behavior testable from the command line.

## 3. Dashboard Panels

Implement the SRS presentation guidance in this order:

1. Input and workflow controls: upload, source sampling rate, task, compatible model, predict.
2. Primary result: condition or fault class, confidence, class probabilities, health index.
3. Signal evidence: waveform and FFT of the first analyzed window, with processing metadata.
4. Quality and traceability: valid windows, discarded tail, validation warnings, input assumptions.
5. ML evidence: key features, feature importance, validation metrics, confusion matrix.
6. Limitations: CWRU benchmark status, domain shift, non-RUL health index, and no safety guarantee.

Use expanders or secondary tabs for the evidence panels so the primary diagnosis remains readable.

## 4. Tests

Add a `tests/` package with unit, integration, negative, and acceptance tests. At minimum cover every test item listed in SRS section 10: loading, resampling, windowing, feature formulas, label mapping, model loading, probability mapping, pipeline integration, and all invalid-input cases.

## 5. Documentation and Reproducibility

Add:

- a dashboard user guide;
- a developer guide for retraining and artifact generation;
- an artifact schema document;
- a model registry generated from evaluation outputs;
- a test report mapped to SRS acceptance criteria;
- environment/package lock information;
- a documented data-download step and raw-data availability check.

# What to Update

## `src/inference.py`

Preserve the existing numerical behavior but split the code into explicit loading, validation, preprocessing, windowing, feature extraction, model execution, aggregation, and result-building functions. Add configuration-driven validation rather than relying on module constants alone.

## `app/dashboard.py`

Replace the current always-run-both-tasks flow with the SRS sequence. Add task/model selectors, use the model registry to populate compatible choices, display the complete processing metadata, and add secondary evidence panels.

## `src/train.py`

Keep the current grouped/nested evaluation. Extend the saved artifact payload with all inference configuration and a stable model identifier. Consider saving task-specific candidate artifacts only if the dashboard is intended to offer SVM and XGBoost at runtime.

## `src/evaluate.py`

Produce a machine-readable model registry or `results/model_registry.json` in addition to CSV and Markdown reports. Include the selected deployment model and the metrics required by the dashboard.

## `website/`

Decide whether the static site is a supported product surface or an explanatory demo. If supported, it must implement the same task/model availability rules and artifact metadata as Streamlit. Otherwise label it clearly as a static demo and make Streamlit the reference implementation. Regenerate `site-data.js` whenever model artifacts change.

## `README.md` and documentation

Document both run modes, the requirement for raw CWRU files when using CWRU samples, upload sampling-rate assumptions, model availability, and the distinction between benchmark performance and field validation.

# What to Remove or Retire

No core CWRU or ML stage should be removed. The following behaviors should be removed or retired:

- Remove the implicit behavior where the dashboard always executes both tasks when the SRS requires a selected classification task.
- Remove any UI promise of SVM or XGBoost selection unless compatible saved artifacts are actually retained and validated.
- Retire duplicated prediction logic in the static site if it diverges from the canonical Python inference path; either regenerate it from artifacts or mark it as a demo.
- Remove ambiguous labels such as a bare “confidence” when the value is a class probability from an uncalibrated model. Label it as model probability/confidence and add the calibration limitation.
- Do not add or retain wording that implies Health Index is RUL, remaining life, safety certification, or a maintenance threshold.

# Recommended Build Sequence

| Phase | Work | Exit condition |
|---|---|---|
| 1. Contract hardening | Define artifact schema, model registry, inference result schema, and feature-version identifier | A model can be rejected cleanly when incompatible. |
| 2. Inference refactor | Implement explicit task/model selection and structured results without changing feature formulas | Existing sample predictions remain numerically consistent. |
| 3. Dashboard workflow | Add upload, sampling rate, task, compatible model, predict/reset flow | User can complete the SRS user flow without code. |
| 4. Evidence panels | Add signal/FFT, processing info, feature table, feature importance, metrics, confusion matrix, limitations | All 12 dashboard feature groups are visible. |
| 5. Testing | Add unit, integration, negative, acceptance, and artifact compatibility tests | SRS acceptance criteria AC-001 through AC-021 are traceable to tests. |
| 6. Static-site decision | Align static demo with registry or explicitly separate it from supported product behavior | There is one documented source of truth for inference. |
| 7. Field-readiness boundary | Add data collection and validation plan for real fan motors | CWRU claims remain separate from field claims. |

# Acceptance Checklist for the Updated Project

- [ ] CWRU dataset loads from the manifest and raw files remain unchanged.
- [ ] CSV, NPY, and MAT inputs validate empty, short, malformed, NaN, infinity, and unsupported-extension cases.
- [ ] Sampling-rate assumptions are visible and stored with the model configuration.
- [ ] 2048-sample non-overlapping windows and discarded tails are reported.
- [ ] Exactly the approved 25 features, in saved order, are used at inference.
- [ ] Binary and multiclass tasks can be selected independently.
- [ ] Only available compatible models appear in the model selector.
- [ ] Window-level results and recording-level probability aggregation are testable.
- [ ] All supported class probabilities are displayed.
- [ ] Health Index is shown only with its non-RUL/non-safety explanation.
- [ ] Key features, feature importance, validation metrics, and confusion matrix are accessible.
- [ ] Data-quality warnings and CWRU/domain limitations are visible.
- [ ] Saved artifacts include feature order, label mapping, preprocessing, window, sampling, model, and validation metadata.
- [ ] Unit, integration, negative, and acceptance tests are automated.
- [ ] README and user/developer documentation describe the completed workflow.

# Final Recommendation

Treat the current project as a strong ML benchmark implementation with a basic diagnostic dashboard, not as a finished SRS-compliant product. Preserve the data and training pipeline, then build the missing contract and dashboard evidence layers around it. The first implementation milestone should be the model registry plus structured inference result because every major dashboard gap depends on knowing which task/model/configuration actually produced a result.

The SRS does not require a CNN, RUL estimation, anomaly detection, probability calibration, multi-sensor fusion, or field validation in the baseline build. Those should remain future work until the dashboard and classical-ML acceptance criteria are complete.

# Source Files Consulted

- Supplied SRS: `C:\Users\IDEAPAD\Downloads\IEEE_SRS_Bearing_Fault_Diagnosis_v2_0_UserFlow_Dashboard.docx`
- Repository guide: `docs/project_guide.md`
- ML guide: `docs/ML_guide.md`
- Dashboard: `app/dashboard.py`
- Inference: `src/inference.py`
- Feature extraction: `src/features.py`
- Training: `src/train.py`
- Evaluation: `src/evaluate.py`
- Static-site export and smoke check: `website/export_assets.py`, `website/check.mjs`
