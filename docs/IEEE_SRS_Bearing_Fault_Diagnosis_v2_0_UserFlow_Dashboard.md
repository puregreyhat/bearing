**SOFTWARE REQUIREMENTS SPECIFICATION**

**Machine Learning-Based Bearing Fault Diagnosis  
for Ceiling Fan Motors**

IEEE-Style SRS  
Version 2.0  
Primary Benchmark: CWRU Bearing Dataset

| **Item**          | **Value**                                              |
| ----------------- | ------------------------------------------------------ |
| Document Type     | Software Requirements Specification                    |
| System Type       | Machine Learning / Condition Monitoring                |
| Application       | Bearing condition and fault diagnosis                  |
| Benchmark Dataset | Case Western Reserve University (CWRU) Bearing Dataset |
| Version           | 2.0                                                    |
| Status            | Baseline requirements for implementation               |

# Document Control

| **Item**          | **Value**                              |
| ----------------- | -------------------------------------- |
| Document title    | SRS — ML-Based Bearing Fault Diagnosis |
| Version           | 2.0                                    |
| Prepared for      | Academic / Industry Project            |
| Primary benchmark | CWRU Bearing Dataset                   |
| Status            | Baseline                               |

## Revision History

2.0 Current Updated finalized user flow, model-selection interaction, and dashboard requirements.

| **Version** | **Date** | **Description**                 |
| ----------- | -------- | ------------------------------- |
| 1.0         | Current  | Initial IEEE-style SRS baseline |

# Table of Contents

1. 1\. Introduction
2. 2\. Overall Description
3. 3\. Specific Functional Requirements
4. 4\. Non-Functional Requirements
5. 5\. External Interface Requirements
6. 6\. Data Requirements
7. 7\. System Architecture and Data Flow
8. 8\. Machine Learning Requirements
9. 9\. Dashboard and User Interaction Requirements

**DF-012 — Data Quality and Limitations** — Communicate input/data-quality status, validation warnings, preprocessing assumptions, and the distinction between CWRU benchmark performance and field reliability.

**DF-011 — Model Performance** — Provide relevant validation information such as balanced accuracy, macro-F1, binary PR-AUC/ROC-AUC where applicable, validation scheme, and aggregate variability.

**DF-010 — Model Explanation / Feature Importance** — Provide feature-importance information for applicable models to help explain the diagnosis.

**DF-009 — Key Signal Features** — Present relevant extracted features from the approved 25-feature set in a readable form.

**DF-008 — Frequency Spectrum** — Display an FFT/frequency-spectrum visualization using the same frequency-domain processing assumptions as the ML pipeline.

**DF-007 — Signal Processing Information** — Show sampling rate, window size, number of valid windows analyzed, and signal duration where available.

**DF-006 — Input Signal Visualization** — Display the uploaded vibration waveform for visual inspection.

**DF-005 — Selected ML Model** — Show the selected SVM, Random Forest, or XGBoost model and prevent selection of unavailable/incompatible models.

**DF-004 — Selected Classification Task** — Show whether Binary or Multiclass Classification was selected.

**DF-003 — Health Index** — Display Health Index = 100 × predicted Healthy probability and clearly label it as a model-derived indicator, not RUL, exact lifetime, or a safety guarantee.

**DF-002 — Class Probabilities** — Display the probability for every supported class, not only the winning class.

**DF-001 — Prediction Summary** — Display Healthy/Faulty for binary classification and the detected fault category for multiclass classification.

**Dashboard Presentation Guidance** The 12 requirements should not be implemented as 12 equally prominent cards. The primary view should emphasize Prediction Summary, Class Probabilities, Health Index, Selected Task/Model, Input Signal, and Frequency Spectrum. Processing information, key features, feature importance, model performance, data quality, and limitations may be placed in secondary panels or expanders.

10. 10\. Testing and Verification Requirements
11. 11\. Acceptance Criteria

**AC-021 —** Dashboard clearly distinguishes the Health Index/model confidence from RUL or a physical safety guarantee.

**AC-020 —** Dashboard provides all 12 finalized feature groups: prediction summary, class probabilities, Health Index, selected task, selected model, input waveform, signal-processing information, frequency spectrum, key features, model explanation/feature importance, model performance, and data quality/limitations.

**AC-019 —** The selected model is actually used for inference and the selected task/model are shown in the results.

**AC-018 —** User can select an available compatible model for the selected classification task.

**AC-017 —** User can select Binary or Multiclass classification and the corresponding supported target classes are presented.

**AC-016 —** User can upload a supported .mat, .csv, or .npy vibration signal.

12. 12\. Constraints, Assumptions, and Limitations
13. 13\. Future Enhancements
14. 14\. Deliverables
15. Appendix A — Feature Definitions
16. Appendix B — Requirement Traceability Matrix
17. Appendix C — Glossary
18. References and Source Basis

# 1. Introduction

## 1.1 Purpose

This Software Requirements Specification defines the functional, non-functional, data, interface, machine-learning, testing, and deployment requirements for a vibration-based bearing fault diagnosis system. The system analyzes motor vibration signals, extracts signal characteristics, classifies bearing condition and fault category, and presents the diagnosis through a health-monitoring dashboard.

## 1.2 Scope

The system covers the workflow from vibration-signal input through preprocessing, fixed-length windowing, feature extraction, model training and evaluation, prediction, and dashboard presentation.

Vibration Signal → Preprocessing → Windowing → Feature Extraction → ML Classification → Diagnosis → Health Dashboard

## 1.3 Intended Users

- Maintenance or engineering users who need a simple bearing-condition diagnosis.
- ML/development users who prepare data, train models, evaluate performance, and maintain the system.
- Future industry users who provide vibration data from actual ceiling-fan motors.

## 1.4 Definitions and Acronyms

| **Term**     | **Meaning**                                                        |
| ------------ | ------------------------------------------------------------------ |
| CWRU         | Case Western Reserve University Bearing Dataset                    |
| ML           | Machine Learning                                                   |
| FFT          | Fast Fourier Transform                                             |
| SVM          | Support Vector Machine                                             |
| RF           | Random Forest                                                      |
| XGBoost      | Extreme Gradient Boosting                                          |
| CNN          | Convolutional Neural Network                                       |
| RMS          | Root Mean Square                                                   |
| RPM          | Revolutions Per Minute                                             |
| LOLO         | Leave-One-Load-Out validation                                      |
| Health Index | Model-derived healthy-class probability expressed on a 0–100 scale |

# 2. Overall Description

## 2.1 Product Perspective

The system is a modular ML diagnostic application. Raw recordings, signal processing, feature generation, model training, evaluation, inference, and the user interface shall remain separable modules.

## 2.2 Major System Components

| **Component**        | **Responsibility**                                                    |
| -------------------- | --------------------------------------------------------------------- |
| Dataset Management   | Load and organize CWRU recordings and metadata.                       |
| Signal Preprocessing | Sampling-rate normalization and input validation.                     |
| Signal Segmentation  | Divide recordings into fixed-length windows.                          |
| Feature Extraction   | Compute time- and frequency-domain vibration features.                |
| Dataset Validation   | Check labels, missing values, numeric types, and leakage constraints. |
| Model Training       | Train SVM, Random Forest, and XGBoost baselines.                      |
| Model Evaluation     | Perform grouped validation and calculate metrics.                     |
| Model Selection      | Select the deployment model.                                          |
| Inference            | Apply saved preprocessing and model to new signals.                   |
| Dashboard            | Display condition, fault type, probabilities, and health information. |

## 2.3 Product Functions

**Finalized User Flow: Upload Signal → Select Classification Task → Select Compatible Model → Predict → View Results.**

The user-facing workflow shall guide the user sequentially from supported signal upload through task selection, compatible model selection, prediction, and result interpretation.

- Load and inspect the CWRU benchmark dataset.
- Preprocess vibration signals.
- Segment recordings into fixed-length windows.
- Extract time- and frequency-domain features.
- Build Healthy/Faulty and multiclass targets.
- Train SVM, Random Forest, and XGBoost models.
- Perform leakage-safe evaluation.
- Compare model performance.
- Save the selected model and feature/preprocessing configuration.
- Accept new vibration signals and display diagnosis.

# 3. Specific Functional Requirements

## 3.1 Dataset and Input Requirements

**FR-001 — Dataset Input**

The system shall support loading the CWRU bearing dataset for model development.

**FR-002 — Dataset Inspection**

The system shall allow inspection of recordings, sampling rates, operating/load conditions, bearing condition, fault type, channels, and signal length.

**FR-003 — Signal Loading**

The system shall load numerical vibration signals from supported .mat, .csv, and .npy inputs.

**FR-004 — Signal Validation**

The system shall detect empty signals, insufficient samples, non-numeric values, missing values, and infinite values.

## 3.2 Signal Preprocessing Requirements

**FR-005 — Sampling-Rate Handling**

The system shall identify the signal sampling rate and apply configured preprocessing when resampling is required.

**FR-006 — CWRU Normal Downsampling**

For the current CWRU pipeline, 48 kHz normal recordings shall be converted to 12 kHz using the defined zero-phase FIR decimation by four, while raw files remain untouched.

**FR-007 — Preprocessing Consistency**

Training and inference shall use the same preprocessing assumptions.

## 3.3 Signal Segmentation Requirements

**FR-008 — Fixed-Length Windowing**

The system shall divide signals into fixed-length windows; the primary training configuration shall use 2048 samples.

**FR-009 — Non-Overlapping Windows**

The primary training pipeline shall use non-overlapping windows.

**FR-010 — Incomplete Tail Handling**

Incomplete final windows shall be discarded unless explicit padding is configured.

## 3.4 Feature Extraction Requirements

**FR-011 — Time-Domain Features**

The system shall extract RMS, standard deviation, peak, peak-to-peak, mean, Pearson kurtosis, skewness, crest factor, shape factor, impulse factor, and margin factor.

**FR-012 — Frequency-Domain Features**

The system shall extract FFT peak frequency/amplitude, spectral centroid, spectral spread, spectral entropy, spectral kurtosis-like measure, and eight band-energy values.

**FR-013 — Feature Dataset**

Each complete signal window shall produce one structured feature row with traceability metadata.

**FR-014 — Feature Boundary**

Recording IDs, fault diameter, RPM, load, window size, sampling rate, window ID, bearing condition, and fault type shall not be model features when they are metadata or labels.

## 3.5 Classification Requirements

**FR-015 — Binary Classification**

The system shall support Healthy versus Faulty classification.

**FR-016 — Multiclass Classification**

The current CWRU implementation shall support Normal, Inner Race, Outer Race, and Ball.

**FR-017 — Fault-Diameter Interpretation**

The multiclass target shall represent fault category, not fault diameter.

**FR-018 — Label Encoding**

Class-name to encoded-value mappings shall be stored with the model artifact.

## 3.6 Machine-Learning Model Requirements

User-Facing Model Selection: SVM, Random Forest, and XGBoost are candidate models. The dashboard shall expose only models that are actually trained, saved, and compatible with the selected classification task. If only the final deployment Random Forest artifact is available, the UI shall expose only Random Forest until the other task-specific artifacts are intentionally retained and made available.

**FR-019 — SVM Baseline**

The system shall provide an SVM baseline with appropriate scaling and class balancing.

**FR-020 — Random Forest Baseline**

The system shall provide a Random Forest baseline with class balancing and configurable tree-depth/minimum-leaf hyperparameters.

**FR-021 — XGBoost Baseline**

The system shall provide an XGBoost baseline with configurable tree depth and learning rate and appropriate weighting.

**FR-022 — Optional 1D-CNN**

The system may provide a 1D-CNN operating directly on vibration windows.

**FR-023 — Reproducibility**

Training and evaluation shall use fixed random seeds where supported.

## 3.7 Validation and Evaluation Requirements

**FR-024 — Leakage-Safe Splitting**

Windows from the same source recording shall not be placed in both training and test sets within an outer evaluation split.

**FR-025 — Grouped Validation**

The evaluation pipeline shall support grouped validation by File_ID, including StratifiedGroupKFold and Leave-One-Load-Out.

**FR-026 — Nested Tuning**

Hyperparameter tuning shall occur inside the outer training split.

**FR-027 — Metrics**

The system shall calculate balanced accuracy, macro F1, accuracy, per-class precision/recall, and binary PR-AUC/ROC-AUC where applicable.

**FR-028 — Confusion Matrix**

The system shall generate confusion matrices.

**FR-029 — Inference Time**

The system shall record inference time.

**FR-030 — Model Comparison**

The system shall compare candidate models using defined metrics and validation schemes.

**FR-031 — Deployment Selection**

The deployment model shall be selected using the defined LOLO macro-F1 criterion and then refit on the complete feature dataset.

## 3.8 Inference Requirements

**FR-032 — New Signal Inference**

The system shall accept unseen vibration signals and apply the same preprocessing and feature pipeline.

**FR-033 — Window-Level Prediction**

The system shall generate predictions for each valid window.

**FR-034 — Recording-Level Aggregation**

The system shall aggregate window-level class probabilities for an overall recording-level result.

**FR-035 — Prediction Probability**

The system shall display probabilities for supported classes.

**FR-036 — Bearing Condition**

The system shall report Healthy or Faulty.

**FR-037 — Fault Type**

The multiclass system shall report the predicted fault category.

## 3.9 Health Index Requirements

**FR-038 — Health Index**

The system shall calculate Health Index = 100 × predicted Healthy probability.

**FR-039 — Health Interpretation**

The system shall state that the health index is not RUL, exact lifetime, or a maintenance safety guarantee.

## 3.10 Dashboard Requirements

**FR-040 — Dashboard**

The system shall provide a user-facing dashboard for vibration diagnosis.

**FR-041 — File Upload**

The dashboard shall allow upload of a supported vibration signal.

**FR-042 — Prediction Display**

The dashboard shall display condition, fault type where applicable, class probabilities, confidence, and health index.

**FR-043 — Error Display**

The dashboard shall show understandable messages for invalid files, non-numeric signals, insufficient samples, invalid values, and unavailable models.

**FR-044 — Feature Importance**

The system shall provide feature-importance information for applicable models.

## 3.11 Model Artifact Requirements

**FR-045 — Model Saving**

The system shall save trained models for reuse without retraining.

**FR-046 — Configuration Saving**

The saved model/configuration shall preserve feature names/order, label mapping, preprocessing assumptions, window size, and sampling-rate requirements.

# 4. Non-Functional Requirements

| **ID**  | **Requirement**     | **Description**                                                                                                  |
| ------- | ------------------- | ---------------------------------------------------------------------------------------------------------------- |
| NFR-001 | Usability           | A non-ML user should be able to upload a signal, start analysis, and interpret the result without code.          |
| NFR-002 | Reliability         | Invalid inputs shall result in clear errors instead of application crashes.                                      |
| NFR-003 | Performance         | Inference shall complete within a reasonable time for normal-sized signals and inference time shall be measured. |
| NFR-004 | Maintainability     | Major pipeline stages shall be separated into maintainable modules.                                              |
| NFR-005 | Scalability         | The design should allow new recordings, conditions, models, and company/fan datasets.                            |
| NFR-006 | Data Integrity      | Raw CWRU recordings shall remain unmodified.                                                                     |
| NFR-007 | Reproducibility     | Seeds, hyperparameters, model configuration, and evaluation results shall be recorded.                           |
| NFR-008 | Safe Interpretation | Probabilities and health index shall not be presented as guaranteed physical safety.                             |
| NFR-009 | Security            | Uploaded files shall be treated as data and not executed as code.                                                |

# 5. External Interface Requirements

## 5.1 User Interface

- Input-file selection/upload area.
- Input guidance and validation messages.
- Bearing condition result.
- Fault type result.
- Class probabilities.
- Health index.
- Number of windows analyzed.
- Clear error messages.

## 5.2 Software Interfaces

| **Interface**        | **Purpose**                                         |
| -------------------- | --------------------------------------------------- |
| NumPy / Pandas       | Numeric arrays and structured data handling.        |
| SciPy                | Signal preprocessing and FFT/signal processing.     |
| scikit-learn         | SVM, Random Forest, validation, metrics, pipelines. |
| XGBoost              | Gradient-boosted-tree classification.               |
| PyTorch / TensorFlow | Optional 1D-CNN.                                    |
| Streamlit            | Proposed web dashboard.                             |
| Joblib / Pickle      | Proposed model serialization.                       |

## 5.3 File Interfaces

| **Format**          | **Use**                                                            |
| ------------------- | ------------------------------------------------------------------ |
| .mat                | CWRU/source vibration recordings and compatible inputs.            |
| .csv                | Feature tables and numeric vibration inputs.                       |
| .npy                | Numeric vibration arrays.                                          |
| Model artifact      | Saved trained model/configuration.                                 |
| Manifest / metadata | Traceability between recordings, labels, and operating conditions. |

# 6. Data Requirements

## 6.1 Current CWRU Data Interpretation

The current project uses CWRU as the benchmark. The multiclass label represents fault category, not fault diameter. The current categories are Normal, Inner Race, Outer Race, and Ball. Fault diameter and load remain metadata.

| **Data Item**         | **Current Interpretation**                                  |
| --------------------- | ----------------------------------------------------------- |
| Bearing condition     | Healthy or Faulty                                           |
| Multiclass fault type | Normal, Inner Race, Outer Race, Ball                        |
| Fault diameter        | Metadata; not the multiclass target                         |
| Load                  | Operating-condition metadata; 0–3 HP in the benchmark setup |
| Channel               | Drive-End vibration signal (\_DE_time)                      |
| Primary window        | 2048 samples                                                |
| Training rate         | 12 kHz                                                      |

## 6.2 Data Quality Rules

- Every model row shall have a valid source-recording identifier.
- Model features shall be numeric.
- Missing and infinite values shall be rejected or explicitly handled.
- Labels shall agree with the source manifest.
- Window IDs shall be unique.
- Train/test recording groups shall be disjoint within each outer split.

# 7. System Architecture and Data Flow

## 7.1 High-Level Architecture

User-facing flow: Upload Signal → Select Classification Task → Select Compatible Model → Predict → View Results.

CWRU / Field Signal → Data Loader → Preprocessing → 2048-Sample Windows → 25-Feature Extraction → Leakage-Safe Training/Evaluation → SVM / RF / XGBoost → Model Selection → Saved Model → Inference → Health Dashboard

## 7.2 Training Flow

19. Load source recording.
20. Select Drive-End vibration signal.
21. Apply sampling-rate preprocessing where required.
22. Generate non-overlapping windows.
23. Extract approved features.
24. Validate feature rows and metadata.
25. Create binary and/or multiclass targets.
26. Perform grouped outer evaluation and inner tuning.
27. Aggregate metrics.
28. Select deployment model.
29. Refit selected configuration and save artifact.

## 7.3 Inference Flow

1\. Upload Signal — User uploads a supported .mat, .csv, or .npy vibration signal.

2\. Select Classification Task — Binary Classification (Healthy/Faulty) or Multiclass Classification (Normal/Inner Race/Outer Race/Ball).

3\. Select Compatible Model — User selects SVM, Random Forest, or XGBoost only from trained/available models for the selected task.

4\. Predict — Validate input, apply preprocessing, create non-overlapping 2048-sample windows, extract the approved 25 features, run the selected model, and aggregate window-level probabilities.

5\. View Results — Display diagnosis, probabilities, Health Index, task/model context, signal evidence, processing information, feature information, model explanation, model performance, and data-quality/limitation information.

30. User uploads vibration signal.
31. Validate numeric content and minimum length.
32. Normalize sampling rate if required.
33. Create complete windows.
34. Extract exact saved feature names/order.
35. Run saved classifier.
36. Aggregate window-level probabilities.
37. Display condition, fault type, probabilities, and health index.

# 8. Machine Learning Requirements

## 8.1 Feature Vector

| **#** | **Feature**        |
| ----- | ------------------ |
| 1     | RMS                |
| 2     | Std                |
| 3     | Peak               |
| 4     | Kurtosis           |
| 5     | Crest_Factor       |
| 6     | FFT_Peak_Frequency |
| 7     | Peak_to_Peak       |
| 8     | Mean               |
| 9     | Skewness           |
| 10    | Shape_Factor       |
| 11    | Impulse_Factor     |
| 12    | Margin_Factor      |
| 13    | FFT_Peak_Amplitude |
| 14    | Spectral_Centroid  |
| 15    | Spectral_Spread    |
| 16    | Spectral_Entropy   |
| 17    | Spectral_Kurtosis  |
| 18    | Band_Energy_1      |
| 19    | Band_Energy_2      |
| 20    | Band_Energy_3      |
| 21    | Band_Energy_4      |
| 22    | Band_Energy_5      |
| 23    | Band_Energy_6      |
| 24    | Band_Energy_7      |
| 25    | Band_Energy_8      |

## 8.2 Candidate Models

| **Model**     | **Role**           | **Key Configuration**                                                         |
| ------------- | ------------------ | ----------------------------------------------------------------------------- |
| SVM           | Classical baseline | Scaling + SVC + class balancing; tune C and gamma.                            |
| Random Forest | Tree ensemble      | 250 trees; balanced classes; tune max_depth and min_samples_leaf.             |
| XGBoost       | Boosting baseline  | 200 trees; subsample 0.9; colsample_bytree 0.9; tune depth and learning rate. |
| 1D-CNN        | Optional           | Direct operation on vibration windows.                                        |

## 8.3 Validation

Grouped validation is required because windows from the same recording are correlated. The outer evaluation supports StratifiedGroupKFold and Leave-One-Load-Out. Inner tuning occurs only within the outer training split.

# 9. Dashboard and User Interaction Requirements

| **UI Element** | **Required Content**                               |
| -------------- | -------------------------------------------------- |
| Input area     | Upload supported vibration signal.                 |
| Input guidance | Supported format, sampling rate, minimum samples.  |
| Condition      | HEALTHY or FAULTY.                                 |
| Fault type     | NORMAL, INNER RACE, OUTER RACE, or BALL.           |
| Probabilities  | Probability for each class.                        |
| Health index   | 0–100 model-derived healthy probability indicator. |
| Window count   | Number of valid windows analyzed.                  |
| Error area     | Human-readable validation/inference errors.        |

## 9.1 Example Output

BEARING CONDITION: FAULTY  
FAULT TYPE: INNER RACE  
CONFIDENCE: 98.4%  
NORMAL: 0.0%  
INNER RACE: 98.4%  
OUTER RACE: 0.4%  
BALL: 1.2%  
HEALTH INDEX: 0

# 10. Testing and Verification Requirements

## 10.1 Unit Testing

- Signal loading
- Sampling-rate conversion
- Window generation
- RMS/statistical features
- FFT features
- Band-energy calculation
- Label encoding/decoding
- Model loading
- Prediction probability mapping

## 10.2 Integration Testing

- Signal → preprocessing → windowing → feature extraction
- Feature matrix → trained model → prediction
- Uploaded file → dashboard → final diagnosis

## 10.3 Negative Testing

- Empty file
- Non-numeric values
- NaN/Inf values
- Signal shorter than 2048 samples
- Unsupported extension
- Missing model artifact
- Feature-name mismatch

# 11. Acceptance Criteria

| **ID** | **Acceptance Criterion**                                                   |
| ------ | -------------------------------------------------------------------------- |
| AC-001 | CWRU dataset loads successfully.                                           |
| AC-002 | Valid fixed-length windows are produced.                                   |
| AC-003 | Configured time/frequency features are extracted.                          |
| AC-004 | Feature rows are traceable and valid.                                      |
| AC-005 | SVM, Random Forest, and XGBoost pipelines train successfully.              |
| AC-006 | Outer evaluation keeps source recordings/groups disjoint.                  |
| AC-007 | Required classification metrics are produced.                              |
| AC-008 | Confusion matrices are generated.                                          |
| AC-009 | Binary prediction returns Healthy/Faulty.                                  |
| AC-010 | Multiclass prediction returns a supported fault category.                  |
| AC-011 | Dashboard accepts a valid signal.                                          |
| AC-012 | Dashboard displays condition, fault type, probabilities, and health index. |
| AC-013 | Saved model can be loaded for inference.                                   |
| AC-014 | Invalid inputs generate clear errors.                                      |
| AC-015 | Health index is not presented as RUL or a safety guarantee.                |

# 12. Constraints, Assumptions, and Limitations

## 12.1 Constraints

- CWRU is laboratory-collected benchmark data.
- The current input contract requires consistent preprocessing.
- Primary training window is 2048 samples.
- The initial system is a classifier, not a prognostics/RUL system.

## 12.2 Assumptions

- The signal belongs to a compatible motor/bearing system.
- At least one complete window can be formed.
- Training and inference use the same feature definitions and preprocessing.

## 12.3 Limitations

High benchmark performance on CWRU does not guarantee field performance on ceiling-fan motors. The intended progression is to establish the methodology on CWRU and validate it later on naturally collected fan-motor data. Domain shift, sensor mounting, operating conditions, and naturally evolving faults may affect performance.

# 13. Future Enhancements

| **ID** | **Enhancement**                | **Description**                                                        |
| ------ | ------------------------------ | ---------------------------------------------------------------------- |
| FE-001 | Naturally degraded fan dataset | Collect healthy and naturally degraded ceiling-fan bearing data.       |
| FE-002 | Multi-sensor fusion            | Combine vibration with temperature, current, RPM, and operating state. |
| FE-003 | Probability calibration        | Calibrate model probabilities before presenting confidence.            |
| FE-004 | Unknown/anomaly detection      | Handle signals outside known training classes.                         |
| FE-005 | Envelope-spectrum features     | Add bearing-frequency/envelope features where justified.               |
| FE-006 | Transfer learning              | Adapt CWRU representations to actual fan-motor data.                   |
| FE-007 | 1D-CNN comparison              | Compare raw-signal deep learning with engineered features.             |
| FE-008 | Prognostics                    | Develop RUL only if suitable run-to-failure data becomes available.    |

# 14. Deliverables

38. SRS document
39. CWRU benchmark preparation and feature dataset
40. Preprocessing and feature-extraction modules
41. SVM, Random Forest, and XGBoost training pipelines
42. Leakage-safe evaluation scripts and reports
43. Saved model artifacts
44. Inference module
45. Health-monitoring dashboard
46. Test cases and results
47. Final report and user documentation

# Appendix A — Feature Definitions

| **Feature**        | **Definition**                                                   |
| ------------------ | ---------------------------------------------------------------- |
| RMS                | Root-mean-square magnitude of the window.                        |
| Std                | Population standard deviation.                                   |
| Peak               | Maximum absolute signal magnitude.                               |
| Kurtosis           | Pearson kurtosis; high values indicate impulsive behavior.       |
| Crest Factor       | Peak divided by RMS.                                             |
| FFT Peak Frequency | Frequency of the largest non-DC FFT amplitude.                   |
| Peak-to-Peak       | Maximum minus minimum signal value.                              |
| Mean               | Arithmetic mean.                                                 |
| Skewness           | Third standardized central moment.                               |
| Shape Factor       | RMS divided by mean absolute value.                              |
| Impulse Factor     | Peak divided by mean absolute value.                             |
| Margin Factor      | Peak divided by squared mean square-root absolute amplitude.     |
| FFT Peak Amplitude | Largest non-DC FFT amplitude.                                    |
| Spectral Centroid  | Power-weighted center of the spectrum.                           |
| Spectral Spread    | Power-weighted spread around the centroid.                       |
| Spectral Entropy   | Entropy of normalized spectral power.                            |
| Spectral Kurtosis  | Pearson kurtosis-like measure of spectral power.                 |
| Band Energy 1–8    | Summed FFT power in eight approximately equal frequency regions. |

# Appendix B — Requirement Traceability Matrix

| **Requirement Area** | **IDs**       | **Verification Focus**                    |
| -------------------- | ------------- | ----------------------------------------- |
| Dataset/Input        | FR-001–FR-004 | Dataset availability and input validation |
| Preprocessing        | FR-005–FR-007 | Sampling rate and consistency             |
| Segmentation         | FR-008–FR-010 | Fixed non-overlapping windows             |
| Features             | FR-011–FR-014 | Feature dataset correctness               |
| Classification       | FR-015–FR-018 | Binary/multiclass targets                 |
| Models               | FR-019–FR-023 | SVM/RF/XGBoost/optional CNN               |
| Evaluation           | FR-024–FR-031 | Leakage-safe assessment                   |
| Inference            | FR-032–FR-037 | New-signal prediction                     |
| Health Index         | FR-038–FR-039 | Correct interpretation                    |
| Dashboard            | FR-040–FR-044 | User-facing diagnosis                     |
| Artifacts            | FR-045–FR-046 | Reusable model/configuration              |

# Appendix C — Glossary

| **Term**          | **Definition**                                                              |
| ----------------- | --------------------------------------------------------------------------- |
| Bearing           | Machine component that supports a rotating shaft and reduces friction.      |
| Vibration Signal  | Numeric time-series measurement representing mechanical vibration.          |
| Window            | Fixed-length portion of a longer vibration recording used as one ML sample. |
| Fault Category    | Physical bearing-fault class such as Inner Race, Outer Race, or Ball.       |
| Fault Diameter    | Physical size of a seeded benchmark fault; not the multiclass target.       |
| Load              | Operating load under which a recording was collected.                       |
| Drive-End (DE)    | Motor shaft/bearing side used as the selected vibration channel.            |
| Data Leakage      | Improper transfer of evaluation information into training.                  |
| Balanced Accuracy | Average recall across classes.                                              |
| Macro F1          | Average F1 score across classes with equal class weighting.                 |
| Health Index      | 100 times predicted Healthy probability; not RUL.                           |

# References and Source Basis

1\. Uploaded project requirements / ML development documentation supplied for this project. This SRS preserves the supplied project's scope, CWRU benchmark basis, signal-processing workflow, classical ML baselines, leakage-safe evaluation, dashboard requirement, and future industry-validation direction.

2\. Case Western Reserve University Bearing Data Center. Used as the benchmark dataset source; project-specific preprocessing, labels, and model requirements are governed by the supplied project documentation.
