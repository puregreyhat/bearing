# Machine-learning guide

This document explains how the bearing-diagnosis models in this repository are
created, evaluated, saved, and used for inference. It describes the code as it
exists today; it is not a generic machine-learning tutorial.

## 1. What the system predicts

The project uses drive-end vibration recordings from the Case Western Reserve
University (CWRU) Bearing Data Center. It trains two related classifiers:

1. **Binary diagnosis** (`binary`): `Healthy` versus `Faulty`.
2. **Multiclass diagnosis** (`multiclass`): `Normal`, `Ball`, `Inner_Race`, or
   `Outer_Race`.

The multiclass label describes the fault category, not the fault diameter.
The dataset contains three fault diameters (`0.007`, `0.014`, and `0.021`
inches), four loads (`0` through `3` HP), four healthy recordings, and 36
fault recordings. The selected channel is the drive-end (`*_DE_time`) signal.
The source and label mapping are recorded in
[`docs/data_manifest.csv`](./docs/data_manifest.csv).

The models are intended as a CWRU laboratory benchmark. The dashboard itself
warns that the models have not been validated on ceiling-fan motors or other
field equipment.

## 2. End-to-end pipeline

The pipeline is deliberately split into reproducible stages:

```text
CWRU .mat files
    |
    v
download.py -> data/raw/*.mat + docs/data_manifest.csv
    |
    v
extract.py -> data/processed/cwru_features.csv
    |
    v
validate.py -> dataset validation report and signal plots
    |
    v
train.py -> grouped CV results + feature importance + joblib models
    |
    v
evaluate.py -> metric summary, confusion matrices, leakage audit
    |
    v
inference.py / app/dashboard.py -> predictions for uploaded signals
```

The normal recording files are originally sampled at 48 kHz. They are
downsampled by four using a zero-phase FIR filter, so every extracted feature
row is ultimately based on a 12 kHz signal. Fault recordings are already at
12 kHz. Raw files are not modified.

## 3. Reproducing the dataset

Create an environment and install the dependencies:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

On Windows, the equivalent commands are:

```powershell
py -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

Download and inspect the recordings:

```bash
.venv/bin/python src/download.py
```

`download.py` downloads the selected `.mat` files from the CWRU URLs, reads
each file to identify its drive-end channel and RPM, and writes the manifest.
It does not download a random subset: the recording registry is explicitly
defined in `src/download.py`.

## 4. Windowing and feature extraction

Run:

```bash
.venv/bin/python src/extract.py
```

The extractor builds both supported window sizes:

- `data/processed/cwru_features.csv`: 2,048 samples per window
- `data/processed/cwru_features_4096.csv`: 4,096 samples per window

Training currently uses the 2,048-sample file. A window is approximately
170.7 ms at 12 kHz. Windows are non-overlapping; an incomplete final tail is
discarded. Each row receives a `Window_ID` containing the recording ID, window
size, and zero-based window number.

For a signal window `x`, `src/features.py` calculates:

### Time-domain features

- `RMS`: square root of mean squared amplitude
- `Std`, `Mean`, and `Peak_to_Peak`
- `Peak`: maximum absolute amplitude
- `Kurtosis`: Pearson kurtosis (`fisher=False`)
- `Skewness`
- `Crest_Factor = Peak / RMS`
- `Shape_Factor = RMS / Mean(|x|)`
- `Impulse_Factor = Peak / Mean(|x|)`
- `Margin_Factor = Peak / (Mean(sqrt(|x|))^2)`

Zero denominators are guarded and produce zero for the corresponding ratio.

### Frequency-domain features

The window is multiplied by a Hann window and transformed using a real FFT.
The single-sided spectrum is scaled by:

```text
abs(rfft(x * Hann)) * 2 / sum(Hann)
```

The DC bin is set to zero before spectral calculations. The extractor then
calculates:

- `FFT_Peak_Frequency`
- `FFT_Peak_Amplitude`
- `Spectral_Centroid`
- `Spectral_Spread`
- `Spectral_Entropy`
- `Spectral_Kurtosis`
- `Band_Energy_1` through `Band_Energy_8`

The power spectrum is divided into eight equal-length FFT-bin groups for the
band energies. Spectral entropy uses the non-zero normalized power bins. Any
NaN or infinity causes feature extraction to fail explicitly.

## 5. The model feature boundary

The training script creates its feature list by excluding identifiers, labels,
and acquisition metadata. The excluded columns are:

```text
File_ID, Fault_Diameter, RPM, Load, Window_Size, Sampling_Rate,
Window_ID, Bearing_Condition, Fault_Type
```

This leaves exactly 25 signal features. The authoritative generated list is
[`results/feature_list.json`](./results/feature_list.json):

```text
RMS, Std, Peak, Kurtosis, Crest_Factor, FFT_Peak_Frequency,
Peak_to_Peak, Mean, Skewness, Shape_Factor, Impulse_Factor,
Margin_Factor, FFT_Peak_Amplitude, Spectral_Centroid,
Spectral_Spread, Spectral_Entropy, Spectral_Kurtosis,
Band_Energy_1 ... Band_Energy_8
```

This boundary is important. Including `Load`, `RPM`, `Fault_Diameter`, or a
recording identifier would allow a model to learn experimental metadata or
recording-specific signatures instead of vibration behavior. The training code
asserts that excluded columns do not enter the feature matrix.

## 6. The three candidate model families

`src/train.py:model_specs` constructs one pipeline for each candidate:

### SVM

```text
StandardScaler -> SVC(class_weight="balanced", probability=True)
```

The grid searches:

- `C`: `1`, `10`
- `gamma`: `"scale"`, `0.01`

Scaling is inside the scikit-learn pipeline, so it is fitted only on the
training portion of each inner fold.

### Random forest

```text
RandomForestClassifier(
    n_estimators=250,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
```

The grid searches:

- `max_depth`: `None`, `12`
- `min_samples_leaf`: `1`, `3`

### XGBoost

```text
XGBClassifier(
    n_estimators=200,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42,
    n_jobs=4
)
```

The objective is `binary:logistic` for the binary task and `multi:softprob`
for the multiclass task. The grid searches:

- `max_depth`: `3`, `5`
- `learning_rate`: `0.05`, `0.1`

XGBoost receives balanced per-sample weights. SVM and random forest use
`class_weight="balanced"`. The project does not use SMOTE or other synthetic
oversampling.

## 7. Tasks and label encoding

For each task, `LabelEncoder` converts the string labels into the integer
classes expected by the estimators. The fitted class order is stored in the
model artifact, so probabilities can be mapped back to the correct label.

The same feature matrix is used for both tasks:

```python
tasks = {
    "binary": df["Bearing_Condition"],
    "multiclass": df["Fault_Type"],
}
```

The binary metrics treat `Healthy` as the positive class for PR-AUC and
ROC-AUC.

## 8. Leakage-safe evaluation

Run:

```bash
.venv/bin/python src/validate.py
.venv/bin/python src/train.py
.venv/bin/python src/evaluate.py
```

### Dataset validation

`validate.py` checks:

- unique `Window_ID`s
- no missing or infinite values
- numeric feature columns
- agreement between every row and its manifest label/load
- class balance and per-condition sanity statistics

The current 2,048-sample dataset contains 2,331 windows: 709 Inner Race,
708 Ball, 708 Outer Race, and 206 Normal. The smaller Normal class is why
balanced training and balanced metrics are used.

### Outer evaluation schemes

For each task and each model, training evaluates both:

1. **Stratified grouped 4-fold CV**: `StratifiedGroupKFold` keeps all windows
   from a `File_ID` in one fold while approximately preserving class balance.
2. **Leave-One-Load-Out (LOLO)**: `LeaveOneGroupOut` holds out one complete
   load at a time. All windows from that load are test data.

The grouping is essential because adjacent windows from one recording are
correlated. Randomly splitting windows would put near-duplicate recording
content in both training and test sets.

There are four outer folds in each scheme. Each outer training split runs a
three-fold `StratifiedGroupKFold` grid search. Thus, hyperparameters are chosen
without seeing the outer test fold. The assertions in `train.py` verify that
training and test `File_ID` sets are disjoint and that every test fold contains
all task classes.

The fixed random seed is `42`. Feature extraction is deterministic, and the
later stochastic estimators and splitters use the same seed.

## 9. Metrics and generated evaluation artifacts

For every outer fold, the trainer records:

- balanced accuracy
- macro F1
- ordinary accuracy (recorded but intentionally not the headline metric)
- per-class precision and recall
- binary Healthy PR-AUC and ROC-AUC
- training time
- inference milliseconds per window
- selected hyperparameters

`evaluate.py` aggregates these into mean and standard deviation, writes
confusion matrices, computes feature importance, and creates the leakage audit.
Important outputs are:

- [`results/fold_metrics.csv`](./results/fold_metrics.csv)
- [`results/fold_predictions.csv`](./results/fold_predictions.csv)
- [`results/metrics_summary.csv`](./results/metrics_summary.csv)
- [`results/model_evaluation.md`](./results/model_evaluation.md)
- [`results/leakage_audit.md`](./results/leakage_audit.md)
- [`results/feature_importance_folds.csv`](./results/feature_importance_folds.csv)
- [`results/feature_importance_top.csv`](./results/feature_importance_top.csv)
- `results/confusion_<task>_<scheme>_<model>.png`

Balanced accuracy and macro F1 are emphasized because Normal has substantially
fewer windows than the three fault categories. Feature importance is reported
for random forest and XGBoost using both native importance and out-of-fold
permutation importance.

## 10. Final deployment model

After all outer folds finish, `train.py` selects one model per task using the
highest mean LOLO macro F1. It takes the modal (most common) LOLO
hyperparameter setting for that winning model, then refits it on **all**
2,331 feature rows.

The current saved benchmark results select random forest for both tasks:

- binary random forest: LOLO macro F1 `1.0000 ± 0.0000`
- multiclass random forest: LOLO macro F1 `0.9810 ± 0.0361`

The complete comparison, including SVM and XGBoost, is in
[`results/model_evaluation.md`](./results/model_evaluation.md). Very high
scores should be interpreted as CWRU benchmark separability, not as proof of
field reliability.

The final artifacts are:

- [`models/binary_model.joblib`](./models/binary_model.joblib)
- [`models/multiclass_model.joblib`](./models/multiclass_model.joblib)

Each joblib file contains the fitted pipeline, feature names, class names,
task name, selection rule, and cross-validation macro-F1 values.

## 11. Runtime inference

The dashboard and smoke checks use
[`src/inference.py`](./src/inference.py):

1. Read a `.csv`, `.npy`, or `.mat` signal.
2. Convert it to a finite one-dimensional numeric array.
3. If the input is 48 kHz, apply the same factor-four zero-phase FIR
   decimation used during training.
4. Require at least 2,048 samples at 12 kHz.
5. Trim the incomplete tail and reshape into non-overlapping 2,048-sample
   windows.
6. Extract the exact feature names stored in the binary model artifact.
7. Call `predict_proba` on every window.
8. Average class probabilities across windows for each task.

The dashboard displays the most probable binary condition, the most probable
fault type, class probabilities, and a health index:

```text
Health Index = 100 * probability(Healthy)
```

This is a model confidence score, not remaining useful life. A long recording
can produce multiple predictions, but averaging them does not turn the model
into a temporal degradation or prognostics model.

Start the dashboard with:

```bash
.venv/bin/streamlit run app/dashboard.py
```

The example files in [`demo_csv_files/`](./demo_csv_files/) are 4,096-sample,
12 kHz signals and therefore produce two analysis windows. They are examples
from the benchmark, not independent validation data.

## 12. Practical limitations and safe interpretation

- **Domain shift:** training and evaluation use CWRU laboratory recordings.
  Motor construction, sensors, mounting, speed, noise, and load can differ
  substantially in deployment.
- **Recording-level sample size:** thousands of windows come from 40 source
  recordings. Window count must not be confused with independent-machine
  count.
- **Class imbalance:** Normal has 206 windows versus roughly 708–709 per fault
  category. Use balanced metrics when comparing future experiments.
- **No temporal model:** windows are classified independently and then
  averaged. There is no sequence model, maintenance history, or RUL target.
- **Input contract:** inference expects a numeric one-dimensional signal and a
  declared sampling rate of 12 kHz or 48 kHz.
- **Training/evaluation consistency:** any new preprocessing, window size, or
  feature must be applied identically during training and inference, and the
  resulting model's feature list must be stored with the artifact.

For a new experiment, preserve the same grouped evaluation discipline: split
by recording or machine before fitting preprocessing or tuning, keep metadata
out of the feature matrix unless it is intentionally part of the prediction
problem, and report results by held-out operating condition.
