# Complete Project Guide and Examiner/Viva Handbook

## CWRU Bearing Fault Diagnosis: Dataset → Features → Machine Learning → Dashboard

This guide explains the project exactly as implemented. It is intended for project demonstrations, reports, viva examinations, interviews, and handover to another developer. It distinguishes what the system actually proves from what it does not prove.

---

## 1. The 30-second explanation

This project diagnoses bearing condition from vibration signals. It downloads real Case Western Reserve University (CWRU) bearing recordings, keeps the raw MATLAB files untouched, converts each Drive-End vibration signal into non-overlapping windows, extracts 25 time- and frequency-domain features, and trains three classical machine-learning models. It evaluates both Healthy-vs-Faulty and four-class fault-type classification using recording-grouped and Leave-One-Load-Out splits so windows from the same recording cannot leak into both training and testing. The best models are saved and served through a Streamlit dashboard that accepts `.mat`, `.csv`, or `.npy` vibration signals.

## 2. The one-minute viva answer

> I used the real CWRU bearing dataset because it is a standard public vibration benchmark with documented bearing faults, motor loads, speeds, and sensor channels. I selected 40 recordings: four normal baselines and 36 Drive-End fault recordings covering inner-race, ball, and outer-race faults at three fault diameters and four motor loads. Normal files were recorded at 48 kHz, so I converted them to 12 kHz using an anti-aliasing FIR decimator before combining them with the 12 kHz fault data. I segmented every signal into non-overlapping 2048-sample windows, extracted 25 signal-only statistical and FFT features, and explicitly excluded IDs, labels, load, RPM, and other metadata from the model. I compared SVM, Random Forest, and XGBoost using nested group-aware validation and Leave-One-Load-Out testing. Random Forest was selected by mean Leave-One-Load-Out macro-F1. The dashboard reuses the same preprocessing and feature code, predicts condition and fault type, and reports a health index based on healthy-class probability. The result is a reproducible CWRU benchmark, not yet a field-validated ceiling-fan diagnostic system.

---

## 3. Problem statement

Bearings develop defects such as:

- Inner-race faults
- Ball or rolling-element faults
- Outer-race faults

These defects change the vibration produced by a rotating motor. The goal is to learn patterns in vibration windows and answer two questions:

1. Binary task: Is the bearing `Healthy` or `Faulty`?
2. Multiclass task: Is the signal `Normal`, `Inner_Race`, `Outer_Race`, or `Ball`?

The intended industry context is Novela Engineers and ceiling-fan motors. However, the model is trained on a laboratory benchmark, so the current project demonstrates the full diagnostic workflow but does not claim production accuracy on real ceiling fans.

## 4. System architecture

```text
Official CWRU website
        │
        ▼
40 untouched .mat recordings ───────► data manifest
        │
        ▼
Select *_DE_time vibration channel
        │
        ├── 48 kHz Normal ─► FIR decimate ×4 ─► 12 kHz
        └── 12 kHz faults ────────────────────► 12 kHz
        │
        ▼
Non-overlapping windows: 2048 and 4096 samples
        │
        ▼
25 signal features per window
        │
        ▼
Validated feature CSV
        │
        ├── StratifiedGroupKFold by File_ID
        └── Leave-One-Load-Out by motor load
        │
        ▼
SVM / Random Forest / XGBoost
        │
        ▼
Metrics, confusion matrices, importance, leakage audit
        │
        ▼
Saved binary + multiclass models
        │
        ▼
Streamlit dashboard
```

The most important architectural rule is that a complete recording is treated as one group. Windows from one file never appear in both sides of an outer train/test split.

---

## 5. Project structure

| Path | Purpose |
|---|---|
| `data/raw/` | The 40 downloaded `.mat` recordings; never rewritten by the pipeline |
| `data/processed/cwru_features.csv` | Main 2048-sample feature dataset |
| `data/processed/cwru_features_4096.csv` | Alternative 4096-sample feature dataset |
| `docs/data_manifest.csv` | Recording identity, labels, load, RPM, sampling information, source URL, channel and sample count |
| `src/download.py` | Recording registry, download, MATLAB-key inspection, manifest creation |
| `src/features.py` | All 25 signal-feature formulas |
| `src/extract.py` | Downsampling, windowing and feature CSV generation |
| `src/validate.py` | Dataset integrity, label, balance and signal sanity checks |
| `src/train.py` | Feature boundary, grouped splits, nested tuning, training, metrics, importance and saved models |
| `src/evaluate.py` | Summary tables, confusion matrices, importance plots and leakage audit |
| `src/inference.py` | Shared dashboard input parsing, resampling, windowing and prediction |
| `models/` | Saved binary and multiclass Random Forest pipelines |
| `results/` | Metrics, predictions, plots, feature lists and audit reports |
| `app/dashboard.py` | Interactive Streamlit dashboard |
| `requirements.txt` | Python dependencies |
| `README.md` | Minimal reproduction commands |

---

## 6. Technology stack and why each technology was used

| Technology | Where it is used | Why it was selected |
|---|---|---|
| Python | Entire project | Strong scientific ecosystem, readable scripts, reproducible pipeline |
| `pathlib` | File paths | Standard-library, cross-platform path handling |
| `urllib.request` | Downloading `.mat` files | Standard library is sufficient; no extra HTTP dependency required |
| NumPy | Signal arrays, FFT, numerical features | Efficient vectorized numerical operations |
| Pandas | Manifest, feature tables, metrics CSVs | Convenient typed tabular processing and CSV export |
| SciPy `loadmat` | MATLAB files | Reads native CWRU `.mat` recordings directly |
| SciPy `decimate` | 48 kHz → 12 kHz normal data | Applies anti-alias filtering before downsampling |
| SciPy statistics | Kurtosis and skewness | Tested statistical implementations with explicit Pearson kurtosis choice |
| NumPy real FFT | Frequency-domain features | Efficient single-sided spectrum for real-valued signals |
| Matplotlib | Signal, FFT, confusion and importance plots | Stable static scientific plotting |
| scikit-learn | Pipelines, splits, tuning, SVM, RF and metrics | Mature ML framework with group-aware validation |
| XGBoost | Boosted-tree baseline | Strong nonlinear tabular classifier for comparison with SVM and RF |
| Joblib | Model serialization | Native scikit-learn model persistence |
| Streamlit | Dashboard | Turns Python inference into an interactive app with little separate frontend code |
| Tabulate | Markdown result tables | Lets Pandas export examiner-readable Markdown tables |

Why there is no database: the dataset and results are file-based and small enough for CSV and model artifacts. A database would not improve this offline benchmark.

Why there is no deep-learning framework: a 1D-CNN was optional. The 25-feature classical pipeline already provides grouped evaluation, explainability, fast inference, and a small deployment footprint. A CNN becomes useful when enough target-domain raw fan data exists to justify its extra tuning and compute.

Why SMOTE is not installed: imbalance is handled through class weights. Synthetic oversampling was not necessary for the required baseline and could distort the geometry of feature space. If added later, it must operate inside each training fold only.

---

## 7. Dataset details

### 7.1 What is CWRU?

CWRU refers to the Case Western Reserve University Bearing Data Center dataset. The original experiment used a motor test rig with accelerometers and deliberately seeded bearing faults under controlled motor loads. The files contain vibration arrays and, in most cases, RPM.

### 7.2 Selected sensor

Only the Drive-End accelerometer channel is used. Its MATLAB variable name ends in `_DE_time`.

Common CWRU suffixes are:

- `DE_time`: Drive-End accelerometer vibration
- `FE_time`: Fan-End accelerometer vibration
- `BA_time`: base accelerometer vibration
- `RPM`: rotational speed

The project avoids mixing sensors because sensor position changes the vibration distribution and would introduce an extra domain variable.

### 7.3 Selected recordings

The project uses 40 files:

- Four normal recordings at loads 0, 1, 2 and 3 HP
- 12 inner-race recordings: 3 diameters × 4 loads
- 12 ball recordings: 3 diameters × 4 loads
- 12 outer-race-at-6-o'clock recordings: 3 diameters × 4 loads

| Condition | Diameter | Load 0 | Load 1 | Load 2 | Load 3 |
|---|---:|---:|---:|---:|---:|
| Normal | 0 | 97 | 98 | 99 | 100 |
| Inner Race | 0.007 in | 105 | 106 | 107 | 108 |
| Ball | 0.007 in | 118 | 119 | 120 | 121 |
| Outer Race @ 6 o'clock | 0.007 in | 130 | 131 | 132 | 133 |
| Inner Race | 0.014 in | 169 | 170 | 171 | 172 |
| Ball | 0.014 in | 185 | 186 | 187 | 188 |
| Outer Race @ 6 o'clock | 0.014 in | 197 | 198 | 199 | 200 |
| Inner Race | 0.021 in | 209 | 210 | 211 | 212 |
| Ball | 0.021 in | 222 | 223 | 224 | 225 |
| Outer Race @ 6 o'clock | 0.021 in | 234 | 235 | 236 | 237 |

The raw files are downloaded from direct official CWRU URLs of the form:

```text
https://engineering.case.edu/sites/default/files/<file_id>.mat
```

### 7.4 Loads and speed

The four nominal operating points are approximately:

| Load | Approximate speed |
|---:|---:|
| 0 HP | 1797 RPM |
| 1 HP | 1772 RPM |
| 2 HP | 1750 RPM |
| 3 HP | 1730 RPM |

The manifest records measured RPM when present in the MATLAB file. Therefore, actual values can differ slightly from the nominal table—for example, normal file 100 reports 1725 RPM.

### 7.5 The file 99 edge case

`99.mat` contains more than one Drive-End-looking key, including data with an `X098` prefix. The downloader therefore does not blindly take the first matching key. It prefers the exact key matching the file ID, `X099_DE_time`. This prevents a subtle channel-selection error.

### 7.6 Authenticity and traceability

Authenticity is supported by:

- Official source URLs in the manifest
- Original `.mat` files retained in `data/raw/`
- Recorded file IDs and filenames
- Recorded selected MATLAB channel
- Recorded source and target sampling rates
- Recorded sample counts, loads, RPM, conditions and fault diameters

Current limitation: the pipeline does not store cryptographic file hashes. Adding SHA-256 values would strengthen byte-level provenance if the project moves into a regulated workflow.

---

## 8. Sampling-rate normalization

### 8.1 The problem

The selected fault recordings are 12 kHz, but normal baseline files 97–100 are 48 kHz. Mixing them without correction would be invalid because:

- FFT bins would represent different physical frequencies.
- A 2048-sample window would cover different time durations.
- The model could detect sampling-rate artifacts instead of bearing condition.

### 8.2 The implemented solution

Normal data is converted from 48 kHz to 12 kHz using:

```python
decimate(signal, 4, ftype="fir", zero_phase=True)
```

Meaning:

- Decimation factor: 4
- Input rate: 48,000 samples/second
- Output rate: 12,000 samples/second
- Filter type: FIR
- Default FIR design: order 80, or 81 coefficients, with a Hamming window for factor 4
- `zero_phase=True`: avoids phase displacement of signal events

### 8.3 Why filtering is mandatory

Simply taking every fourth sample would allow frequency content above the new 6 kHz Nyquist frequency to fold into lower frequencies. This is aliasing. The FIR low-pass filter suppresses that high-frequency content before downsampling.

### 8.4 Nyquist frequency

At 12 kHz sampling:

```text
Nyquist frequency = sampling rate / 2 = 6000 Hz
```

All FFT features therefore cover 0–6000 Hz.

---

## 9. Segmentation

### 9.1 Main window

The main dataset uses 2048 samples per window.

At 12 kHz:

```text
Window duration = 2048 / 12000 = 0.1707 seconds
FFT-bin spacing = 12000 / 2048 = 5.8594 Hz
```

This provides a practical balance between:

- Enough time samples for stable statistics
- Useful frequency resolution
- Enough windows for model training
- Low inference latency

### 9.2 Alternative window

The project also creates a 4096-sample version:

```text
Window duration = 4096 / 12000 = 0.3413 seconds
FFT-bin spacing = 12000 / 4096 = 2.9297 Hz
```

The 4096 version has better frequency resolution but produces fewer training examples and has higher latency.

### 9.3 Overlap

Overlap is zero. The step size equals the window size.

Reasons:

- Adjacent samples are not duplicated across windows.
- Leakage risk is reduced.
- Dataset size remains honest.
- Window IDs map cleanly to recording and position.

Even if overlap were introduced, File_ID grouping would still be required. Overlap must never cross train/test boundaries.

### 9.4 Incomplete tails

If the remaining signal has fewer than one full window, that tail is dropped. No padding, interpolation, or synthetic vibration is added.

### 9.5 Window IDs

Each ID has the form:

```text
<file_id>_<window_size>_<zero-padded index>
```

Example:

```text
105_2048_0003
```

This is the fourth 2048-sample window from file 105.

### 9.6 Generated row counts

- 2048-sample dataset: 2,331 windows
- 4096-sample dataset: 1,146 windows

The main four-class distribution is:

| Class | Windows |
|---|---:|
| Normal | 206 |
| Inner Race | 709 |
| Ball | 708 |
| Outer Race | 708 |

Binary distribution:

- Healthy: 206
- Faulty: 2,125
- Faulty-to-Healthy ratio: approximately 10.3:1

Normal load 0 has only 29 windows because file 97 is shorter after correct 48→12 kHz conversion. Normal loads 1–3 have 59 windows each.

---

## 10. Feature engineering

The project extracts 25 signal-only features from every window. Metadata is retained in the CSV for auditing but not passed to the model.

Let a window contain samples \(x_1, x_2, ..., x_N\), where \(N=2048\) for the main dataset.

### 10.1 Time-domain features

#### RMS

```text
RMS = sqrt(mean(x²))
```

RMS represents vibration energy or effective amplitude. Fault impacts often increase it.

#### Standard deviation

```text
Std = sqrt(mean((x - mean(x))²))
```

NumPy's population definition is used (`ddof=0`). It measures spread around the mean.

#### Peak

```text
Peak = max(abs(x))
```

Captures the largest absolute impact.

#### Peak-to-peak

```text
Peak-to-peak = max(x) - min(x)
```

Measures the full amplitude range.

#### Mean

```text
Mean = sum(x) / N
```

Vibration should usually be centered near zero. A nonzero mean can indicate offset or preprocessing differences.

#### Pearson kurtosis

```text
Kurtosis = E[(x - μ)^4] / σ^4
```

The implementation uses `scipy.stats.kurtosis(..., fisher=False)`, so it is Pearson kurtosis. A Gaussian distribution is approximately 3, not 0. High kurtosis indicates impulsive behavior often associated with localized faults.

#### Skewness

```text
Skewness = E[(x - μ)^3] / σ^3
```

Measures asymmetry of the amplitude distribution.

#### Crest factor

```text
Crest Factor = Peak / RMS
```

High values indicate sharp peaks relative to average energy.

#### Shape factor

```text
Shape Factor = RMS / mean(abs(x))
```

Describes waveform shape independently of simple amplitude scaling.

#### Impulse factor

```text
Impulse Factor = Peak / mean(abs(x))
```

Emphasizes isolated high-amplitude impacts.

#### Margin factor

```text
Margin Factor = Peak / mean(sqrt(abs(x)))²
```

Another impulsiveness measure commonly used in bearing analysis.

All ratio features return zero when their denominator is zero. This prevents NaN or infinity in silent windows.

### 10.2 Frequency-domain preparation

Before the FFT, a Hann window is applied:

```text
x_windowed[n] = x[n] × Hann[n]
```

Why: finite signal cutting creates discontinuities at the window edges. The Hann window reduces spectral leakage caused by these edge discontinuities.

Do not confuse spectral leakage with ML data leakage. Spectral leakage is a signal-processing effect; data leakage is unwanted information crossing training and testing.

Because the vibration is real-valued, `rfft` calculates only the nonnegative half of the spectrum. Amplitude is scaled by `2 / sum(Hann)`. The DC bin at 0 Hz is set to zero so constant offset cannot become the dominant frequency.

Power is defined as:

```text
Power[k] = Amplitude[k]²
```

### 10.3 Frequency-domain features

#### FFT peak frequency

The frequency whose non-DC FFT amplitude is largest.

#### FFT peak amplitude

The largest non-DC FFT amplitude.

#### Spectral centroid

```text
Centroid = Σ(f[k] × p[k])
```

where normalized spectral power is:

```text
p[k] = Power[k] / ΣPower
```

The centroid is the spectrum's center of mass.

#### Spectral spread

```text
Spread = sqrt(Σ((f[k] - Centroid)² × p[k]))
```

Measures how widely power is distributed around the centroid.

#### Spectral entropy

```text
Entropy = -Σ(p[k] × log2(p[k]))
```

Zero-probability bins are excluded from the logarithm. Low entropy means concentrated power; high entropy means more distributed spectral power.

#### Spectral kurtosis-like feature

The code computes Pearson kurtosis of the power-spectrum values. It measures whether spectral power is concentrated in a few extreme bins. It should be described as a spectral-kurtosis-like summary, not the formal time-varying spectral kurtosis estimator used in advanced envelope analysis.

#### Eight band energies

The single-sided power vector from 0 to 6000 Hz is split into eight nearly equal index groups, and each group's power is summed:

```text
Band Energy i = Σ Power[k] for bins in band i
```

Each band is approximately 750 Hz wide at 12 kHz sampling. `np.array_split` is used, so bin counts can differ by at most one.

### 10.4 Exact model feature list

1. RMS
2. Std
3. Peak
4. Kurtosis
5. Crest_Factor
6. FFT_Peak_Frequency
7. Peak_to_Peak
8. Mean
9. Skewness
10. Shape_Factor
11. Impulse_Factor
12. Margin_Factor
13. FFT_Peak_Amplitude
14. Spectral_Centroid
15. Spectral_Spread
16. Spectral_Entropy
17. Spectral_Kurtosis
18. Band_Energy_1
19. Band_Energy_2
20. Band_Energy_3
21. Band_Energy_4
22. Band_Energy_5
23. Band_Energy_6
24. Band_Energy_7
25. Band_Energy_8

### 10.5 Explicitly excluded model columns

- `File_ID`
- `Fault_Diameter`
- `RPM`
- `Load`
- `Window_Size`
- `Sampling_Rate`
- `Window_ID`
- `Bearing_Condition`
- `Fault_Type`

Why exclude them: they are metadata or labels, not vibration-derived evidence. Including them could let the model memorize experimental conditions or directly read the answer.

---

## 11. Dataset validation

`src/validate.py` performs these checks before training:

1. Every `Window_ID` is unique.
2. The feature CSV contains no missing values.
3. Numeric columns contain no positive or negative infinity.
4. Each row's condition, fault type, and load match its source file in the manifest.
5. Numeric feature columns have numeric dtypes.
6. Class counts are reported by type, diameter and load.
7. Class imbalance is flagged when the largest class is more than 1.5 times the smallest.
8. Counts below 300 windows per fault type are reported.
9. Healthy and Faulty RMS/kurtosis summaries are compared as a sanity check.
10. Example raw waveforms and spectra are plotted for every fault type.

Observed sanity statistics:

| Condition | Mean RMS | Median RMS | Mean Pearson kurtosis | Median Pearson kurtosis |
|---|---:|---:|---:|---:|
| Faulty | 0.2910 | 0.1848 | 8.8656 | 6.4331 |
| Healthy | 0.0608 | 0.0587 | 2.9651 | 2.9736 |

This difference is evidence that extraction is plausible; it is not a model-performance result.

---

## 12. Data leakage and why a random window split is wrong

Windows from one physical recording are highly related. If windows from the same file are randomly divided between training and testing, the model can learn recording-specific characteristics rather than general fault behavior. Overlapping windows make this even worse.

This project prevents that by grouping on `File_ID`.

For every outer split, the training code asserts:

```python
set(train_file_ids).isdisjoint(test_file_ids)
```

It also asserts that every test fold contains every class. Because there are exactly four Normal recordings and four outer folds, each test fold contains Normal data.

No random-window-split result is used or presented as a headline.

---

## 13. Evaluation strategies

### 13.1 Four-fold StratifiedGroupKFold

Purpose: estimate generalization to unseen recordings while maintaining class representation.

- Number of outer folds: 4
- Group: `File_ID`
- Stratification label: task label
- Shuffle: yes
- Random seed: 42
- Guarantee checked in code: train/test file sets are disjoint and all test classes exist

### 13.2 Leave-One-Load-Out

This is the primary result.

Four folds are created:

- Train on loads 1, 2, 3; test on load 0
- Train on loads 0, 2, 3; test on load 1
- Train on loads 0, 1, 3; test on load 2
- Train on loads 0, 1, 2; test on load 3

Why it is stricter: motor load affects RPM and vibration amplitude. Testing on an unseen load asks whether the learned fault signature generalizes beyond the operating conditions used for training.

Every load contains one normal recording and nine fault recordings, so each test fold contains all required classes.

### 13.3 Nested tuning

The outer test fold is never used for hyperparameter selection. Inside each outer training partition, a three-fold `StratifiedGroupKFold` grouped by `File_ID` is used by `GridSearchCV`.

```text
Outer split: honest performance estimation
    └── Inner grouped split: hyperparameter selection
```

This is called nested cross-validation.

### 13.4 Pipeline safety

SVM scaling is inside a scikit-learn `Pipeline`:

```text
StandardScaler → SVC
```

Therefore, scaler mean and variance are learned only from each training fold. Random Forest and XGBoost do not require scaling because tree splits depend on value ordering, not Euclidean scale.

---

## 14. Imbalance handling

The binary task is approximately 10.3:1 Faulty-to-Healthy. Plain accuracy could therefore look high even if Healthy detection is poor.

Implemented handling:

- SVM: `class_weight="balanced"`
- Random Forest: `class_weight="balanced"`
- XGBoost: `compute_sample_weight("balanced", y)` passed only for the current training fold

Balanced class weights are approximately inversely proportional to class frequency:

```text
weight(class c) = total samples / (number of classes × samples in c)
```

SMOTE is not used. If added, it must be fitted inside each inner training fold, never before cross-validation.

---

## 15. Models and hyperparameters

### 15.1 Support Vector Machine

Model: RBF-kernel SVC with class balancing and probability estimates.

Tuned grid:

- `C`: 1 or 10
- `gamma`: `scale` or 0.01

Why SVM: it is a strong nonlinear baseline for small-to-medium numerical feature datasets.

Role of `C`: controls the penalty for training errors. Higher `C` fits training samples more aggressively.

Role of `gamma`: controls how local the RBF influence is. High gamma produces more complex boundaries.

Why scaling: SVM distances are sensitive to feature units.

### 15.2 Random Forest

Base configuration:

- 250 trees
- `class_weight="balanced"`
- `random_state=42`
- parallel tree building with `n_jobs=-1`

Tuned grid:

- Maximum depth: unlimited or 12
- Minimum leaf size: 1 or 3

Why Random Forest:

- Captures nonlinear feature interactions
- Handles different feature scales
- Robust on tabular data
- Provides native feature importance
- Fast prediction
- Requires little preprocessing

### 15.3 XGBoost

Base configuration:

- 200 boosted trees
- Subsample: 0.9
- Column sample per tree: 0.9
- Four worker threads
- Seed: 42
- Binary objective: `binary:logistic`
- Multiclass objective: `multi:softprob`

Tuned grid:

- Maximum depth: 3 or 5
- Learning rate: 0.05 or 0.1

Why XGBoost: it is an effective boosted-tree comparison and often provides strong tabular performance with compact inference.

### 15.4 Model selection

For each task:

1. Calculate mean Leave-One-Load-Out macro-F1 for SVM, RF, and XGBoost.
2. Select the model with the highest mean.
3. Select the most frequently chosen hyperparameter combination across the four Leave-One-Load-Out folds.
4. Refit that model on the complete validated feature dataset.
5. Save model, exact feature order, class order, task name, selection reason, and cross-validation scores in a Joblib artifact.

Random Forest won both tasks.

Final saved configuration:

```text
RandomForestClassifier(
    n_estimators=250,
    class_weight="balanced",
    max_depth=None,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1
)
```

---

## 16. Metrics and why they matter

### 16.1 Confusion matrix terms

For a chosen positive class:

- TP: positive correctly predicted
- TN: negative correctly predicted
- FP: negative incorrectly predicted positive
- FN: positive incorrectly predicted negative

### 16.2 Precision

```text
Precision = TP / (TP + FP)
```

Of samples predicted as a class, how many were correct?

### 16.3 Recall

```text
Recall = TP / (TP + FN)
```

Of all true samples of a class, how many were found?

### 16.4 F1

```text
F1 = 2 × Precision × Recall / (Precision + Recall)
```

### 16.5 Macro-F1

Compute F1 separately for every class, then take the unweighted mean. Every class matters equally even when counts differ.

### 16.6 Balanced accuracy

Balanced accuracy is the mean recall across classes. For binary classification it is:

```text
Balanced accuracy = (Healthy recall + Faulty recall) / 2
```

### 16.7 ROC-AUC

Measures ranking across thresholds using true-positive rate versus false-positive rate. Healthy is treated as the positive class in this project.

### 16.8 PR-AUC

Area under the precision-recall curve. It is particularly informative for an imbalanced target. Healthy is the positive class here because it is the minority class.

### 16.9 Why plain accuracy is not the headline

A classifier predicting every binary sample as Faulty would achieve approximately 91% accuracy but 0% Healthy recall. Therefore, the report headlines balanced accuracy, macro-F1, Healthy precision/recall, PR-AUC and ROC-AUC.

### 16.10 Timing

- Training time includes the inner grid search and refit for an outer fold.
- Inference time is measured for `predict` plus `predict_proba`, divided by the number of test windows.
- Timing varies with hardware and system load and is not a hard real-time guarantee.

---

## 17. Primary results

All following values are mean ± standard deviation across the four Leave-One-Load-Out folds.

### 17.1 Binary Healthy vs Faulty

| Model | Balanced accuracy | Macro-F1 | Healthy precision | Healthy recall | Healthy PR-AUC | Healthy ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Random Forest | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| SVM | 0.8982 ± 0.2030 | 0.8894 ± 0.2180 | 0.7822 ± 0.4246 | 0.8190 ± 0.3621 | 0.7966 ± 0.4068 | 0.9727 ± 0.0545 |
| XGBoost | 0.9995 ± 0.0009 | 0.9956 ± 0.0088 | 0.9839 ± 0.0323 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |

### 17.2 Four-class fault type

| Model | Balanced accuracy | Macro-F1 | Ball recall | Inner-race recall | Normal recall | Outer-race recall |
|---|---:|---:|---:|---:|---:|---:|
| Random Forest | 0.9813 ± 0.0356 | 0.9810 ± 0.0361 | 0.9929 ± 0.0107 | 0.9336 ± 0.1328 | 1.0000 ± 0.0000 | 0.9986 ± 0.0028 |
| SVM | 0.9333 ± 0.1288 | 0.9075 ± 0.1803 | 0.8941 ± 0.2081 | 0.9209 ± 0.1471 | 1.0000 ± 0.0000 | 0.9181 ± 0.1601 |
| XGBoost | 0.9727 ± 0.0498 | 0.9710 ± 0.0533 | 0.9774 ± 0.0348 | 0.9336 ± 0.1328 | 0.9828 ± 0.0345 | 0.9972 ± 0.0056 |

### 17.3 Important error-analysis point

Load 0 is the hardest four-class condition. Random Forest's load-0 inner-race recall is 0.7345, while loads 1–3 are almost perfect. This is why per-load reporting is more honest than presenting only an overall score.

### 17.4 Why perfect binary performance was not accepted blindly

Thirty-four fold/model results crossed the 99.5% red-flag threshold. The project rechecked:

- Disjoint train/test File_ID sets
- Signal-only feature list
- Exclusion of load, RPM, diameter, IDs and labels
- Complete held-out load in Leave-One-Load-Out
- Every test class present
- Inner-only preprocessing and tuning

No structural leakage was found. The perfect binary Random Forest result is therefore reported, but interpreted as separation within the CWRU benchmark—not proof of real-world fan performance.

Possible reasons for high CWRU scores include controlled laboratory conditions, deliberately seeded faults, consistent sensor placement, and a substantial domain difference between healthy and faulted recordings.

---

## 18. Explainability

Two importance methods are saved for RF and XGBoost:

### Native tree importance

Measures how much tree splits reduce impurity or loss. It is fast but can favor continuous/high-cardinality features and can distribute importance unpredictably among correlated features.

### Out-of-fold permutation importance

For each held-out load, one feature is randomly shuffled and the drop in macro-F1 is measured. A large drop suggests the model relies on that feature for generalization.

Selected observations:

- Binary XGBoost relies strongly on spectral spread and RMS.
- Four-class RF and XGBoost emphasize higher-frequency band energies, especially bands 6 and 8.
- Binary RF has near-zero individual permutation importance because many correlated features can substitute for one another while the classifier remains perfect. Zero permutation importance does not mean the complete feature set is useless.

Feature importance shows association with predictions; it does not prove physical causality.

---

## 19. Dashboard and inference workflow

### 19.1 Accepted input

- `.mat`: selects a Drive-End `_DE_time` variable
- `.npy`: loads a numeric NumPy array with `allow_pickle=False`
- `.csv`: loads a numeric comma-separated array
- Built-in CWRU sample chosen from the manifest

### 19.2 Input validation

The inference code rejects:

- Unsupported file extensions
- Ambiguous MATLAB Drive-End channels
- Non-one-dimensional arrays after squeezing
- NaN or infinite values
- Sampling rates other than 12 kHz or 48 kHz
- Signals shorter than 2048 samples after resampling

### 19.3 Processing

1. Read the signal.
2. If it is 48 kHz, apply the same FIR decimation by four.
3. Divide it into non-overlapping 2048-sample windows.
4. Drop the incomplete tail and notify the user.
5. Extract the same 25 features in the saved order.
6. Run binary and multiclass `predict_proba`.
7. Average probabilities across all complete windows.
8. Display predicted condition, confidence, fault type, health index, probability bar chart, first waveform and first FFT.

### 19.4 Health index

```text
Health Index = 100 × predicted probability of Healthy
```

Examples:

- Healthy probability 0.92 → Health Index 92
- Healthy probability 0.10 → Health Index 10

It is not remaining useful life, percentage bearing life, a safety certification, or a calibrated maintenance threshold. Random Forest probabilities can also be poorly calibrated outside the training domain.

---

## 20. Reproducibility

Reproducibility controls include:

- Seed 42 for NumPy, splitters, SVM probability fitting, Random Forest, XGBoost and permutation importance
- Explicit file registry
- Untouched raw recordings
- Manifest of preprocessing choices
- Deterministic non-overlapping windows
- Saved feature list and order
- Saved per-window predictions and per-fold metrics
- Saved selected model and class order

What may still vary:

- Wall-clock training/inference timings
- Small numerical differences across package/BLAS versions
- The official source could change a hosted file in the future because hashes are not currently pinned

---

## 21. Output artifacts and what each proves

| Artifact | Evidence provided |
|---|---|
| `docs/data_manifest.csv` | Exact recording provenance and preprocessing |
| `data/processed/cwru_features.csv` | Main extracted dataset |
| `results/dataset_validation.md` | Dataset passed integrity and label checks |
| `results/feature_list.json` | Exact 25 model inputs and order |
| `results/fold_metrics.csv` | Every task/scheme/model/fold metric and timing |
| `results/fold_predictions.csv` | Out-of-fold prediction for every window |
| `results/metrics_summary.csv` | Mean and standard deviation tables |
| `results/model_evaluation.md` | Human-readable primary and per-fold results |
| `results/leakage_audit.md` | Investigation of suspiciously high scores |
| `results/confusion_*.png` | Aggregated confusion matrices |
| `results/feature_importance_folds.csv` | Native and permutation values per fold |
| `results/feature_importance_*.png` | Importance plots |
| `models/binary_model.joblib` | Deployment binary model plus metadata |
| `models/multiclass_model.joblib` | Deployment fault-type model plus metadata |

---

## 22. Known limitations

1. **Domain gap:** CWRU laboratory motors are not Novela ceiling-fan motors.
2. **Seeded faults:** EDM defects differ from natural wear, lubrication problems, looseness and mixed faults.
3. **Sensor setup:** Field sensor type, mounting, orientation and gain may differ.
4. **Noise:** Real fans experience structural, electrical and environmental noise absent from the benchmark.
5. **Operating range:** Loads and speeds are limited to the CWRU test conditions.
6. **Class set:** The model assumes one of four known classes; it has no explicit unknown/anomaly class.
7. **Probability calibration:** RF probability is not independently calibrated for real fans.
8. **Normal data scarcity:** Only four normal source recordings are available.
9. **Recording independence:** CWRU files are grouped correctly, but all still come from the same experimental platform.
10. **No prospective test:** No future, independently collected ceiling-fan dataset has been evaluated.
11. **No severity deployment:** Fault diameter is deliberately excluded from features and the deployed task.
12. **No raw-signal CNN:** Optional and intentionally not built without a clear benefit.
13. **No envelope analysis:** Expected bearing characteristic-frequency features were optional and not used.
14. **No hash manifest:** Source URLs are preserved, but raw-byte hashes are not.

---

## 23. Safe future work

- Collect naturally healthy and naturally degraded fan data.
- Record accelerometer sampling rate, mounting, motor current, temperature, RPM and operating state.
- Keep each physical fan or recording session as a split group.
- Test cross-device validation: train on some fans, test on unseen fans.
- Calibrate probability on a target-domain validation set.
- Add an unknown/anomaly rejection threshold.
- Explore envelope-spectrum and bearing-frequency features when bearing geometry and RPM are reliable.
- Explore transfer learning or domain adaptation after target-domain data exists.
- Compare a 1D-CNN using raw windows with exactly the same group splits.

Do not create unsafe mechanical or electrical faults on operational fan motors. Use naturally degraded units, controlled laboratory safety procedures, or vendor-approved test specimens.

---

## 24. Examiner questions and strong answers

### Q1. What is the main objective?

To classify bearing health and fault type from vibration signals using a reproducible, leakage-safe CWRU benchmark and expose the selected models through a dashboard.

### Q2. Why did you use CWRU?

It is public, widely studied, contains real measured vibration, and documents fault type, defect size, load, speed and sensor channel. It supports reproducible comparisons. Its limitation is domain mismatch with ceiling fans.

### Q3. Is the vibration synthetic?

No. All vibration values come from downloaded CWRU `.mat` files. No random signal, interpolation, augmentation or synthetic fault vibration is used.

### Q4. Why only Drive-End data?

To keep the sensing position consistent. Mixing Drive-End, Fan-End and base signals would add sensor-location variation and could create shortcuts.

### Q5. Why are Normal files downsampled?

They are 48 kHz while the fault files are 12 kHz. All records must represent the same sampling rate for comparable time windows and FFT frequencies.

### Q6. Why not just select every fourth Normal sample?

That would alias content above 6 kHz into the new passband. FIR decimation filters before downsampling.

### Q7. Why is the Nyquist frequency 6 kHz?

Nyquist is half the 12 kHz sampling rate.

### Q8. Why 2048 samples?

It gives a 0.1707-second window and 5.859-Hz FFT-bin spacing at 12 kHz, balancing time locality, stable features, sample count and latency.

### Q9. Why produce 4096 too?

It provides a documented alternative with about 2.93-Hz resolution for future comparison, while the trained baseline remains 2048.

### Q10. Why no overlap?

It avoids duplicated samples, simplifies traceability and reduces leakage risk. More overlapping rows are not more independent physical evidence.

### Q11. What happens to an incomplete final window?

It is dropped. The pipeline does not pad or fabricate samples.

### Q12. What is RMS?

The square root of mean squared amplitude. It reflects effective vibration energy.

### Q13. Is kurtosis excess or Pearson?

Pearson. `fisher=False` is used, so a Gaussian distribution is around 3.

### Q14. What does a high crest factor mean?

The signal contains a high peak relative to its RMS, suggesting impulsive behavior.

### Q15. Why apply a Hann window?

To reduce FFT spectral leakage created by discontinuities at the ends of a finite signal window.

### Q16. Why remove DC?

So constant sensor offset does not become the dominant FFT peak.

### Q17. What is spectral centroid?

The power-weighted average frequency—the spectrum's center of mass.

### Q18. What is spectral entropy?

The Shannon entropy of normalized spectral power. It measures how concentrated or dispersed the energy is.

### Q19. What are band energies?

The summed FFT power in eight approximately equal frequency regions between 0 and 6 kHz.

### Q20. Why exclude Load and RPM?

The model must diagnose from vibration, not use operating metadata as a shortcut. Leave-One-Load-Out also tests operation at an unseen load.

### Q21. Why retain excluded columns in the CSV?

They are essential for traceability, grouping, validation and per-load analysis, but are removed before forming `X`.

### Q22. What is data leakage here?

Information from the test set influencing training. A key example would be windows from the same recording appearing in both sets.

### Q23. Why not random train/test split?

Random splitting of windows would place nearly identical recording-specific patterns on both sides and exaggerate performance.

### Q24. What is StratifiedGroupKFold?

It creates folds that approximately preserve class balance while keeping every group—in this case every File_ID—entirely in one fold.

### Q25. What is Leave-One-Load-Out?

One motor load is held out completely for testing while the other three loads train the model. It evaluates operating-condition generalization.

### Q26. What is nested cross-validation?

An outer split measures performance and an inner split selects hyperparameters. The outer test fold never influences tuning.

### Q27. Why use a Pipeline?

It ensures transformations such as SVM scaling are fitted only on training data and applied consistently at inference.

### Q28. Why does Random Forest not need scaling?

Tree thresholds depend on feature ordering, not Euclidean distance or absolute unit scale.

### Q29. How is imbalance handled?

SVM and RF use balanced class weights. XGBoost receives balanced sample weights computed only for each training fold.

### Q30. Why not headline accuracy?

With roughly 91% Faulty rows, predicting everything as Faulty would look accurate while never finding Healthy. Balanced accuracy and macro-F1 expose this failure.

### Q31. Why is Healthy the positive class for PR-AUC?

Healthy is the minority binary class here. Treating it as positive makes the precision-recall measurement focus on performance for that underrepresented class.

### Q32. What does macro-F1 do?

It averages class F1 values equally, so a majority class cannot dominate the score.

### Q33. Why compare SVM, RF and XGBoost?

They represent three strong but different classical approaches: kernel boundaries, bagged decision trees, and boosted decision trees.

### Q34. Why did Random Forest win?

It had the highest mean Leave-One-Load-Out macro-F1 for both tasks and performed robustly without scaling.

### Q35. How was the final model trained?

After honest outer evaluation and model selection, the winning configuration was refitted on the complete validated dataset for deployment.

### Q36. Is evaluating the refitted full-data model honest?

No performance claim is taken from the refitted deployment model. Reported performance comes only from held-out outer folds.

### Q37. Why is binary RF exactly 100%?

The CWRU classes are highly separable under controlled conditions. Structural leakage checks passed, but the score remains a red flag and is not generalized to real fans.

### Q38. Which condition was hardest?

The four-class load-0 fold, especially Inner Race for Random Forest, with recall 0.7345.

### Q39. What is a confusion matrix?

Rows represent true classes and columns predicted classes. Off-diagonal cells show which faults are confused.

### Q40. What does feature importance mean?

It estimates which features influence predictions. It does not prove those features physically cause a fault.

### Q41. Why can permutation importance be zero?

Correlated features can replace one another. Shuffling one feature may not hurt performance even though the group of related features matters.

### Q42. How does the dashboard classify a long signal?

It segments all complete 2048-sample windows, predicts each, and averages class probabilities.

### Q43. How is the health index calculated?

One hundred times the binary model's Healthy probability.

### Q44. Is the health index remaining useful life?

No. It is only a model confidence-derived score and has no time-to-failure interpretation.

### Q45. Can this run directly on a fan today?

Technically it can process a compatible signal, but its result is not validated for fan maintenance decisions until target-domain data is collected and tested.

### Q46. How do you handle bad input?

The loader validates extension, array dimensionality, finite values, sampling rate, minimum length and ambiguous MATLAB channels, then displays errors in Streamlit.

### Q47. Why `allow_pickle=False` for NPY?

It avoids loading arbitrary pickled Python objects and limits input to numerical arrays.

### Q48. How is reproducibility achieved?

Fixed seeds, explicit file registry, deterministic preprocessing, saved feature order, grouped fold predictions, metrics and model metadata.

### Q49. What would you improve first?

Collect several independent healthy and naturally degraded Novela fan recordings and evaluate with fan/device-grouped splits.

### Q50. What is the biggest limitation?

The CWRU-to-ceiling-fan domain gap. Strong benchmark metrics do not guarantee field performance.

---

## 25. Recommended demonstration sequence

1. Show `docs/data_manifest.csv` and explain real-file traceability.
2. Show one raw `.mat` file and its `_DE_time` variable.
3. Explain 48→12 kHz FIR decimation.
4. Show `cwru_features.csv` and point out labels, features and metadata.
5. Show `results/feature_list.json` and emphasize excluded metadata.
6. Show `results/dataset_validation.md`.
7. Explain grouped and Leave-One-Load-Out validation.
8. Show `results/model_evaluation.md` and the load-0 weakness.
9. Show one confusion matrix and one feature-importance plot.
10. Run the Streamlit dashboard using one Normal and one fault sample.
11. End with the domain-gap limitation and target-fan validation plan.

---

## 26. Troubleshooting

### `ModuleNotFoundError`

Activate/use the project virtual environment and install requirements:

```bash
.venv/bin/pip install -r requirements.txt
```

### Download fails

Confirm internet access and the official CWRU site. Do not replace the data with synthetic signals.

### Dashboard says signal is too short

Provide at least 2048 samples at 12 kHz, or at least 8192 samples when selecting 48 kHz because downsampling by four must still leave 2048 samples.

### CSV fails to load

The dashboard expects numeric comma-separated signal values without text headers.

### MATLAB channel is ambiguous

Use a filename whose numeric stem matches the CWRU file ID or remove unrelated Drive-End arrays. Built-in samples use the manifest's exact channel.

### Port already in use

```bash
.venv/bin/streamlit run app/dashboard.py --server.port 8502
```

### Results changed slightly

Check package versions, raw files and hardware threading. Classification metrics should be reproducible with the fixed seeds, while timing naturally varies.

---

## 27. Final conclusion

The project is a complete, reproducible vibration-diagnosis benchmark: real CWRU acquisition, traceable preprocessing, physically meaningful features, leakage-safe grouped validation, imbalance-aware metrics, model comparison, explainability, saved inference artifacts and an interactive dashboard. Its strongest engineering choice is not the classifier; it is the disciplined separation between raw recordings, auditable metadata, signal-only features, grouped testing and honest domain-limit reporting.
