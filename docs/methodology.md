# Methodology

- Source: official CWRU Bearing Data Center files.
- Recordings: normal 97–100; 12 kHz drive-end inner-race, ball, and 6-o'clock outer-race faults at 0.007, 0.014, and 0.021 inches for loads 0–3 HP.
- Channel: the single `*_DE_time` variable in each file.
- Sampling: fault files are 12 kHz. Normal baselines are 48 kHz and are reduced to 12 kHz with `scipy.signal.decimate(signal, 4, ftype="fir", zero_phase=True)`: decimation factor 4, the SciPy default order-80 Hamming-window FIR anti-alias filter, and zero-phase filtering.
- Segmentation: 2048 and 4096 samples, zero overlap, incomplete tail dropped. Window IDs encode file, size, and zero-based index.
- Normal load 0 has 29 windows after decimation because file 97 is shorter; normal loads 1–3 have 59 each.
- Features: time statistics, peak ratios, Hann-window single-sided FFT peak, spectral moments/entropy/kurtosis, and eight equal-width band energies. `Kurtosis` and `Spectral_Kurtosis` use Pearson kurtosis (`fisher=False`), so a Gaussian/Normal signal is approximately 3.
- Model inputs contain only the 25 signal features listed in `results/feature_list.json`. IDs, labels, load, fault diameter, RPM, window size, and sampling rate are excluded.
- Evaluation: four-fold `StratifiedGroupKFold` grouped by File_ID and four Leave-One-Load-Out folds. Hyperparameter selection uses three-fold `StratifiedGroupKFold` only inside each outer training fold. Scaling is inside the SVM pipeline.
- Imbalance: SVM and Random Forest use `class_weight="balanced"`; XGBoost receives balanced sample weights. SMOTE is not used.
- Reproducibility: deterministic processing; later stochastic stages use seed 42.
