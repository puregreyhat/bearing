# Dashboard User Guide

## Supported product

The supported SRS product surface is the Streamlit dashboard in `app/dashboard.py`. It accepts one numeric vibration channel from CSV, NPY, or Drive-End MAT input, processes it using the shared CWRU contract, and reports binary or multiclass bearing diagnosis.

## Start the dashboard

```powershell
conda activate bearing
streamlit run app/dashboard.py
```

Open the local URL printed by Streamlit. The dashboard requires compatible artifacts in `models/` and the generated registry in `results/model_registry.json`.

## Diagnosis workflow

1. Choose `Upload signal` or `CWRU sample`.
2. For an upload, provide a numeric `.csv`, `.npy`, or Drive-End `.mat` signal.
3. Set the source sampling rate to 12,000 or 48,000 Hz. A 48 kHz signal is FIR-decimated by four to the 12 kHz model rate.
4. Select `Binary` for Healthy/Faulty or `Multiclass` for Ball/Inner Race/Normal/Outer Race.
5. Select a compatible available model: Random Forest, SVM, or XGBoost.
6. Click `Run Diagnosis`.
7. Review the primary diagnosis first, then probabilities, processing details, waveform/FFT, features, model evidence, data quality, and limitations.

Signals must contain at least 2,048 samples after resampling. Windows are non-overlapping; incomplete tail samples are discarded and reported.

## Reading results

- **Predicted class** is the class with the largest recording-level mean probability.
- **Confidence** is the selected model probability for that predicted class. It is not calibrated certainty.
- **Health Index** is `100 × Healthy probability` for the binary task. It is not remaining useful life, a maintenance score, or a safety guarantee.
- **Window decisions** show each complete window and the recording-level result is the mean of window probabilities.
- **LOLO macro-F1** is the primary reported benchmark selection metric because loads are held out during validation.

## Input guidance and limitations

Use one numeric signal channel with finite values. Do not include headers, timestamps, labels, RPM, load, or other metadata in the signal vector. The current models are trained on CWRU laboratory recordings and are not validated on real ceiling-fan field data. Differences in sensor, mounting, machine, load, operating speed, noise, or fault mechanism can cause domain shift.

For the browser-only explanatory demo, use `website/dist/`. It contains embedded samples and exported Random Forests only; it is not a replacement for the Streamlit dashboard.

