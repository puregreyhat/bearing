# CWRU Bearing Fault Diagnosis

End-to-end bearing-fault benchmark using real CWRU drive-end vibration recordings only.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python src/download.py
.venv/bin/python src/extract.py
.venv/bin/python src/validate.py
.venv/bin/python src/train.py
.venv/bin/python src/evaluate.py
.venv/bin/streamlit run app/dashboard.py
```

Raw `.mat` files remain untouched in `data/raw`. Normal recordings (97–100) are 48 kHz and are FIR-decimated by four during extraction; all feature rows therefore represent 12 kHz signals. Segmentation is non-overlapping with incomplete tails dropped. Outputs are `cwru_features.csv` (2048 samples) and `cwru_features_4096.csv`.

Training evaluates binary and four-class tasks with nested, file-grouped cross-validation and Leave-One-Load-Out testing. The dashboard accepts numeric `.csv`, `.npy`, or Drive-End `.mat` signals at 12 or 48 kHz. Results are CWRU laboratory benchmarks and must not be treated as validated ceiling-fan performance.
