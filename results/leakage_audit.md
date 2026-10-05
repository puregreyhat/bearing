# Leakage audit

34 fold/model results reached 99.5%, so leakage was explicitly rechecked:

- `train.py` asserts disjoint File_ID sets for every outer fold.
- Feature selection hard-excludes IDs, labels, load, RPM, diameter, sampling rate, and window metadata.
- Leave-One-Load-Out test rows contain only the held-out load.
- Every test fold contains Healthy/Normal data and all task classes.
- Preprocessing and tuning occur inside each outer training fold.

The high scores therefore remain reported as CWRU benchmark separability, not assumed deployment performance.
