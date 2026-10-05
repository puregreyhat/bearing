# CWRU Bearing Fault Diagnosis Report

## Scope

Forty official CWRU recordings were processed: four normal baselines and 36 Drive-End fault recordings covering inner race, ball, and 6-o'clock outer race faults at three diameters and four loads. The feature CSV contains 2,331 non-overlapping 2048-sample windows.

## Evaluation

Leave-One-Load-Out is the primary generalisation test. All models use only the 25 signal features in `results/feature_list.json`; recording identifiers and operating metadata are excluded. Full per-fold results, mean ± standard deviation, confusion matrices, and the leakage audit are in `results/model_evaluation.md`.

The selected deployment models are Random Forests: binary balanced accuracy and macro-F1 were 1.0000 ± 0.0000; four-class balanced accuracy was 0.9813 ± 0.0356 and macro-F1 was 0.9810 ± 0.0361. The perfect binary benchmark result passed the implemented group-disjointness and feature-exclusion checks, but remains a warning against assuming field performance.

## Limitations and next steps

- CWRU uses a laboratory motor, seeded EDM faults, fixed sensors, and controlled loads; it is not ceiling-fan motor data.
- The health index is healthy-class probability, not remaining useful life or a calibrated safety score.
- Validate prospectively on Novela fan motors with naturally degraded bearings before operational use.
- Future work: domain adaptation/transfer learning and synchronized accelerometer, temperature, current, and RPM measurements. Do not create deliberate unsafe faults on real motors.
