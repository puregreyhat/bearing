# Dataset validation

- Rows: 2331; columns: 34
- Missing values: 0; infinite values: 0
- Duplicate Window_IDs: 0
- Labels cross-checked against manifest: PASS
- Numeric feature dtypes: PASS
- Imbalance flagged (>1.5x largest/smallest class): YES
- At least 300 windows per Fault_Type: WARNING — Normal

## Windows per fault type

| Fault_Type   |   windows |
|:-------------|----------:|
| Inner_Race   |       709 |
| Ball         |       708 |
| Outer_Race   |       708 |
| Normal       |       206 |

## Balance by fault type, diameter, and load

|                          |   windows |
|:-------------------------|----------:|
| ('Ball', 0.007, 0)       |        59 |
| ('Ball', 0.007, 1)       |        59 |
| ('Ball', 0.007, 2)       |        59 |
| ('Ball', 0.007, 3)       |        59 |
| ('Ball', 0.014, 0)       |        59 |
| ('Ball', 0.014, 1)       |        59 |
| ('Ball', 0.014, 2)       |        59 |
| ('Ball', 0.014, 3)       |        59 |
| ('Ball', 0.021, 0)       |        59 |
| ('Ball', 0.021, 1)       |        59 |
| ('Ball', 0.021, 2)       |        59 |
| ('Ball', 0.021, 3)       |        59 |
| ('Inner_Race', 0.007, 0) |        59 |
| ('Inner_Race', 0.007, 1) |        59 |
| ('Inner_Race', 0.007, 2) |        59 |
| ('Inner_Race', 0.007, 3) |        60 |
| ('Inner_Race', 0.014, 0) |        59 |
| ('Inner_Race', 0.014, 1) |        59 |
| ('Inner_Race', 0.014, 2) |        59 |
| ('Inner_Race', 0.014, 3) |        59 |
| ('Inner_Race', 0.021, 0) |        59 |
| ('Inner_Race', 0.021, 1) |        59 |
| ('Inner_Race', 0.021, 2) |        59 |
| ('Inner_Race', 0.021, 3) |        59 |
| ('Normal', 0.0, 0)       |        29 |
| ('Normal', 0.0, 1)       |        59 |
| ('Normal', 0.0, 2)       |        59 |
| ('Normal', 0.0, 3)       |        59 |
| ('Outer_Race', 0.007, 0) |        59 |
| ('Outer_Race', 0.007, 1) |        59 |
| ('Outer_Race', 0.007, 2) |        59 |
| ('Outer_Race', 0.007, 3) |        59 |
| ('Outer_Race', 0.014, 0) |        59 |
| ('Outer_Race', 0.014, 1) |        59 |
| ('Outer_Race', 0.014, 2) |        59 |
| ('Outer_Race', 0.014, 3) |        59 |
| ('Outer_Race', 0.021, 0) |        59 |
| ('Outer_Race', 0.021, 1) |        59 |
| ('Outer_Race', 0.021, 2) |        59 |
| ('Outer_Race', 0.021, 3) |        59 |

## Healthy/faulty sanity statistics

| Bearing_Condition   |   ('RMS', 'mean') |   ('RMS', 'median') |   ('Kurtosis', 'mean') |   ('Kurtosis', 'median') |
|:--------------------|------------------:|--------------------:|-----------------------:|-------------------------:|
| Faulty              |         0.291016  |           0.184773  |                8.86563 |                  6.43305 |
| Healthy             |         0.0608155 |           0.0587358 |                2.96514 |                  2.97363 |

The distributions differ; this is a sanity observation, not a model-performance claim.
Example waveforms and spectra are saved in `results/signal_spectra_examples.png`.
