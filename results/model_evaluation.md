# Model evaluation

All values below are mean ± standard deviation across the four Leave-One-Load-Out folds. Plain accuracy is intentionally not a headline metric. PR-AUC and ROC-AUC treat Healthy as the positive class. SMOTE was not used; imbalance handling is class weighting (or balanced sample weights for XGBoost), so there is no oversampled variant to compare.

### binary — rf

balanced_accuracy: 1.0000 ± 0.0000; macro_f1: 1.0000 ± 0.0000; precision_Healthy: 1.0000 ± 0.0000; recall_Healthy: 1.0000 ± 0.0000; pr_auc_Healthy: 1.0000 ± 0.0000; roc_auc_Healthy: 1.0000 ± 0.0000; training_time_seconds: 3.6841 ± 0.6745; inference_ms_per_window: 0.0934 ± 0.0033

### binary — svm

balanced_accuracy: 0.8982 ± 0.2030; macro_f1: 0.8894 ± 0.2180; precision_Healthy: 0.7822 ± 0.4246; recall_Healthy: 0.8190 ± 0.3621; pr_auc_Healthy: 0.7966 ± 0.4068; roc_auc_Healthy: 0.9727 ± 0.0545; training_time_seconds: 0.1908 ± 0.0128; inference_ms_per_window: 0.0078 ± 0.0016

### binary — xgboost

balanced_accuracy: 0.9995 ± 0.0009; macro_f1: 0.9956 ± 0.0088; precision_Healthy: 0.9839 ± 0.0323; recall_Healthy: 1.0000 ± 0.0000; pr_auc_Healthy: 1.0000 ± 0.0000; roc_auc_Healthy: 1.0000 ± 0.0000; training_time_seconds: 0.8283 ± 0.0046; inference_ms_per_window: 0.0060 ± 0.0002

### multiclass — rf

balanced_accuracy: 0.9813 ± 0.0356; macro_f1: 0.9810 ± 0.0361; recall_Ball: 0.9929 ± 0.0107; recall_Inner_Race: 0.9336 ± 0.1328; recall_Normal: 1.0000 ± 0.0000; recall_Outer_Race: 0.9986 ± 0.0028; training_time_seconds: 3.8359 ± 0.1544; inference_ms_per_window: 0.1002 ± 0.0109

### multiclass — svm

balanced_accuracy: 0.9333 ± 0.1288; macro_f1: 0.9075 ± 0.1803; recall_Ball: 0.8941 ± 0.2081; recall_Inner_Race: 0.9209 ± 0.1471; recall_Normal: 1.0000 ± 0.0000; recall_Outer_Race: 0.9181 ± 0.1601; training_time_seconds: 0.4754 ± 0.0409; inference_ms_per_window: 0.0191 ± 0.0107

### multiclass — xgboost

balanced_accuracy: 0.9727 ± 0.0498; macro_f1: 0.9710 ± 0.0533; recall_Ball: 0.9774 ± 0.0348; recall_Inner_Race: 0.9336 ± 0.1328; recall_Normal: 0.9828 ± 0.0345; recall_Outer_Race: 0.9972 ± 0.0056; training_time_seconds: 3.8681 ± 0.2028; inference_ms_per_window: 0.0089 ± 0.0004

## Per-fold Leave-One-Load-Out results

### binary

| model   |   held_out_load |   balanced_accuracy |   macro_f1 |   precision_Healthy |   recall_Healthy |   pr_auc_Healthy |   roc_auc_Healthy |
|:--------|----------------:|--------------------:|-----------:|--------------------:|-----------------:|-----------------:|------------------:|
| rf      |          0.0000 |              1.0000 |     1.0000 |              1.0000 |           1.0000 |           1.0000 |            1.0000 |
| rf      |          1.0000 |              1.0000 |     1.0000 |              1.0000 |           1.0000 |           1.0000 |            1.0000 |
| rf      |          2.0000 |              1.0000 |     1.0000 |              1.0000 |           1.0000 |           1.0000 |            1.0000 |
| rf      |          3.0000 |              1.0000 |     1.0000 |              1.0000 |           1.0000 |           1.0000 |            1.0000 |
| svm     |          0.0000 |              0.5937 |     0.5624 |              0.1455 |           0.2759 |           0.1863 |            0.8909 |
| svm     |          1.0000 |              1.0000 |     1.0000 |              1.0000 |           1.0000 |           1.0000 |            1.0000 |
| svm     |          2.0000 |              1.0000 |     1.0000 |              1.0000 |           1.0000 |           1.0000 |            1.0000 |
| svm     |          3.0000 |              0.9991 |     0.9953 |              0.9833 |           1.0000 |           1.0000 |            1.0000 |
| xgboost |          0.0000 |              0.9981 |     0.9824 |              0.9355 |           1.0000 |           1.0000 |            1.0000 |
| xgboost |          1.0000 |              1.0000 |     1.0000 |              1.0000 |           1.0000 |           1.0000 |            1.0000 |
| xgboost |          2.0000 |              1.0000 |     1.0000 |              1.0000 |           1.0000 |           1.0000 |            1.0000 |
| xgboost |          3.0000 |              1.0000 |     1.0000 |              1.0000 |           1.0000 |           1.0000 |            1.0000 |

### multiclass

| model   |   held_out_load |   balanced_accuracy |   macro_f1 |   recall_Ball |   recall_Inner_Race |   recall_Normal |   recall_Outer_Race |
|:--------|----------------:|--------------------:|-----------:|--------------:|--------------------:|----------------:|--------------------:|
| rf      |          0.0000 |              0.9280 |     0.9270 |        0.9774 |              0.7345 |          1.0000 |              1.0000 |
| rf      |          1.0000 |              0.9986 |     0.9986 |        0.9944 |              1.0000 |          1.0000 |              1.0000 |
| rf      |          2.0000 |              1.0000 |     1.0000 |        1.0000 |              1.0000 |          1.0000 |              1.0000 |
| rf      |          3.0000 |              0.9986 |     0.9986 |        1.0000 |              1.0000 |          1.0000 |              0.9944 |
| svm     |          0.0000 |              0.7401 |     0.6371 |        0.5819 |              0.7006 |          1.0000 |              0.6780 |
| svm     |          1.0000 |              1.0000 |     1.0000 |        1.0000 |              1.0000 |          1.0000 |              1.0000 |
| svm     |          2.0000 |              1.0000 |     1.0000 |        1.0000 |              1.0000 |          1.0000 |              1.0000 |
| svm     |          3.0000 |              0.9930 |     0.9930 |        0.9944 |              0.9831 |          1.0000 |              0.9944 |
| xgboost |          0.0000 |              0.8980 |     0.8911 |        0.9266 |              0.7345 |          0.9310 |              1.0000 |
| xgboost |          1.0000 |              0.9958 |     0.9958 |        0.9831 |              1.0000 |          1.0000 |              1.0000 |
| xgboost |          2.0000 |              1.0000 |     1.0000 |        1.0000 |              1.0000 |          1.0000 |              1.0000 |
| xgboost |          3.0000 |              0.9972 |     0.9972 |        1.0000 |              1.0000 |          1.0000 |              0.9887 |

## Leakage audit

34 fold/model results reached 99.5%, so leakage was explicitly rechecked:

- `train.py` asserts disjoint File_ID sets for every outer fold.
- Feature selection hard-excludes IDs, labels, load, RPM, diameter, sampling rate, and window metadata.
- Leave-One-Load-Out test rows contain only the held-out load.
- Every test fold contains Healthy/Normal data and all task classes.
- Preprocessing and tuning occur inside each outer training fold.

The high scores therefore remain reported as CWRU benchmark separability, not assumed deployment performance.


## Explainability

Native RF/XGBoost importance and out-of-fold permutation importance are in `feature_importance_folds.csv`; the top ten per task/model are in `feature_importance_top.csv`.
