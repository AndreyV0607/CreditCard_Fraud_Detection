# Credit Card Fraud Detection

Credit card fraud detection project using a highly imbalanced dataset. The workflow is split across separate notebooks so the exploratory analysis, model selection, tuning process, and final model are easy to follow and reproduce.

## Project Objective

The main goal is not to maximize `accuracy`, because fraud is an extremely rare class. In this problem, a model can achieve very high accuracy simply by predicting almost every transaction as "No Fraud".

For that reason, this project focuses on metrics that are more useful for fraud detection:

- Fraud `Recall`: how many real fraud cases the model detects.
- `False Negatives`: fraud cases the model misses.
- `False Positives`: legitimate transactions flagged as suspicious.
- Fraud `Precision`: how reliable the fraud alerts are.
- Fraud `F1 Score`: balance between precision and recall.
- `PR-AUC`: more informative than ROC-AUC when the positive class is very rare.

The final model decision is framed as a business problem: reduce the risk of letting fraud pass through, while avoiding an unmanageable number of false alerts.

## Project Structure

```text
.
├── data/
│   └── creditcard.csv
├── models/
│   └── xgboost_original_recall95_f1/
│       ├── metadata.json
│       ├── xgboost_original_recall95_f1.joblib
│       └── xgboost_original_recall95_f1.json
└── notebooks/
    ├── eda.ipynb
    ├── modes_training.ipynb
    └── xgboost_final_original_recall90.ipynb
```

Note: the model folder keeps the name `xgboost_original_recall95_f1` from an earlier experiment, but the currently saved final model uses `target_recall = 0.90`, as recorded in `models/xgboost_original_recall95_f1/metadata.json`.

## Dataset

The dataset contains:

- `284,807` transactions.
- `31` columns.
- Anonymized variables `V1` through `V28`.
- Original variables `Time`, `Amount`, and `Class`.
- `Class = 1` represents fraud.
- `Class = 0` represents a legitimate transaction.

The EDA confirms that the dataset is heavily imbalanced. This shapes the entire modeling strategy: global metrics such as accuracy are not enough to evaluate the model.

## Notebooks

### 1. `notebooks/eda.ipynb`

Contains the exploratory data analysis:

- Loading `data/creditcard.csv`.
- Reviewing data types, missing values, and dataset dimensions.
- Inspecting the target distribution.
- Correlation matrix.
- Correlations with `Class`.
- Creating an initial reduced dataset by dropping columns such as `Time`, `Amount`, `V28`, `V27`, `V26`, `V25`, `V24`, `V23`, and `V22`.

This notebook is used to understand the problem, especially the extreme imbalance between fraud and non-fraud transactions.

### 2. `notebooks/modes_training.ipynb`

Contains model experimentation and comparison:

- Baseline neural network.
- Tuned neural network using Keras Tuner.
- Random Forest with `class_weight="balanced"`.
- XGBoost with `scale_pos_weight`.
- Threshold tuning using precision-recall curves.
- Class-level metric comparison.
- Experiments targeting high fraud recall.

Representative observed results:

| Model / experiment | TP | FN | FP | Fraud recall | Fraud precision | Comment |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Tuned Neural Network, threshold 0.5 | 80 | 18 | approx. 18 | 81.6% | 82.0% | Good precision, but misses too many fraud cases. |
| Balanced Random Forest | 87 | 11 | 217 | 88.8% | 28.6% | Better recall than conservative models, with relatively few FP. |
| Baseline XGBoost | 80 | 18 | 148 | 81.6% | 35.1% | Reasonable precision, but recall is not enough for fraud. |
| Neural Network with oversampling and aggressive threshold | 96 | 2 | 27,357 | 98.0% | 0.35% | Detects almost all fraud, but creates too many false alerts. |
| XGBoost near 95% recall target | 93 | 5 | 6,361 | 94.9% | 1.44% | Reduces missed fraud, but still has a high operational cost. |
| Final XGBoost with 90% minimum recall | 91 | 7 | 1,415 | 92.9% | 6.04% | Better balance between risk and review workload. |

The main lesson is that increasing recall without controlling false positives can make the model operationally impractical. For that reason, the final threshold is selected using recall, fraud F1, PR-AUC, and the number of false positives.

### 3. `notebooks/xgboost_final_original_recall90.ipynb`

Clean notebook for the final model:

- Uses the full original dataset, not the reduced dataset.
- Creates stratified train, validation, and test splits.
- Trains XGBoost candidates with different `scale_pos_weight` multipliers.
- Evaluates each candidate with `PR-AUC`.
- Searches for thresholds that keep `recall_fraud >= 0.90`.
- Among those thresholds, selects the one that maximizes fraud `F1`.
- In case of ties, favors fewer false positives.
- Evaluates the selected model on the test set.
- Saves the model, threshold, features, parameters, and metrics.

Split used:

| Split | Rows | Fraud | Non-fraud |
| --- | ---: | ---: | ---: |
| Train | 182,276 | 315 | 181,961 |
| Validation | 45,569 | 79 | 45,490 |
| Test | 56,962 | 98 | 56,864 |

## Final Model

The final model is an `XGBClassifier` trained on the original dataset.

Main parameters:

```python
{
    "n_estimators": 700,
    "learning_rate": 0.03,
    "max_depth": 4,
    "min_child_weight": 3,
    "subsample": 0.80,
    "colsample_bytree": 0.80,
    "reg_lambda": 3.0,
    "reg_alpha": 0.10,
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    "random_state": 42,
    "scale_pos_weight": 4621.2317
}
```

Selected configuration:

- Model: `spw_x8`.
- Threshold: `0.007152053993195295`.
- Target recall used: `0.90`.
- Validation PR-AUC: `0.8202`.
- Test PR-AUC: `0.8667`.

Test results:

| Metric | Value |
| --- | ---: |
| Fraud precision | 6.04% |
| Fraud recall | 92.86% |
| Fraud F1 | 11.35% |
| PR-AUC | 0.8667 |
| True Positives | 91 |
| False Negatives | 7 |
| False Positives | 1,415 |
| True Negatives | 55,449 |

Test confusion matrix:

|  | Pred No Fraud | Pred Fraud |
| --- | ---: | ---: |
| Actual No Fraud | 55,449 | 1,415 |
| Actual Fraud | 7 | 91 |

## Why This Model Was Selected

The final model was not selected because it had the highest accuracy. It was selected because it offers the best observed compromise between:

- Keeping fraud recall high.
- Reducing missed fraud cases.
- Keeping false positives from becoming operationally unmanageable.
- Using PR-AUC and fraud F1 as criteria better aligned with an imbalanced dataset.
- Allowing the decision threshold to be adjusted according to business risk tolerance.

During experimentation, some models achieved higher recall, but at the cost of many more false positives. For example, the neural network with a very aggressive threshold detected 96 out of 98 fraud cases in the test set, but generated 27,357 false positives. That result may only be acceptable if the client prioritizes minimizing missed fraud almost exclusively and has the operational capacity to review a very large number of alerts.

The XGBoost experiment near 95% recall detected 93 out of 98 fraud cases, but produced 6,361 false positives. The final model detects 91 out of 98 fraud cases and reduces false positives to 1,415. In other words, it accepts 2 additional missed fraud cases compared with the 95% recall experiment, but removes thousands of false alerts.

## Client Flexibility to Adjust Risk

The model outputs probabilities. The final classification depends on the selected threshold. This gives the client flexibility to define a risk policy.

If the client wants to minimize fraud risk as much as possible:

- Lower the threshold.
- Recall increases.
- False negatives decrease.
- False positives increase.
- The alert review workload increases.

If the client wants to reduce operational friction:

- Raise the threshold.
- False positives decrease.
- Precision increases.
- Recall may decrease.
- Some additional fraud cases may pass undetected.

Observed tradeoff example:

| Configuration | TP | FN | FP | Fraud recall | Interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| Aggressive XGBoost near 95% recall | 93 | 5 | 6,361 | 94.9% | Fewer missed fraud cases, many false alerts. |
| Final XGBoost with 90% recall target | 91 | 7 | 1,415 | 92.9% | Fewer false alerts while still keeping recall high. |
| Very aggressive Neural Network | 96 | 2 | 27,357 | 98.0% | Maximum protection, very high operational cost. |

For this reason, the project does not treat one threshold as an absolute truth. It provides a base model and a way to select thresholds according to the relative cost of:

- An undetected fraud case.
- A legitimate transaction sent to review.
- The capacity of the fraud review team.
- The client's risk appetite.

## Saved Artifacts

The final model is saved in:

```text
models/xgboost_original_recall95_f1/
├── xgboost_original_recall95_f1.joblib
├── xgboost_original_recall95_f1.json
└── metadata.json
```

`xgboost_original_recall95_f1.joblib` contains:

- Trained model.
- Selected threshold.
- Feature list.
- Experiment metadata.

`xgboost_original_recall95_f1.json` contains the native XGBoost model.

`metadata.json` contains parameters, validation metrics, test metrics, threshold, and the split configuration.

## How to Run

Activate the virtual environment:

```bash
source .venv/bin/activate
```

Run the notebooks in this order:

1. `notebooks/eda.ipynb`
2. `notebooks/modes_training.ipynb`
3. `notebooks/xgboost_final_original_recall90.ipynb`

The final notebook can be run independently as long as `data/creditcard.csv` exists, because it reloads the data, recreates the splits, trains the final model, and saves the artifacts.

## Main Dependencies

The project mainly uses:

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `xgboost`
- `joblib`
- `tensorflow`
- `keras-tuner`
- `imbalanced-learn`

## Final Considerations

This project should be interpreted as a decision-support system, not as a fully automated final decision without review. In fraud detection, the threshold is a business policy as much as a technical choice.

The current final model favors a reasonable balance: it keeps fraud recall high and significantly reduces false positives compared with more aggressive variants. However, if the client has lower risk tolerance, a lower-threshold variant can be used to detect more fraud cases while accepting more false positives.
