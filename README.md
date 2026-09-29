Multi-class breast cancer severity classification with an Artificial Neural Network (Keras / TensorFlow).

Predicts one of five outcomes for each patient: 'No Cancer', 'Stage I', 'Stage II', 'Stage III', 'Stage IV', using the BCRSD tabular dataset (30,000 patients, 47 columns).

Project Description

The project walks through a complete, professional machine-learning workflow on tabular clinical data:

1. Data audit (real vs. fake missing values, duplicates, class balance)
2. Target-leakage detection, the most important step in this project
3. Cleaning and feature engineering
4. Leak-free preprocessing (split first, then impute / scale on the training set only)
5. ANN training with regularization, class weights and early stopping
6. Evaluation with per-class metrics, confusion matrix and permutation feature importance

A central finding is that several columns (mammography BI-RADS, ultrasound, MRI, tumor descriptors, 'Breast_Cancer_Risk') reveal the answer almost directly. The project therefore evaluates the model under three feature scenarios to show honest vs. inflated performance.

Technologies Used
1. Language: Python 3.10+
2. Deep learning: TensorFlow / Keras
3. Data handling: pandas, NumPy
4. Preprocessing and metrics: scikit-learn
5. Visualization: Matplotlib, Seaborn

Dataset
1. 30,000 rows and 47 columns, one row per patient
2. Target: 'Cancer_Severity' (No Cancer 41%, Stage I 23%, Stage II 17%, Stage III 12%, Stage IV 7%)
3. Feature groups: demographics, reproductive and medical history, lifestyle, hormones, serum markers, imaging findings, pathology / biomarkers
4. Note: the text "None" is a real category (e.g. 'Alcohol_Intake', 'Calcification'), so the file must be read with 'keep_default_na=False'
5. The value ranges are unusually clean, which suggests the data is synthetic
6. Link: https://www.kaggle.com/datasets/gowtha69/breast-cancer-risk-and-severity-dataset

Model architecture
Input → Dense(64) → Dropout(0.3) → Dense(32) → Dropout(0.3) → Dense(16) → Dense(5, softmax)
Adam optimizer, categorical cross-entropy, class weights, 'EarlyStopping' and 'ReduceLROnPlateau'.

Results (held-out test set)
| Feature scenario | Features used | Accuracy | Macro-F1 |
|---|---|---|---|
| 'full' | Everything except ID and leakage columns | 100% | 1.000 |
| 'no_imaging' | Removes BI-RADS, ultrasound, MRI, tumor descriptors | 99.7% | 0.996 |
| 'risk_only' | History, lifestyle and hormones only | 48.6% | 0.435 |

The majority-class baseline is 41%. The 'risk_only' result is the most realistic measure of difficulty. Top drivers there: family history, breast density, BRCA mutation, hormone replacement therapy, age and estrogen level.

'Mammography_BIRADS', 'Ultrasound_Result', 'MRI_Result', 'Tumor_*', 'Lymph_Node_Involvement', 'Breast_Cancer_Risk' (and, to a lesser degree, pathology and tumor-marker columns).

Possible Improvements
1. Add 5-fold stratified cross-validation
2. Compare with gradient boosting and logistic regression baselines
3. Use ordinal-aware loss or metrics (stages are ordered)
4. Tune hyperparameters (Keras Tuner / Optuna)
5. Add model explainability with SHAP