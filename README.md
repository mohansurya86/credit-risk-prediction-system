# Credit Risk Prediction System
A Streamlit web application that uses an XGBoost model to predict the probability
that a loan applicant will default, based on their income, loan details, and
credit history.

## Features
- Synthetic data generator (data/generate_data.py) — produces a realistic
  8,000-row loan applicant dataset with a genuinely learnable (not deterministic)
  default outcome.
- Feature engineering — derives debt-to-income ratio, loan-to-income ratio,
  monthly installment burden, and disposable income ratio from raw financial
  fields, in addition to using the raw fields themselves.
- Preprocessing pipeline — median/most-frequent imputation, standard scaling
  for numeric features, one-hot encoding for categoricals, all wrapped in a
  single scikit-learn Pipeline so training and inference always match.
- Hyperparameter tuning + cross-validation — RandomizedSearchCV searches
  25 hyperparameter combinations with 5-fold stratified cross-validation,
  optimizing ROC-AUC. Both cross-validated scores (mean ± std across folds) and
  a final held-out test-set score are saved and shown in the app, so performance
  numbers are robust rather than a single lucky split.
- XGBoost classifier — tuned for class imbalance (~14% default rate) via
  scale_pos_weight.
- Streamlit app (app.py) — interactive form to enter an applicant profile
  and get a default probability, animated gauge + radar risk chart, downloadable
  PDF risk report, and a Model Insights tab with test + cross-validated metrics
  and feature importance.
- Share & Deploy tab — in-app instructions for publishing the app to
  Streamlit Community Cloud so you get a real public URL.
- Deployment-ready settings — .streamlit/config.toml (native dark theme)
  and .gitignore included.

## Project Structure

credit_risk_app/
├── app.py                    # Streamlit application
├── requirements.txt
├── README.md
├── .streamlit/
│   └── config.toml           # native Streamlit theme settings
├── .gitignore
├── data/
│   ├── generate_data.py      # synthetic dataset generator
│   └── loan_data.csv         # generated dataset (created by running the script)
├── src/
│   └── train_model.py        # preprocessing + feature engineering + tuned XGBoost training
└── models/                   # created by train_model.py
    ├── credit_risk_pipeline.joblib
    ├── feature_importance.csv
    └── metrics.json

## Setup

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

## Usage

1. Generate the dataset:
   python data/generate_data.py

2. Train the model:
   python src/train_model.py
   This runs a hyperparameter search (25 candidates × 5 folds = 125 fits), so it
   takes a few minutes. It prints cross-validated and held-out test-set ROC-AUC /
   accuracy / F1, and saves the trained pipeline, feature importances, and
   metrics to models/.

3. Launch the app:
   streamlit run app.py

4. Open the URL Streamlit prints (typically http://localhost:8501), fill in
   an applicant profile, click "Predict Default Risk", and optionally
   download the PDF risk report.

## Getting a shareable public link

The app includes a Share & Deploy tab with step-by-step instructions for
publishing to Streamlit Community Cloud (https://share.streamlit.io), which is
free and gives you a real URL like https://your-app-name.streamlit.app instead
of only localhost. In short: push the project to a public GitHub repo
(including models/, .streamlit/config.toml, and requirements.txt), then
connect that repo on share.streamlit.io and point it at app.py.

## Notes on the dataset

This project ships with a synthetic dataset so it runs end-to-end with no
external downloads. The generator encodes realistic risk relationships (higher
debt-to-income and loan-to-income ratios, more late payments, unemployment, and
lower credit scores all increase default probability) plus random noise, so the
model has to actually learn rather than memorize a deterministic rule.

To use real data instead, replace data/loan_data.csv with your own dataset
containing the same columns (see NUMERIC_FEATURES / CATEGORICAL_FEATURES in
src/train_model.py), or adjust those lists to match your columns.

## Retraining with your own data

If your column names differ, update NUMERIC_FEATURES, CATEGORICAL_FEATURES,
and engineer_features() in both src/train_model.py and app.py (the
app's engineer_features() must mirror the training script's exactly, since
it builds the same engineered columns before calling pipeline.predict_proba).

## Author

Meesala Mohan Surya Sai
Intern ID: CITS9842
```
