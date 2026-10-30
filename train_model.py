"""
train_model.py
---------------
Preprocessing + feature engineering + XGBoost training for the
Credit Risk Prediction System.

What this does, end to end:
  1. Loads the dataset and engineers financial ratio features.
  2. Builds a single scikit-learn Pipeline (imputation -> scaling/encoding -> XGBoost)
     so training and inference can never drift apart.
  3. Runs a RandomizedSearchCV hyperparameter search with 5-fold stratified
     cross-validation, optimizing for ROC-AUC.
  4. Refits the best pipeline on the full training set and reports both the
     cross-validated scores (mean +/- std across folds) and a final held-out
     test-set score, so you can talk about generalization honestly.
  5. Saves the trained pipeline, feature importances, and a metrics.json file
     consumed by the Streamlit app.

Run:
    python src/train_model.py
Produces:
    models/credit_risk_pipeline.joblib
    models/feature_importance.csv
    models/metrics.json
"""

import json
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from xgboost import XGBClassifier
except ImportError:
    sys.exit(
        "ERROR: xgboost is not installed in this environment.\n"
        "Fix: activate your virtual environment, then run:\n"
        "    pip install -r requirements.txt"
    )

RANDOM_SEED = 42
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "loan_data.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

NUMERIC_FEATURES = [
    "age",
    "annual_income",
    "employment_years",
    "credit_score",
    "credit_history_years",
    "num_late_payments",
    "existing_debt",
    "loan_amount",
    "loan_term_months",
    # engineered
    "debt_to_income_ratio",
    "loan_to_income_ratio",
    "monthly_installment",
    "disposable_income_ratio",
]
CATEGORICAL_FEATURES = ["employment_status", "home_ownership", "loan_purpose"]

# Hyperparameter search space for RandomizedSearchCV
PARAM_DISTRIBUTIONS = {
    "model__n_estimators": [150, 250, 350, 450, 600],
    "model__max_depth": [3, 4, 5, 6, 7],
    "model__learning_rate": [0.01, 0.03, 0.05, 0.08, 0.1],
    "model__subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "model__colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
    "model__min_child_weight": [1, 2, 3, 5, 7],
    "model__reg_lambda": [0.5, 1.0, 1.5, 2.0],
    "model__reg_alpha": [0.0, 0.1, 0.5, 1.0],
}
N_SEARCH_ITER = 25  # number of random hyperparameter combinations to try
CV_FOLDS = 5


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive risk ratios from raw financial fields.

    IMPORTANT: this function is duplicated in app.py so the app can build the
    same engineered columns at inference time. If you change this function,
    update the copy in app.py to match, or predictions will be wrong.
    """
    df = df.copy()
    monthly_income = df["annual_income"] / 12.0

    df["monthly_installment"] = df["loan_amount"] / df["loan_term_months"]
    df["debt_to_income_ratio"] = (
        (df["existing_debt"] * 0.03) + df["monthly_installment"]
    ) / (monthly_income + 1.0)
    df["loan_to_income_ratio"] = df["loan_amount"] / (df["annual_income"] + 1.0)
    df["disposable_income_ratio"] = 1.0 - df["debt_to_income_ratio"].clip(upper=2.0)

    return df


def build_pipeline(scale_pos_weight: float = 1.0) -> Pipeline:
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, NUMERIC_FEATURES),
        ("cat", categorical_transformer, CATEGORICAL_FEATURES),
    ])

    model = XGBClassifier(
        eval_metric="auc",
        random_state=RANDOM_SEED,
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight,
    )

    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def load_data() -> pd.DataFrame:
    if not os.path.exists(DATA_PATH):
        sys.exit(
            f"ERROR: dataset not found at {DATA_PATH}\n"
            "Fix: run 'python data/generate_data.py' first."
        )
    df = pd.read_csv(DATA_PATH)
    required_cols = {"default", "annual_income", "loan_amount", "loan_term_months"}
    missing = required_cols - set(df.columns)
    if missing:
        sys.exit(f"ERROR: dataset is missing required columns: {missing}")
    return df


def main():
    t0 = time.time()
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("Loading data...")
    df = load_data()
    df = engineer_features(df)

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df["default"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    neg, pos = np.bincount(y_train)
    scale_pos_weight = neg / pos
    print(f"Class balance in training set -> negative: {neg}, positive: {pos} "
          f"(scale_pos_weight={scale_pos_weight:.2f})")

    base_pipeline = build_pipeline(scale_pos_weight=scale_pos_weight)

    # ------------------------------------------------------------------
    # Hyperparameter search with stratified k-fold cross-validation
    # ------------------------------------------------------------------
    print(f"\nRunning RandomizedSearchCV: {N_SEARCH_ITER} candidates x {CV_FOLDS} folds "
          f"= {N_SEARCH_ITER * CV_FOLDS} fits. This may take a few minutes...")

    cv_splitter = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    search = RandomizedSearchCV(
        estimator=base_pipeline,
        param_distributions=PARAM_DISTRIBUTIONS,
        n_iter=N_SEARCH_ITER,
        scoring="roc_auc",
        cv=cv_splitter,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbose=1,
        refit=True,
    )
    search.fit(X_train, y_train)

    best_pipeline = search.best_estimator_
    print(f"\nBest CV ROC-AUC: {search.best_score_:.4f}")
    print(f"Best params: {json.dumps(search.best_params_, indent=2)}")

    # ------------------------------------------------------------------
    # Cross-validated metrics for the *best* pipeline (mean +/- std),
    # so we can report robust numbers rather than a single lucky split.
    # ------------------------------------------------------------------
    print("\nComputing 5-fold cross-validated metrics for the best pipeline...")
    cv_results = cross_validate(
        best_pipeline, X_train, y_train,
        cv=cv_splitter,
        scoring=["roc_auc", "accuracy", "f1"],
        n_jobs=-1,
    )
    cv_metrics = {
        "roc_auc_mean": float(np.mean(cv_results["test_roc_auc"])),
        "roc_auc_std": float(np.std(cv_results["test_roc_auc"])),
        "accuracy_mean": float(np.mean(cv_results["test_accuracy"])),
        "accuracy_std": float(np.std(cv_results["test_accuracy"])),
        "f1_mean": float(np.mean(cv_results["test_f1"])),
        "f1_std": float(np.std(cv_results["test_f1"])),
    }
    print("Cross-validated metrics (train folds):")
    print(json.dumps(cv_metrics, indent=2))

    # ------------------------------------------------------------------
    # Final held-out test set evaluation (data the model never saw
    # during search or cross-validation)
    # ------------------------------------------------------------------
    y_pred = best_pipeline.predict(X_test)
    y_proba = best_pipeline.predict_proba(X_test)[:, 1]

    test_metrics = {
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_score": float(f1_score(y_test, y_pred)),
    }
    print("\nHeld-out test set metrics:")
    print(json.dumps(test_metrics, indent=2))
    print("\nClassification report (test set):\n", classification_report(y_test, y_pred))

    metrics = {
        **test_metrics,
        "cv": cv_metrics,
        "cv_folds": CV_FOLDS,
        "best_params": search.best_params_,
        "training_seconds": round(time.time() - t0, 1),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
    }
    with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # ------------------------------------------------------------------
    # Feature importance (mapped back to readable names)
    # ------------------------------------------------------------------
    ohe = best_pipeline.named_steps["preprocessor"].named_transformers_["cat"].named_steps["onehot"]
    cat_names = list(ohe.get_feature_names_out(CATEGORICAL_FEATURES))
    all_feature_names = NUMERIC_FEATURES + cat_names
    importances = best_pipeline.named_steps["model"].feature_importances_
    fi_df = pd.DataFrame({"feature": all_feature_names, "importance": importances})
    fi_df = fi_df.sort_values("importance", ascending=False)
    fi_df.to_csv(os.path.join(MODEL_DIR, "feature_importance.csv"), index=False)
    print("\nTop features:\n", fi_df.head(10).to_string(index=False))

    joblib.dump(best_pipeline, os.path.join(MODEL_DIR, "credit_risk_pipeline.joblib"))
    print(f"\nSaved tuned pipeline to {MODEL_DIR}/credit_risk_pipeline.joblib")
    print(f"Total time: {round(time.time() - t0, 1)}s")


if __name__ == "__main__":
    main()
