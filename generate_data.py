"""
generate_data.py
-----------------
Creates a synthetic loan-applicant dataset for the Credit Risk Prediction System.

The target `default` (1 = borrower defaults, 0 = borrower repays) is generated
from a logistic function of realistic risk drivers plus random noise, so the
relationships are learnable but not deterministic -- similar to real credit data.

Run:
    python data/generate_data.py
Produces:
    data/loan_data.csv
"""

import numpy as np
import pandas as pd

RANDOM_SEED = 42


def generate_loan_data(n_samples: int = 8000, seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    age = rng.integers(21, 66, n_samples)
    income = rng.lognormal(mean=10.9, sigma=0.45, size=n_samples).clip(12000, 400000)
    employment_years = np.clip(rng.exponential(scale=5, size=n_samples), 0, 40)
    credit_score = rng.normal(650, 90, n_samples).clip(300, 850)
    credit_history_years = np.clip((age - 21) * rng.uniform(0.3, 0.9, n_samples), 0, 45)
    num_late_payments = rng.poisson(lam=np.clip((750 - credit_score) / 120, 0, None))
    existing_debt = rng.lognormal(mean=8.5, sigma=0.8, size=n_samples).clip(0, 200000)

    loan_amount = rng.lognormal(mean=9.3, sigma=0.55, size=n_samples).clip(1000, 100000)
    loan_term_months = rng.choice([12, 24, 36, 48, 60, 84], size=n_samples,
                                   p=[0.10, 0.15, 0.30, 0.20, 0.15, 0.10])

    home_ownership = rng.choice(["RENT", "MORTGAGE", "OWN"], size=n_samples, p=[0.42, 0.40, 0.18])
    employment_status = rng.choice(
        ["EMPLOYED", "SELF_EMPLOYED", "UNEMPLOYED"], size=n_samples, p=[0.75, 0.18, 0.07]
    )
    purpose = rng.choice(
        ["DEBT_CONSOLIDATION", "EDUCATION", "MEDICAL", "HOME_IMPROVEMENT", "BUSINESS", "OTHER"],
        size=n_samples, p=[0.32, 0.12, 0.10, 0.18, 0.13, 0.15],
    )

    # engineered-style ratios used to *drive* the true risk (feature engineering
    # is repeated on the model side so it also learns from raw columns)
    dti = (existing_debt * 0.03 + loan_amount / loan_term_months) / (income / 12 + 1)
    loan_to_income = loan_amount / (income + 1)

    # true (latent) log-odds of default
    logit = (
        -3.2
        + 3.0 * dti
        + 1.8 * loan_to_income
        - 0.012 * (credit_score - 650)
        - 0.05 * employment_years
        + 0.18 * num_late_payments
        - 0.02 * credit_history_years
        + np.where(employment_status == "UNEMPLOYED", 1.1, 0.0)
        + np.where(employment_status == "SELF_EMPLOYED", 0.25, 0.0)
        + np.where(home_ownership == "RENT", 0.25, 0.0)
        + np.where(purpose == "BUSINESS", 0.2, 0.0)
        - 0.10 * (age > 45).astype(float)
    )
    logit += rng.normal(0, 0.6, n_samples)  # noise so the problem isn't trivial
    prob_default = 1 / (1 + np.exp(-logit))
    default = rng.binomial(1, prob_default)

    df = pd.DataFrame({
        "age": age,
        "annual_income": income.round(2),
        "employment_years": employment_years.round(1),
        "employment_status": employment_status,
        "home_ownership": home_ownership,
        "credit_score": credit_score.round(0).astype(int),
        "credit_history_years": credit_history_years.round(1),
        "num_late_payments": num_late_payments,
        "existing_debt": existing_debt.round(2),
        "loan_amount": loan_amount.round(2),
        "loan_term_months": loan_term_months,
        "loan_purpose": purpose,
        "default": default,
    })

    # sprinkle a few realistic missing values to exercise preprocessing
    for col in ["employment_years", "credit_history_years", "existing_debt"]:
        missing_idx = rng.choice(df.index, size=int(0.02 * n_samples), replace=False)
        df.loc[missing_idx, col] = np.nan

    return df


if __name__ == "__main__":
    df = generate_loan_data()
    out_path = "data/loan_data.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df):,} rows to {out_path}")
    print(f"Default rate: {df['default'].mean():.2%}")
    print(df.head())
