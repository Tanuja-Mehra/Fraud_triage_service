"""prepare.py - Part A: Data Foundation (Audit, Clean, Engineer)."""

import pandas as pd
import numpy as np

KNOWN_CATEGORIES = [
    "Grocery",
    "Electronics",
    "Travel",
    "Food Delivery",
    "Fashion",
    "Utilities",
]
KNOWN_CITIES = ["Pune", "Mumbai", "Delhi", "Bengaluru", "Chennai", "Kolkata"]
KNOWN_MODES = ["UPI", "Card", "NetBanking", "Wallet"]


def audit_transactions(raw_df: pd.DataFrame) -> dict:
    """Audits the raw transaction dataset against specification requirements."""
    initial_rows, initial_cols = raw_df.shape
    exact_duplicates = int(raw_df.duplicated().sum())

    deduped = raw_df.drop_duplicates()
    missing_amount = int(deduped["amount_inr"].isna().sum())
    missing_distance = int(deduped["distance_from_home_km"].isna().sum())

    valid_amounts = pd.to_numeric(deduped["amount_inr"], errors="coerce")
    out_of_range_amount = int(
        ((valid_amounts < 1.0) | (valid_amounts > 1000000.0)).sum()
    )

    cat_spellings = int(raw_df["merchant_category"].nunique())
    city_spellings = int(raw_df["city"].nunique())

    return {
        "initial_rows": initial_rows,
        "initial_cols": initial_cols,
        "exact_duplicates": exact_duplicates,
        "post_dedup_rows": len(deduped),
        "missing_amount": missing_amount,
        "missing_distance": missing_distance,
        "out_of_range_amount": out_of_range_amount,
        "merchant_category_spellings": cat_spellings,
        "city_spellings": city_spellings,
    }


def clean_transactions(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Cleans the raw dataset without modifying input or filling missing distances."""
    df = raw_df.copy()

    # Rule (a): Drop exact duplicate rows
    df = df.drop_duplicates()

    # Rule (b): Strip surrounding spaces and convert to title case
    df["merchant_category"] = (
        df["merchant_category"].astype(str).str.strip().str.title()
    )
    df["city"] = df["city"].astype(str).str.strip().str.title()

    # Ensure amount_inr is numeric
    df["amount_inr"] = pd.to_numeric(df["amount_inr"], errors="coerce")

    # Rule (c): Drop rows where amount_inr is missing
    df = df.dropna(subset=["amount_inr"])

    # Rule (d): Drop rows where amount_inr is below 1 or above 1,000,000
    df = df[(df["amount_inr"] >= 1.0) & (df["amount_inr"] <= 1000000.0)]

    # Rule (e): Reset index
    df = df.reset_index(drop=True)

    # Cast types
    df["distance_from_home_km"] = pd.to_numeric(
        df["distance_from_home_km"], errors="coerce"
    )
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Computes engineered features for single-row or batch DataFrames."""
    res = df.copy()
    res["amount_vs_avg_ratio"] = (
        res["amount_inr"] / res["avg_amount_30d_inr"]
    ).round(4)
    res["is_night"] = (res["hour_of_day"] <= 5).astype(int)
    res["high_value"] = (res["amount_inr"] > 50000.0).astype(int)
    return res


if __name__ == "__main__":
    raw = pd.read_csv("transactions_raw.csv")
    audit = audit_transactions(raw)

    print("=== PART A: AUDIT RESULTS ===")
    for k, v in audit.items():
        print(f"  {k}: {v}")

    cleaned = clean_transactions(raw)
    featured = add_features(cleaned)

    print("\n=== PART A: CLEAN & FEATURE CHECKPOINTS ===")
    print(f"Cleaned rows: {len(cleaned)}")
    print(f"Merchant categories:\n{cleaned['merchant_category'].value_counts()}")
    print(f"Payment modes:\n{cleaned['payment_mode'].value_counts()}")
    print(
        f"Min amount: {cleaned['amount_inr'].min():.2f}, Max amount: {cleaned['amount_inr'].max():.2f}"
    )
    print(
        f"Missing distance_from_home_km in clean: {cleaned['distance_from_home_km'].isna().sum()}"
    )
    print(f"Sum is_night: {featured['is_night'].sum()}")
    print(f"Sum high_value: {featured['high_value'].sum()}")
    print(f"Max amount_vs_avg_ratio: {featured['amount_vs_avg_ratio'].max():.2f}")

    total_fraud = int(cleaned["is_fraud"].sum())
    fraud_rate = (total_fraud / len(cleaned)) * 100.0
    print(f"\nFraud count: {total_fraud} / {len(cleaned)} ({fraud_rate:.2f}%)")

    # Baseline: Predict all 0
    accuracy = ((cleaned["is_fraud"] == 0).sum() / len(cleaned)) * 100.0
    print(f"Never-Fraud Baseline Accuracy: {accuracy:.2f}%, Recall: 0.00%")

    # Leakage Audit Table
    cb_grp = cleaned.groupby("chargeback_filed")["is_fraud"].agg(
        total="count", fraud="sum"
    )
    print("\nLeakage Audit (chargeback_filed vs is_fraud):")
    print(cb_grp)