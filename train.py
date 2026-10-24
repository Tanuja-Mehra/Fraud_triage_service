"""train.py - Part B: Model Training, Class Imbalance Handling, Threshold Tuning."""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

from prepare import clean_transactions, add_features


def get_feature_matrix_and_target():
    raw_df = pd.read_csv("transactions_raw.csv")
    cleaned = clean_transactions(raw_df)
    featured = add_features(cleaned)

    y = featured["is_fraud"].astype(int)
    X_raw = featured.drop(
        columns=["txn_id", "customer_id", "chargeback_filed", "is_fraud"]
    )

    # One-hot encoding with fixed columns
    X = pd.get_dummies(X_raw, dtype=int)
    return X, y, featured["chargeback_filed"]


def build_final_model():
    X, y, _ = get_feature_matrix_and_target()
    columns_list = list(X.columns)

    # 60/20/20 stratified split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )

    # Median imputation from training split only
    train_median = float(X_train["distance_from_home_km"].median())
    X_train_filled = X_train.copy()
    X_val_filled = X_val.copy()

    X_train_filled["distance_from_home_km"] = X_train_filled[
        "distance_from_home_km"
    ].fillna(train_median)
    X_val_filled["distance_from_home_km"] = X_val_filled[
        "distance_from_home_km"
    ].fillna(train_median)

    # Grid Search tuning
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rf_base = RandomForestClassifier(
        n_estimators=200, class_weight="balanced", random_state=42
    )
    param_grid = {
        "max_depth": [4, 6, 8, None],
        "min_samples_leaf": [5, 10, 20],
    }
    grid = GridSearchCV(
        rf_base, param_grid, cv=cv, scoring="f1", n_jobs=-1
    )
    grid.fit(X_train_filled, y_train)

    best_rf = grid.best_estimator_

    # Threshold selection on validation set: highest t in [0.1..0.9] with recall >= 0.65
    val_probs = best_rf.predict_proba(X_val_filled)[:, 1]
    t_star = 0.1
    for t in [round(0.1 * i, 1) for i in range(9, 0, -1)]:
        preds = (val_probs >= t).astype(int)
        rec = recall_score(y_val, preds, zero_division=0)
        if rec >= 0.65:
            t_star = t
            break

    return {
        "model": best_rf,
        "columns": columns_list,
        "median": train_median,
        "threshold": t_star,
    }


if __name__ == "__main__":
    X, y, chargeback = get_feature_matrix_and_target()
    columns_list = list(X.columns)
    print(f"Input columns ({len(columns_list)}): {columns_list}\n")

    # Splits
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )

    train_median = float(X_train["distance_from_home_km"].median())
    print(
        f"Splits: Train={len(X_train)} (fraud={y_train.sum()}), Val={len(X_val)} (fraud={y_val.sum()}), Test={len(X_test)} (fraud={y_test.sum()})"
    )
    print(f"Training median distance_from_home_km: {train_median:.2f}\n")

    X_train_imp = X_train.copy()
    X_val_imp = X_val.copy()
    X_test_imp = X_test.copy()
    X_train_imp["distance_from_home_km"] = X_train_imp[
        "distance_from_home_km"
    ].fillna(train_median)
    X_val_imp["distance_from_home_km"] = X_val_imp[
        "distance_from_home_km"
    ].fillna(train_median)
    X_test_imp["distance_from_home_km"] = X_test_imp[
        "distance_from_home_km"
    ].fillna(train_median)

    # 1. Three Imbalance Treatments on Validation Split
    print("=== THREE IMBALANCE VARIANTS (Logistic Regression, Threshold 0.5) ===")

    # (i) Plain
    p_plain = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )
    p_plain.fit(X_train_imp, y_train)
    y_val_plain = (p_plain.predict_proba(X_val_imp)[:, 1] >= 0.5).astype(int)
    print(
        f"Plain: Prec={precision_score(y_val, y_val_plain):.3f}, Rec={recall_score(y_val, y_val_plain):.3f}, F1={f1_score(y_val, y_val_plain):.3f}"
    )
    print(f"Confusion Matrix:\n{confusion_matrix(y_val, y_val_plain)}")

    # (ii) Balanced
    p_bal = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "lr",
                LogisticRegression(
                    max_iter=1000, class_weight="balanced", random_state=42
                ),
            ),
        ]
    )
    p_bal.fit(X_train_imp, y_train)
    y_val_bal = (p_bal.predict_proba(X_val_imp)[:, 1] >= 0.5).astype(int)
    print(
        f"\nBalanced: Prec={precision_score(y_val, y_val_bal):.3f}, Rec={recall_score(y_val, y_val_bal):.3f}, F1={f1_score(y_val, y_val_bal):.3f}"
    )
    print(f"Confusion Matrix:\n{confusion_matrix(y_val, y_val_bal)}")

    # (iii) Oversampled
    train_full = X_train_imp.copy()
    train_full["is_fraud"] = y_train
    non_fraud = train_full[train_full["is_fraud"] == 0]
    fraud = train_full[train_full["is_fraud"] == 1]
    fraud_oversampled = fraud.sample(
        n=len(non_fraud), replace=True, random_state=42
    )
    stacked = pd.concat([non_fraud, fraud_oversampled], axis=0)

    X_train_os = stacked.drop(columns=["is_fraud"])
    y_train_os = stacked["is_fraud"]

    p_os = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )
    p_os.fit(X_train_os, y_train_os)
    y_val_os = (p_os.predict_proba(X_val_imp)[:, 1] >= 0.5).astype(int)
    print(f"\nOversampled Training Table Rows: {len(stacked)}")
    print(
        f"Oversampled: Prec={precision_score(y_val, y_val_os):.3f}, Rec={recall_score(y_val, y_val_os):.3f}, F1={f1_score(y_val, y_val_os):.3f}"
    )
    print(f"Confusion Matrix:\n{confusion_matrix(y_val, y_val_os)}")

    # 2. Cross-Validation and Tuning
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Untuned RF
    rf_untuned = RandomForestClassifier(
        n_estimators=200, class_weight="balanced", random_state=42
    )
    cv_scores_rf = [
        f1_score(y_train.iloc[val_idx], rf_untuned.fit(X_train_imp.iloc[trn_idx], y_train.iloc[trn_idx]).predict(X_train_imp.iloc[val_idx]))
        for trn_idx, val_idx in cv.split(X_train_imp, y_train)
    ]
    print(
        f"\nUntuned RF 5-Fold CV F1: Mean={np.mean(cv_scores_rf):.3f}, Std={np.std(cv_scores_rf):.3f}"
    )

    # Grid search
    grid = GridSearchCV(
        rf_untuned,
        {"max_depth": [4, 6, 8, None], "min_samples_leaf": [5, 10, 20]},
        cv=cv,
        scoring="f1",
        n_jobs=-1,
    )
    grid.fit(X_train_imp, y_train)
    print(
        f"Best Tuned RF: Params={grid.best_params_}, Mean CV F1={grid.best_score_:.3f}"
    )

    best_rf = grid.best_estimator_

    # 3. Validation Threshold Table
    val_probs = best_rf.predict_proba(X_val_imp)[:, 1]
    print("\n=== VALIDATION THRESHOLD SWEEP ===")
    t_star = 0.1
    for t in [round(0.1 * i, 1) for i in range(1, 10)]:
        p_t = (val_probs >= t).astype(int)
        flagged_count = int(p_t.sum())
        prec = precision_score(y_val, p_t, zero_division=0)
        rec = recall_score(y_val, p_t, zero_division=0)
        f1 = f1_score(y_val, p_t, zero_division=0)
        print(
            f"Threshold {t:.1f}: Flagged={flagged_count:4d}, Prec={prec:.3f}, Rec={rec:.3f}, F1={f1:.3f}"
        )

    for t in [round(0.1 * i, 1) for i in range(9, 0, -1)]:
        if recall_score(y_val, (val_probs >= t).astype(int)) >= 0.65:
            t_star = t
            break
    print(f"\nChosen t_star: {t_star}")

    # 4. Final Evaluation on Test Set at t_star
    test_probs = best_rf.predict_proba(X_test_imp)[:, 1]
    test_preds = (test_probs >= t_star).astype(int)
    print(f"\n=== TEST SPLIT EVALUATION (t_star = {t_star}) ===")
    print(f"Flagged Payments: {int(test_preds.sum())}")
    print(f"Precision: {precision_score(y_test, test_preds):.3f}")
    print(f"Recall:    {recall_score(y_test, test_preds):.3f}")
    print(f"F1 Score:  {f1_score(y_test, test_preds):.3f}")
    cm_test = confusion_matrix(y_test, test_preds)
    print(f"Confusion Matrix:\n{cm_test}")

    # 5. Leakage Experiment
    X_train_leaky = X_train_imp.copy()
    X_val_leaky = X_val_imp.copy()
    X_train_leaky["chargeback_filed"] = chargeback.loc[X_train_leaky.index]
    X_val_leaky["chargeback_filed"] = chargeback.loc[X_val_leaky.index]

    rf_leaky = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        **grid.best_params_,
    )
    rf_leaky.fit(X_train_leaky, y_train)

    clean_val_f1_05 = f1_score(y_val, (val_probs >= 0.5).astype(int))
    leaky_probs = rf_leaky.predict_proba(X_val_leaky)[:, 1]
    leaky_val_f1_05 = f1_score(y_val, (leaky_probs >= 0.5).astype(int))
    print(f"\nLeakage Gap at 0.5: Clean F1={clean_val_f1_05:.3f}, Leaky F1={leaky_val_f1_05:.3f}")

    # 6. Feature Importances
    importances = best_rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    print("\nTop 5 Features:")
    for idx in indices[:5]:
        print(f"  {columns_list[idx]}: {importances[idx]:.4f}")