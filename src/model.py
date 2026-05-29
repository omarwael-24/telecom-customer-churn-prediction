import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score


def train_and_evaluate_churn(df: pd.DataFrame):
    print("--- Starting Advanced Random Forest Pipeline ---")

    if 'churn' not in df.columns:
        raise KeyError("Target column 'churn' not found in the dataframe!")

    X = df.drop(columns=['churn'])
    y = df['churn']

    print(f"Features shape (X): {X.shape}")
    print(f"Target distribution:\n{y.value_counts(normalize=True) * 100}\n")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training samples: {X_train.shape[0]} | Testing samples: {X_test.shape[0]}")

    print("Training Tuned Random Forest (this will take a few seconds longer)...")
    rf_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=10,
        min_samples_leaf=4,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    print("Model training completed successfully!")

    y_pred = rf_model.predict(X_test)
    print("\n--- Model Performance Evaluation ---")
    print(f"Accuracy Score: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    importances = rf_model.feature_importances_
    feature_names = X.columns

    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False).reset_index(drop=True)

    print("\n--- TOP 15 MOST INFLUENTIAL FEATURES (The New Gold Table) ---")
    print(importance_df.head(15))

    return rf_model, importance_df


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from src import data_loader

    raw_df = pd.read_csv(data_loader.data_path)
    cleaned_df = data_loader.Feature_eng_cleaning(raw_df)
    train_and_evaluate_churn(cleaned_df)