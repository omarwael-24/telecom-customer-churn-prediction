import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier
import re
import joblib
from pathlib import Path


def clean_column_names_for_xgboost(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    regex = re.compile(r"[\s\,\[\]\{\}\:\<\>\=\+\-\*\/\(\)\!\@\#\$\%\^\&\|\'\`\~\"\'\?]")
    df.columns = [regex.sub("_", col) for col in df.columns]
    return df


def train_advanced_models(df: pd.DataFrame):
    print("--- Training Final Tuned XGBoost Model ---")

    df = clean_column_names_for_xgboost(df)

    if 'churn' not in df.columns:
        raise KeyError("Target column 'churn' not found in the dataframe!")

    X = df.drop(columns=['churn'])
    y = df['churn']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)

    # حقن الـ Best Hyperparameters اللي طلعت من الـ Tuning
    final_model = XGBClassifier(
        n_estimators=363,
        max_depth=8,
        learning_rate=0.03581571779352703,
        subsample=0.6047297487698939,
        colsample_bytree=0.7816414687556065,
        min_child_weight=7,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss'
    )

    final_model.fit(X_train, y_train)
    preds = final_model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    print(f"\n🏆 Final Tuned XGBoost Accuracy: {acc:.4f}")

    print("\nDetailed Classification Report:")
    print(classification_report(y_test, preds))

    # حفظ الموديل النهائي في الفولدر
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    joblib.dump(final_model, models_dir / 'churn_model.pkl')
    print(f"\n🎉 Success! Optimized model saved to '{models_dir / 'churn_model.pkl'}'")

    importances = final_model.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False).reset_index(drop=True)

    print(f"\n--- Final Top 15 Feature Importance ---")
    print(importance_df.head(15))

    return final_model


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from src import data_loader

    raw_df = pd.read_csv(data_loader.data_path)
    cleaned_df = data_loader.Feature_eng_cleaning(raw_df)
    train_advanced_models(cleaned_df)