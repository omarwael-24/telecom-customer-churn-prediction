import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from data_loader import Feature_eng_cleaning, data_path

optuna.logging.set_verbosity(optuna.logging.WARNING)


def objective(trial, X, y):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 9),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 5.0),
        'tree_method': 'hist',
        'random_state': 42,
        'n_jobs': -1
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    f1_scores = []

    for train_idx, val_idx in cv.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = xgb.XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        preds = model.predict(X_val)
        f1_scores.append(f1_score(y_val, preds))

    return np.mean(f1_scores)


if __name__ == "__main__":
    df_raw = pd.read_csv(data_path)
    df_cleaned = Feature_eng_cleaning(df_raw)

    if 'churn' in df_cleaned.columns:
        X = df_cleaned.drop(columns=['churn'])
        y = df_cleaned['churn']
    else:
        X = df_cleaned.iloc[:, :-1]
        y = df_cleaned.iloc[:, -1]

    X = X.select_dtypes(include=[np.number])

    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, X, y), n_trials=50)

    print(f"Best Trial F1-Score: {study.best_value:.4f}")
    print("Best Hyperparameters:")
    for key, value in study.best_params.items():
        print(f"    '{key}': {value},")