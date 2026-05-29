import pandas as pd
from src import data_loader
from src import advanced_model  # استدعاء الملف الجديد


def main():
    print("============ STARTING TELECOM CHURN PROJECT ============")

    raw_df = pd.read_csv(data_loader.data_path)

    df_cleaned = data_loader.Feature_eng_cleaning(raw_df)
    print(f"Cleaned Data Shape: {df_cleaned.shape}\n")

    # تشغيل التدريب المتقدم بالـ Boosting
    best_model = advanced_model.train_advanced_models(df_cleaned)

    print("\n============ PIPELINE EXECUTED SUCCESSFULLY ============")


if __name__ == "__main__":
    main()