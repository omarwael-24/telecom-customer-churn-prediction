import pandas as pd
from src import data_loader


def main():

    raw_df = pd.read_csv(data_loader.data_path)

    df_cleaned = data_loader.Feature_eng_cleaning(raw_df)
    print(f"Cleaned Data Shape: {df_cleaned.shape}\n")


if __name__ == "__main__":
    main()
