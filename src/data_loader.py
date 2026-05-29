# import libraries
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder

# --------

BASE_DIR = Path(__file__).resolve().parent.parent
data_path = BASE_DIR / "data" / "cleaningv1.csv"

def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    initial_rows = df.shape[0]
    core_cols = ['rev_Mean', 'mou_Mean', 'eqpdays', 'months', 'change_mou', 'totmrc_Mean']
    existing_core = [col for col in core_cols if col in df.columns]
    for col in existing_core:
        # تحويل الأرقام لضمان عدم وجود مشاكل في الحسابات
        df[col] = pd.to_numeric(df[col], errors='coerce')
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1

        lower_bound = q1 - 3.0 * iqr
        upper_bound = q3 + 3.0 * iqr

        # الفلترة
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound) | df[col].isna()]

    df = df.reset_index(drop=True)
    return df

# this is a cleaning funciton of the data . (EDA)
def Feature_eng_cleaning(dataframe: pd.DataFrame) -> pd.DataFrame:
    """this function is to do a feature engineering stage and cleaning the datasets"""
    df = dataframe.copy()

    #deleting space in columns
    df.columns = df.columns.str.strip()

    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])

    if 'uniqsubs' in df.columns:
        df = df[df['uniqsubs'] <= 15]
        df = df.reset_index(drop=True)
    df = remove_outliers(df)
    #==== Adding some  Feature Engineering =====
    if 'eqpdays' in df.columns and 'months' in df.columns:
        df['eqp_age_ratio'] = df['eqpdays'] / ((df['months'] * 30) + 1)

    if 'rev_Mean' in df.columns and 'mou_Mean' in df.columns:
        df['cost_per_minute'] = df['rev_Mean'] / (df['mou_Mean'] + 1)

    if 'vceovr_Mean' in df.columns and 'drop_vce_Mean' in df.columns:
        df['total_wasted_minutes'] = df['vceovr_Mean'] + df['drop_vce_Mean']

    if 'custcare_Mean' in df.columns and 'change_mou' in df.columns:
        df['negotiation_ready_index'] = df['custcare_Mean'] * (df['change_mou'].clip(upper=0).abs() + 1)

    if 'adults' in df.columns and 'phones' in df.columns and 'rev_Mean' in df.columns:
        df['family_retention_potential'] = (df['adults'] * df['phones']) / (df['rev_Mean'] + 1)

    if 'change_mou' in df.columns and 'mou_Mean' in df.columns:
        df['mou_drop_velocity'] = df['change_mou'] / (df['mou_Mean'] + 1)

    if 'change_rev' in df.columns and 'rev_Mean' in df.columns:
        df['rev_drop_velocity'] = df['change_rev'] / (df['rev_Mean'] + 1)

    if 'totmrc_Mean' in df.columns and 'hnd_price' in df.columns:
        df['financial_leverage_ratio'] = df['totmrc_Mean'] / (df['hnd_price'] + 1)

    if 'peak_vce_Mean' in df.columns and 'opk_vce_Mean' in df.columns:
        df['peak_to_offpeak_ratio'] = df['peak_vce_Mean'] / (df['opk_vce_Mean'] + 1)

    if 'vceovr_Mean' in df.columns and 'drop_vce_Mean' in df.columns:
        df['wasted_cost_index'] = df['vceovr_Mean'] * df['drop_vce_Mean']

    if 'ovrmou_Mean' in df.columns and 'mou_Mean' in df.columns:
        df['overage_minutes_ratio'] = df['ovrmou_Mean'] / (df['mou_Mean'] + 1)

    if 'months' in df.columns and 'eqpdays' in df.columns:
        df['loyalty_device_index'] = df['months'] / (df['eqpdays'] + 1)

    if 'ovrrev_Mean' in df.columns and 'totmrc_Mean' in df.columns:
        df['bill_shock_index'] = df['ovrrev_Mean'] / (df['totmrc_Mean'] + 1)

    if 'drop_vce_Mean' in df.columns and 'blck_dat_Mean' in df.columns and 'mou_Mean' in df.columns:
        df['network_failure_index'] = (df['drop_vce_Mean'] + df['blck_dat_Mean']) / (df['mou_Mean'] + 1)

    if 'drop_vce_Mean' in df.columns and 'blck_dat_Mean' in df.columns:
        df['voice_to_data_friction'] = df['drop_vce_Mean'] / (df['blck_dat_Mean'] + 1)

    if 'drop_dat_Mean' in df.columns and 'blck_dat_Mean' in df.columns:
        df['total_data_frustration'] = df['drop_dat_Mean'] + df['blck_dat_Mean']

    if 'drop_vce_Mean' in df.columns and 'drop_dat_Mean' in df.columns:
        df['voice_vs_data_drop_ratio'] = df['drop_vce_Mean'] / (df['drop_dat_Mean'] + 1)

    if 'comp_vce_Mean' in df.columns and 'plcd_vce_Mean' in df.columns:
        df['voice_completion_rate'] = df['comp_vce_Mean'] / (df['plcd_vce_Mean'] + 1)

    if 'comp_dat_Mean' in df.columns and 'plcd_dat_Mean' in df.columns:
        df['data_completion_rate'] = df['comp_dat_Mean'] / (df['plcd_dat_Mean'] + 1)

    if 'avg3mou' in df.columns and 'avg6mou' in df.columns:
        df['mou_velocity_3m_6m'] = df['avg3mou'] / (df['avg6mou'] + 1)

    if 'avg3rev' in df.columns and 'avg6rev' in df.columns:
        df['revenue_velocity_3m_6m'] = df['avg3rev'] / (df['avg6rev'] + 1)

    if 'avg3qty' in df.columns and 'avg6qty' in df.columns:
        df['call_quantity_velocity_3m_6m'] = df['avg3qty'] / (df['avg6qty'] + 1)

    if 'roam_Mean' in df.columns and 'rev_Mean' in df.columns:
        df['roaming_intensity'] = df['roam_Mean'] / (df['rev_Mean'] + 1)

    if 'da_Mean' in df.columns and 'mou_Mean' in df.columns:
        df['directory_assist_intensity'] = df['da_Mean'] / (df['mou_Mean'] + 1)

    if 'eqpdays' in df.columns and 'months' in df.columns:
        df['hardware_vulnerability_score'] = df['eqpdays'] / (df['months'] + 1)


    #Deleting Unwanted string data
    garbage_cols = [
        'ethnic', 'infobase', 'HHstatin', 'dwllsize', 'dwlltype',
        'kid0_2', 'kid3_5', 'kid6_10', 'kid11_15', 'kid16_17'
    ]
    df = df.drop(columns=[col for col in garbage_cols if col in df.columns])

    # one-hot-ecnoding  for new_cell column Y , N , U
    if 'new_cell' in df.columns:
        df = pd.get_dummies(df, columns=['new_cell'], drop_first=True, dtype=int)

    #one-hot-ecnoding for 'asl_flag', 'dualband', 'refurb_new'
    #
    target_columns = [
        'asl_flag', 'dualband', 'refurb_new',
        'area', 'prizm_social_one', 'marital',
        'ownrent'
            ]
    available_cols = [col for col in target_columns if col in df.columns]
    if available_cols:
        #Handling missing values in Categorial columns
        df[available_cols] = df[available_cols].fillna('Unknown')
        df = pd.get_dummies(df, columns=available_cols, drop_first=True, dtype=int)

    # i have found that there is a relationship between uniquesup and churn columns
    # so i categorize the uniquesup column to 3 major categories
    # 0-1 --> single , 1-4 --> family , 4-15 --> large family all that in column "subscriber_type"
    def encode_subscriber_types(df: pd.DataFrame, column_name: str = 'subscriber_type') -> pd.DataFrame:
        df = df.copy()
        bins = [0, 1, 4, 15]
        labels = ['Single', 'Family', 'Large Family']

        df[column_name] = pd.cut(df['uniqsubs'], bins=bins, labels=labels)

        family = [labels]
        encoder = OrdinalEncoder(categories=family)

        df[column_name] = encoder.fit_transform(df[[column_name]]).astype(int)

        df = df.drop(columns='uniqsubs')

        return df

    df = encode_subscriber_types(df, column_name='subscriber_type')

    # ----------Transofrming 'crclscod' column (Credit class code) to percent value of churn.
    def Encoding_Credit_Class_Code(df: pd.DataFrame, column_name: str = "crclscod") -> pd.DataFrame:
        df = df.copy()

        df['F_alphabet_code_class'] = df[column_name].str[0]

        grouping_bymean = df.groupby('F_alphabet_code_class')['churn'].mean()

        df['classCode_ecoding_score'] = df['F_alphabet_code_class'].map(grouping_bymean)

        df = df.drop(columns=['crclscod', 'F_alphabet_code_class'])

        return df

        # get_dumies for new_cell column Y , N , U
        if 'new_cell' in df.columns:
            df = pd.get_dummies(df, columns=['new_cell'], drop_first=True, dtype=int)


    df = Encoding_Credit_Class_Code(df, column_name='crclscod')

    #cleaning numerical data
    def cleaning_numerical_data(df:pd.DataFrame) -> pd.DataFrame:
        df=df.copy()

        usage_cols = [
            'change_mou', 'change_rev', 'drop_vce_Mean', 'blck_vce_Mean', 'custcare_Mean',
            'rev_Mean', 'mou_Mean', 'totmrc_Mean', 'months',
            'attempt_Mean', 'complete_Mean', 'plcd_vce_Mean',
            'peak_vce_Mean', 'opk_vce_Mean', 'mou_peav_Mean', 'mou_opkv_Mean',
            'callwait_Mean', 'threeway_Mean', 'roam_Mean',
            'numbcars', 'lor', 'income', 'adults', 'hnd_webcap',
            'truck', 'creditcd', 'forgntvl', 'rv', 'hnd_price',
            'datovr_Mean', 'vceovr_Mean', 'da_Mean', 'ovrmou_Mean',
            'ovrrev_Mean', 'phones', 'models', 'eqpdays',
            
            #the New feature engineering columns to clean the inf or NAN values 
            'eqp_age_ratio', 'cost_per_minute', 'total_wasted_minutes',
            'negotiation_ready_index', 'family_retention_potential',
            'mou_drop_velocity', 'rev_drop_velocity', 'financial_leverage_ratio',
            'peak_to_offpeak_ratio', 'wasted_cost_index', 'overage_minutes_ratio', 'loyalty_device_index',
            'total_data_frustration','voice_vs_data_drop_ratio',
            'voice_completion_rate','data_completion_rate','mou_velocity_3m_6m','revenue_velocity_3m_6m','call_quantity_velocity_3m_6m',
            'roaming_intensity','directory_assist_intensity',
            'hardware_vulnerability_score'
        ]
        existing_num_cols=[col for col in usage_cols if col in df.columns]
        if existing_num_cols:

            df[existing_num_cols] = df[existing_num_cols].apply(pd.to_numeric, errors='coerce')

            #replacing the inf and -inf values with nan
            df[existing_num_cols] = df[existing_num_cols].replace([np.inf, -np.inf], np.nan)
            median_value = df[existing_num_cols].median()

            df[existing_num_cols]=df[existing_num_cols].fillna(median_value)

            remaining_nans = df[existing_num_cols].isna().sum()
            failed_cols = remaining_nans[remaining_nans > 0].index.tolist()
            if failed_cols:
                df[failed_cols] = df[failed_cols].fillna(0)

            all_remaining_nans = df.isna().sum()
            final_failed_cols = all_remaining_nans[all_remaining_nans > 0].index.tolist()
            if final_failed_cols:
                for col in final_failed_cols:
                    if df[col].dtype in [np.float64, np.int64]:
                        df[col] = df[col].fillna(df[col].median())
                        df[col] = df[col].fillna(0)  # لو العمود كله كان فاضي
                    else:
                        df[col] = df[col].fillna('Unknown')

        return df
    df=cleaning_numerical_data(df)

    return df


# --- Pipeline Execution and Testing Block ---
if __name__ == "__main__":
    df_raw = pd.read_csv(data_path)
    print(f"Original dataset shape: {df_raw.shape}")
    df_cleaned = Feature_eng_cleaning(df_raw)
    print(f"Processed dataset shape: {df_cleaned.shape}")
    print(f"Total NaNs left: {df_cleaned.isna().sum().sum()}")