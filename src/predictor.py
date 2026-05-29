import joblib
import pandas as pd
import re
from src import data_loader

# Load model directly
model = joblib.load('models/churn_model.pkl')

# Read raw data
df = pd.read_csv(data_loader.data_path)
test_data = df.head(5)

# Clean using pipeline
cleaned_df = data_loader.Feature_eng_cleaning(test_data)

# Drop target column
if 'churn' in cleaned_df.columns:
    cleaned_df = cleaned_df.drop(columns=['churn'])

# Clean column names
regex = re.compile(r"[\s\,\[\]\{\}\:\<\>\=\+\-\*\/\(\)\!\@\#\$\%\^\&\|\'\`\~\"\'\?]")
cleaned_df.columns = [regex.sub("_", c) for c in cleaned_df.columns]

# FIX: Add missing dummy columns with 0
model_features = model.get_booster().feature_names
for col in model_features:
    if col not in cleaned_df.columns:
        cleaned_df[col] = 0

# FIX: Align column order exactly like the model
cleaned_df = cleaned_df[model_features]

# Predict classes and probabilities
preds = model.predict(cleaned_df)
probs = model.predict_proba(cleaned_df)

print("\n=================== LIVE INFERENCE TEST ===================")
# Loop over actual predictions length
for i in range(len(preds)):
    p = preds[i]
    status = "🔴 Churn" if p == 1 else "🟢 Loyal"
    conf = probs[i][p] * 100
    print(f"Customer {i+1}: Prediction -> {status} | Confidence -> {conf:.2f}%")
print("===========================================================")