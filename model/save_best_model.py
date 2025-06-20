import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import joblib
import os
import shutil

# ✅ This dataset MUST be the updated one with full innings
df = pd.read_csv("ipl_processed_data.csv")
df.dropna(inplace=True)

X = df[['runs', 'wickets', 'overs']]
y = df['final_score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ XGBoost model
model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

# Save in current directory first
model_filename = "ipl_best_model_xgb.pkl"
joblib.dump(model, model_filename)
print(f"✅ Trained and saved model to '{model_filename}'")

# Define destination
destination = os.path.join("..", "backend", model_filename)

# Create backend folder if it doesn't exist
os.makedirs(os.path.dirname(destination), exist_ok=True)

# Move or overwrite model into backend folder
shutil.move(model_filename, destination)
print(f"✅ Model moved to backend: {destination}")
