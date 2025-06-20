import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# === Load Preprocessed Data ===
df = pd.read_csv("data/ipl_processed_data.csv")
df.dropna(inplace=True)

X = df[['runs', 'wickets', 'overs']]
y = df['final_score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Dictionary to store results ===
results = {}

# === Helper to calculate metrics ===
def evaluate_model(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    rmse_percent = (rmse / y_true.mean()) * 100
    results[name] = {
        'MAE': mae,
        'R2': r2,
        'RMSE': rmse,
        'RMSE%': rmse_percent
    }

# === 1. Linear Regression ===
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
evaluate_model("Linear Regression", y_test, y_pred_lr)

# === 2. Random Forest ===
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
evaluate_model("Random Forest", y_test, y_pred_rf)

# === 3. XGBoost ===
xgb = XGBRegressor(n_estimators=100, random_state=42)
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)
evaluate_model("XGBoost", y_test, y_pred_xgb)

# === Print Results ===
print("\n📊 Model Comparison:\n")
for model, scores in results.items():
    print(f"{model}:")
    print(f"  📉 MAE     = {scores['MAE']:.2f}")
    print(f"  📐 RMSE    = {scores['RMSE']:.2f} ({scores['RMSE%']:.2f}%)")
    print(f"  📈 R²      = {scores['R2']:.2f}\n")

# === Optional: Plot Comparison ===
model_names = list(results.keys())
maes = [results[m]['MAE'] for m in model_names]
r2s = [results[m]['R2'] for m in model_names]
rmses = [results[m]['RMSE'] for m in model_names]

# --- Plot MAE
plt.figure(figsize=(6, 4))
sns.barplot(x=model_names, y=maes, palette="Reds_r")
plt.title("Mean Absolute Error (MAE) Comparison")
plt.ylabel("MAE (Runs)")
plt.tight_layout()
plt.show()

# --- Plot RMSE
plt.figure(figsize=(6, 4))
sns.barplot(x=model_names, y=rmses, palette="Purples")
plt.title("Root Mean Squared Error (RMSE) Comparison")
plt.ylabel("RMSE (Runs)")
plt.tight_layout()
plt.show()

# --- Plot R²
plt.figure(figsize=(6, 4))
sns.barplot(x=model_names, y=r2s, palette="Blues")
plt.title("R² Score Comparison")
plt.ylabel("R² Score")
plt.tight_layout()
plt.show()
