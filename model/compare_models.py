import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import seaborn as sns

# === Load Preprocessed Data ===
df = pd.read_csv('ipl_processed_data.csv')
df.dropna(inplace=True)

X = df[['runs', 'wickets', 'overs']]
y = df['final_score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Dictionary to store results ===
results = {}

# === 1. Linear Regression ===
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
results['Linear Regression'] = {
    'MAE': mean_absolute_error(y_test, y_pred_lr),
    'R2': r2_score(y_test, y_pred_lr)
}

# === 2. Random Forest ===
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
results['Random Forest'] = {
    'MAE': mean_absolute_error(y_test, y_pred_rf),
    'R2': r2_score(y_test, y_pred_rf)
}

# === 3. XGBoost Regressor ===
xgb = XGBRegressor(n_estimators=100, random_state=42)
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)
results['XGBoost'] = {
    'MAE': mean_absolute_error(y_test, y_pred_xgb),
    'R2': r2_score(y_test, y_pred_xgb)
}

# === Print Results ===
print("📊 Model Comparison:\n")
for model, scores in results.items():
    print(f"{model}: MAE = {scores['MAE']:.2f}, R² = {scores['R2']:.2f}")

# === Optional: Plot Comparison ===
model_names = list(results.keys())
maes = [results[m]['MAE'] for m in model_names]
r2s = [results[m]['R2'] for m in model_names]

fig, ax1 = plt.subplots()
sns.barplot(x=model_names, y=maes, palette="Reds_r", ax=ax1)
ax1.set_title("MAE Comparison")
ax1.set_ylabel("Mean Absolute Error")
plt.show()

fig, ax2 = plt.subplots()
sns.barplot(x=model_names, y=r2s, palette="Blues", ax=ax2)
ax2.set_title("R² Score Comparison")
ax2.set_ylabel("R² Score")
plt.show()
