import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# === Step 1: Load Processed Data ===
df = pd.read_csv("data/ipl_processed_data.csv")
df.dropna(inplace=True)

# === Step 2: Features and Target ===
X = df[['runs', 'wickets', 'overs']]
y = df['final_score']

# === Step 3: Train-Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Step 4: Load Trained Model ===
model_path = os.path.join("backend", "ipl_best_model_xgb.pkl")
model = joblib.load(model_path)

# === Step 5: Predict on Test Data ===
y_pred = model.predict(X_test)

# === Step 6: Evaluate ===
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n✅ Model Evaluation:")
print(f"📉 Mean Absolute Error (MAE): {mae:.2f} runs")
print(f"📈 R² Score: {r2:.2f}")

# === Step 7: Plot Actual vs Predicted ===
sns.set(style="whitegrid")
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred, color='blue', alpha=0.6)
plt.xlabel("Actual Final Score")
plt.ylabel("Predicted Final Score")
plt.title("Actual vs Predicted Final Scores")
plt.grid(True)
plt.tight_layout()
plt.show()

# === Step 6.1: Custom Accuracy Calculation ===
tolerance = 10
accurate_predictions = np.abs(y_test - y_pred) <= tolerance
accuracy = np.mean(accurate_predictions) * 100
print(f"\U0001F389 Custom Accuracy (±{tolerance} runs): {accuracy:.2f}%")

