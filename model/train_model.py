import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# === Step 1: Load Data ===
df = pd.read_csv("data/ipl_processed_data.csv")

# Remove any missing data
df.dropna(inplace=True)

# === Step 2: Define Features and Target ===
X = df[['runs', 'wickets', 'overs']]  # input features
y = df['final_score']                 # what we want to predict

# === Step 3: Split Data into Train & Test Sets ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# === Step 4: Train the Model ===
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# === Step 5: Predict on Test Set ===
y_pred = model.predict(X_test)

# === Step 6: Evaluate the Model ===
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"✅ Model Evaluation:")
print(f"MAE (Mean Absolute Error): {mae:.2f}")
print(f"R² Score: {r2:.2f}")

# === Step 7: Plot Actual vs Predicted ===
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual Final Score")
plt.ylabel("Predicted Final Score")
plt.title("Actual vs Predicted Scores")
plt.grid(True)
plt.show()

# === Step 8: Save the Trained Model ===
model = joblib.load('/backend/ipl_best_model_xgb.pkl')
print("✅ Model saved as 'ipl_score_predictor_model.pkl'")
