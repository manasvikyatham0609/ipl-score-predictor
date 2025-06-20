# IPL_Score_Predictor

The aim of this project is to predict the final score of an IPL (Indian Premier League) innings using machine learning techniques, given partial match information such as overs completed, current score, wickets fallen, and other contextual features.

---

## Steps involved in this project are as follows:

### 1. Data Preprocessing
- Loaded the IPL ball-by-ball dataset containing information for each delivery.
- Filtered relevant columns such as match ID, batsman, bowler, runs scored, overs, wickets, etc.
- Removed null values and ensured consistency in data types.
- Aggregated ball-level data to over-level and innings-level formats for modeling.

---

### 2. Feature Engineering
- Extracted several important features such as:
  - **Current Score** (till a given over)
  - **Wickets fallen**
  - **Overs completed**
  - **Batting team**
  - **Bowling team**
  - **Venue**
  - **Run rate**
  - **Balls remaining**
- Encoded categorical variables using One-Hot Encoding and Label Encoding for model compatibility.

---

### 3. Model Training
- Split the dataset into training and testing sets.
- Trained multiple regression models to predict the final score of an innings:
  - **Linear Regression** — RMSE: *xx*
  - **Random Forest Regressor** — RMSE: *xx*
  - **XGBoost Regressor** — RMSE: *xx* *(Best Performing Model)*
- Evaluated the models using **Root Mean Squared Error (RMSE)** and expressed it also as a **percentage of the average final score**.

---

### 4. Model Saving
- The best model (XGBoost) was saved using joblib/pickle for reuse in the Flask backend.
- Stored as `best_model.pkl` under the `model/` directory.

---

### 5. Web Frontend (React)
- Built a user-friendly frontend in **React** where users can input:
  - Batting team
  - Bowling team
  - Venue
  - Overs completed
  - Runs scored
  - Wickets fallen
- On form submission, a request is sent to the Flask backend, and the predicted final score is shown.

---

### 6. Flask Backend
- Serves the ML model through a **REST API**.
- Receives inputs from the frontend, processes them using the same pipeline as training, and returns the predicted score.
- Endpoints:
  - `POST /predict` — returns the predicted final score in JSON format.

---

## CONCLUSION
We have implemented multiple ML models to predict the final score of an IPL innings using minimal match information. The **XGBoost Regressor** provided the most accurate results based on our evaluation metrics. Future work can include:
- Player-specific stats (batsman/bowler form)
- Powerplay and death overs effects
- Match pressure situations and run chase modeling

---

## About
Machine Learning + React + Flask project to build a complete web app for real-time IPL score prediction.

---

## Resources
📂 `/data` — Processed CSV files  
📂 `/model` — Trained ML model  
📂 `/frontend` — React App  
📂 `/backend` — Flask App

---

## Languages
- Python (Scikit-learn, XGBoost, Pandas)
- JavaScript (React)
- HTML/CSS
