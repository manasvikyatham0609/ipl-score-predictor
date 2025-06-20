# 🏏 IPL Score Predictor

A machine learning-powered web app that predicts the final score of an IPL innings based on live match inputs: runs, wickets, and overs.

---

## 🚀 Tech Stack

| Layer     | Tech                      |
|-----------|---------------------------|
| Frontend  | React + Tailwind CSS      |
| Backend   | Flask           |
| ML        | Python, pandas, scikit-learn, xgboost |
| Data      | IPL (2008–2020) ball-by-ball Kaggle dataset |

---

## 📂 Folder Structure

ipl-score-predictor/
├── backend/ ← Flask API + model
├── frontend/ ← React UI
├── model/ ← Data prep + training scripts
├── data/ ← Raw and processed CSVs
├── README.md
├── .gitignore

---

## 💡 Features

- Predicts final score based on:
  - 🏃 Runs Scored
  - 💥 Wickets Fallen
  - ⏱️ Overs Completed
- React UI with live input
- Trained on real IPL data
- Model accuracy evaluated

---

## 📈 Model Evaluation

| Metric             | Value         |
|--------------------|---------------|
| MAE (Mean Error)   | ~14.19 runs     |
| R² Score           | ~0.53         |

---

## 🛠 How to Run Locally

### 🧠 Backend (Flask)
```bash
cd backend
pip install -r requirements.txt
python app.py

🌐 Frontend (React)

cd frontend
npm install
npm start
