# backend/app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)

# Load trained model
model = joblib.load("ipl_best_model_xgb.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    runs = data['runs']
    wickets = data['wickets']
    overs = data['overs']

    input_array = np.array([[runs, wickets, overs]])
    print("INPUT:", input_array)

    prediction = model.predict(input_array)[0]
    print("PREDICTION:", prediction)

    return jsonify({'predicted_score': int(prediction)})


if __name__ == "__main__":
    app.run(debug=True)
