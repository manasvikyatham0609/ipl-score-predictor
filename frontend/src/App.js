import React, { useState } from "react";
import axios from "axios";

function App() {
  const [runs, setRuns] = useState(50);
  const [wickets, setWickets] = useState(2);
  const [overs, setOvers] = useState(10.0);
  const [prediction, setPrediction] = useState(null);

  const handlePredict = async () => {
    const response = await axios.post("http://localhost:5000/predict", {
      runs,
      wickets,
      overs,
    });
    setPrediction(response.data.predicted_score);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-100 to-purple-200 flex flex-col items-center justify-center p-4">
      <div className="bg-white shadow-xl rounded-2xl p-8 w-full max-w-md text-center">
        <h1 className="text-3xl font-bold mb-2">🏏 IPL Score Predictor</h1>
        <p className="text-gray-600 mb-6">
          Predict the <strong>final score</strong> of an IPL innings based on current match situation.
        </p>

        <div className="space-y-4 text-left">
          <div>
            <label className="font-medium">🏃 Runs Scored So Far</label>
            <input
              type="number"
              className="w-full mt-1 p-2 border rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-400"
              value={runs}
              onChange={(e) => setRuns(Number(e.target.value))}
            />
          </div>

          <div>
            <label className="font-medium">💥 Wickets Fallen</label>
            <input
              type="number"
              min="0"
              max="10"
              className="w-full mt-1 p-2 border rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-400"
              value={wickets}
              onChange={(e) => setWickets(Number(e.target.value))}
            />
          </div>

          <div>
            <label className="font-medium">⏱️ Overs Completed</label>
            <input
              type="number"
              step="0.1"
              min="0"
              max="20"
              className="w-full mt-1 p-2 border rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-400"
              value={overs}
              onChange={(e) => setOvers(Number(e.target.value))}
            />
          </div>
        </div>

        <button
          onClick={handlePredict}
          className="mt-6 w-full bg-blue-600 text-white py-2 rounded-xl hover:bg-blue-700 transition"
        >
          📊 Predict Final Score
        </button>

        {prediction !== null && (
          <div className="mt-6 text-green-600 text-lg font-semibold">
            🏁 Predicted Final Score: {prediction} runs
          </div>
        )}

        <hr className="my-6" />
        <p className="text-sm text-gray-500">
          Model trained on IPL ball-by-ball data (2008–2020), powered by XGBoost.
        </p>
      </div>
    </div>
  );
}

export default App;
