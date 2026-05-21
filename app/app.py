import joblib
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

model = joblib.load("models/model.pkl")


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    features = np.array(data["features"]).reshape(1, -1)

    prediction = int(model.predict(features)[0])

    return jsonify({
        "prediction": prediction,
        "risk": "fraud" if prediction == 1 else "legitimate"
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)