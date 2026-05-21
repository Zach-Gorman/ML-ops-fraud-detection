import os
import joblib
import numpy as np


def test_model_exists():
    assert os.path.exists("models/model.pkl")


def test_prediction():
    model = joblib.load("models/model.pkl")
    sample = np.zeros((1, 30))
    prediction = model.predict(sample)
    assert prediction[0] in [0, 1]