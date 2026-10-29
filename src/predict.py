"""Load the trained model and predict the iris species."""
import os
import sys

import joblib
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE, "models", "model.joblib")
FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]


def predict(values, model_path=MODEL_PATH):
    """values: [sepal_length, sepal_width, petal_length, petal_width]"""
    model = joblib.load(model_path)
    row = pd.DataFrame([values], columns=FEATURES)
    return model.predict(row)[0]


if __name__ == "__main__":
    # usage: python src/predict.py 5.1 3.5 1.4 0.2
    args = [float(x) for x in sys.argv[1:5]] or [5.1, 3.5, 1.4, 0.2]
    print("Input:", args)
    print("Predicted species:", predict(args))
