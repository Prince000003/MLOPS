# test change for pull request
"""Train a RandomForest classifier and save the model and metrics."""
import json
import os

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED = os.path.join(BASE, "data", "processed")
MODELS = os.path.join(BASE, "models")


def train(processed_dir=PROCESSED, models_dir=MODELS, seed=42):
    train_df = pd.read_csv(os.path.join(processed_dir, "train.csv"))
    test_df = pd.read_csv(os.path.join(processed_dir, "test.csv"))

    X_train, y_train = train_df.drop(columns="target"), train_df["target"]
    X_test, y_test = test_df.drop(columns="target"), test_df["target"]

    model = RandomForestClassifier(n_estimators=100, random_state=seed)
    model.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test))

    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(model, os.path.join(models_dir, "model.joblib"))
    with open(os.path.join(models_dir, "metrics.json"), "w") as f:
        json.dump({"accuracy": accuracy}, f)

    print(f"Model trained. Accuracy: {accuracy:.4f}")
    return accuracy


if __name__ == "__main__":
    train()
