import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from preprocess import preprocess  # noqa: E402
from train import train  # noqa: E402
from predict import predict  # noqa: E402

RAW = os.path.join(os.path.dirname(__file__), "..", "data", "data.csv")


def test_preprocess_creates_split(tmp_path):
    train_df, test_df = preprocess(RAW, str(tmp_path))
    assert len(train_df) > len(test_df) > 0
    assert (tmp_path / "train.csv").exists()
    assert (tmp_path / "test.csv").exists()


def test_train_accuracy_is_reasonable(tmp_path):
    proc, models = tmp_path / "proc", tmp_path / "models"
    preprocess(RAW, str(proc))
    accuracy = train(str(proc), str(models))
    assert accuracy >= 0.8
    assert (models / "model.joblib").exists()
    assert (models / "metrics.json").exists()


def test_predict_returns_valid_species(tmp_path):
    proc, models = tmp_path / "proc", tmp_path / "models"
    preprocess(RAW, str(proc))
    train(str(proc), str(models))
    result = predict([5.1, 3.5, 1.4, 0.2], str(models / "model.joblib"))
    assert result in {"setosa", "versicolor", "virginica"}
