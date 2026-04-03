import torch

from src.model.student_classification import StudentClassifier
from src.model.student_forecasting import StudentForecaster


def test_student_classifier_shapes():
    model = StudentClassifier(in_channels=2, n_classes=5, hidden_dim=32, n_heads=4, n_layers=1)
    inputs = torch.randn(4, 2, 64)
    out = model(inputs=inputs)
    assert out["logits"].shape == (4, 5)
    assert out["student_hidden"].shape == (4, 32)


def test_student_forecaster_shapes():
    model = StudentForecaster(
        in_channels=3,
        horizon=12,
        hidden_dim=32,
        n_heads=4,
        n_layers=1,
    )
    inputs = torch.randn(4, 3, 48)
    out = model(inputs=inputs)
    assert out["forecast"].shape == (4, 3, 12)
    assert out["student_pred"].shape == (4, 36)
