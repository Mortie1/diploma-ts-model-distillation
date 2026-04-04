import torch

from src.model.student_classification import StudentClassifier


def test_student_classifier_shapes():
    model = StudentClassifier(in_channels=2, n_classes=5, hidden_dim=32, n_heads=4, n_layers=1)
    inputs = torch.randn(4, 2, 64)
    out = model(inputs=inputs)
    assert out["logits"].shape == (4, 5)
    assert out["student_hidden"].shape == (4, 32)
