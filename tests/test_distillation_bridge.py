import torch

from src.distillation.bridge import DistillationBridge


def test_distillation_bridge_mock_outputs():
    bridge = DistillationBridge(
        enabled=True,
        input_key="x",
        student_hidden_key="student_hidden",
        student_pred_key="student_pred",
        student_hidden_dim=16,
        student_pred_dim=5,
        teacher_backend="mock",
        teacher_hidden_dim=32,
    )

    batch = {
        "x": torch.randn(2, 1, 128),
        "student_hidden": torch.randn(2, 16),
        "student_pred": torch.randn(2, 5),
    }
    out = bridge(**batch)
    assert out["teacher_hidden"].shape == (2, 16)
    assert out["teacher_pred"].shape == (2, 5)
