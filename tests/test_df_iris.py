from pathlib import Path
import sys

import numpy as np
import pytest
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from df_iris.model import MultiBranchResNet
from df_iris.dataset import AugmentConfig, DFIrisDataset, IrisSample
from df_iris.metrics import compute_metrics


def test_preprocess_outputs_single_channel_image(tmp_path: Path):
    image_path = tmp_path / "sample.png"
    Image.new("L", (40, 30), color=128).save(image_path)
    dataset = DFIrisDataset(
        [IrisSample(str(image_path), "001", "L", "001_L", "train")],
        {"001_L": 0},
        train=True,
        augment=AugmentConfig("none"),
        image_size=224,
    )

    x, y, identity, path = dataset[0]

    assert x.shape == (1, 224, 224)
    assert y.item() == 0
    assert identity == "001_L"
    assert path == str(image_path)


def test_model_forward_and_embedding_shape():
    model = MultiBranchResNet(num_classes=3, k_local=2, num_tokens=2, proj_dim=16, train_stage="full")
    x = torch.randn(2, 1, 224, 224)
    y = torch.tensor([0, 1])

    loss, logits, parts = model(x, y, current_epoch=1, total_epochs=2, classifier="arcface")
    out = model(x, labels=None)

    assert loss.ndim == 0
    assert logits.shape == (2, 3)
    assert {"global", "local", "integrated"} <= set(parts)
    assert out["integrated_feat"].shape == (2, 16)
    assert out["cam"].ndim == 2
    assert out["cdm"].ndim == 2


def test_metrics_only_report_eer_and_accuracy():
    features = np.asarray(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [0.1, 0.9],
        ],
        dtype=np.float32,
    )
    labels = np.asarray(["a", "a", "b", "b"])

    metrics = compute_metrics(features, labels)

    assert set(metrics) == {
        "eer",
        "eer_percent",
        "eer_threshold",
        "verification_accuracy",
        "verification_accuracy_percent",
        "genuine_pairs",
        "impostor_pairs",
        "num_samples",
        "num_identities",
    }
    assert metrics["eer"] == pytest.approx(0.0)
    assert metrics["verification_accuracy"] == pytest.approx(1.0)
