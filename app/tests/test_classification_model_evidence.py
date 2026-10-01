"""ML-03: evidencia con pesos ImageNet reales y crops reales (ver tests/evidence/ml-03-model.md).

Descarga `ResNet18_Weights.IMAGENET1K_V1` (~45 MB, caché de torch hub) y usa el
DataLoader de ML-02 sobre el manifiesto real. Se activa con RUN_MODEL_EVIDENCE=1.
"""

import os
from pathlib import Path

import pytest
import torch
from torch import nn

from classification.dataset import build_dataloader, load_split
from classification.model import (
    ModelConfig,
    build_model,
    load_checkpoint,
    predict_proba,
    save_checkpoint,
)
from presentation.ml_contracts import TrainingParams

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "reports" / "releases" / "v0.1.1" / "manifest.json"
CROPS_REPORT = ROOT / "reports" / "crops.json"
CROPS_DIR = ROOT / "data" / "crops"
STEPS = 5

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_MODEL_EVIDENCE") != "1",
    reason="Set RUN_MODEL_EVIDENCE=1 with data/crops materialized (descarga pesos ImageNet)",
)


def test_short_run_on_real_crops_updates_the_pretrained_model(tmp_path):
    torch.manual_seed(42)
    params = TrainingParams(
        optimizer="adam",
        batch_size=16,
        max_epochs=1,
        learning_rate=0.001,
        image_size=128,
        hidden_layers=[128],
        dropout=0.2,
        seed=42,
        patience=1,
        min_delta=0.0,
    )

    def split(name):
        return load_split(MANIFEST, CROPS_REPORT, crops_dir=CROPS_DIR, split=name, params=params)

    train = build_dataloader(split("train"), batch_size=params.batch_size, seed=params.seed)
    probe = next(iter(build_dataloader(split("validation"), batch_size=8, seed=0)))

    model = build_model(ModelConfig.from_params(params))
    print(f"\n{model.trainable_summary()}")
    before_weights = {k: v.detach().clone() for k, v in model.state_dict().items()}
    before_proba = predict_proba(model, probe["image"])

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=params.learning_rate
    )
    batches = iter(train)
    model.train()
    for step in range(STEPS):
        batch = next(batches)
        optimizer.zero_grad()
        loss = nn.functional.cross_entropy(model(batch["image"]), batch["label"])
        loss.backward()
        optimizer.step()
        print(f"step {step + 1}: loss={loss.item():.4f} crops={batch['crop_id'][:2]}...")

    after = model.state_dict()
    changed = sorted(
        {
            k.split(".")[0] + "." + k.split(".")[1]
            for k in before_weights
            if not torch.equal(before_weights[k], after[k])
        }
    )
    frozen_changed = [
        k
        for k in before_weights
        if k.startswith(
            (
                "backbone.conv1",
                "backbone.bn1",
                "backbone.layer1",
                "backbone.layer2",
                "backbone.layer3",
            )
        )
        and not torch.equal(before_weights[k], after[k])
    ]
    after_proba = predict_proba(model, probe["image"])
    delta = (after_proba - before_proba).abs().max().item()
    print(f"bloques con pesos cambiados: {changed}")
    print(f"tensores congelados que cambiaron: {len(frozen_changed)}")
    print(
        f"validation crop {probe['crop_id'][0]} ({probe['class_name'][0]}): "
        f"P(dog,cat) antes={before_proba[0].tolist()} después={after_proba[0].tolist()}"
    )
    print(f"máximo cambio de probabilidad en 8 crops de validation: {delta:.4f}")

    assert changed == ["backbone.layer4", "head.0", "head.3"]
    assert frozen_changed == []
    assert delta > 0

    restored = load_checkpoint(save_checkpoint(model, tmp_path / "model.pt"))
    assert torch.equal(predict_proba(restored, probe["image"]), after_proba)
    print("checkpoint recargado: predicciones idénticas")
