from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from df_iris.model import DF_IRIS_MODEL_DEFAULTS, MultiBranchResNet
from df_iris.dataset import (
    AugmentConfig,
    DFIrisDataset,
    assert_subject_disjoint,
    build_protocol,
    build_protocol_from_manifest,
    samples_for,
    train_label_map,
)
from df_iris.metrics import compute_metrics
from df_iris.utils import (
    dataloader_kwargs,
    load_config,
    load_optional_checkpoint,
    optional_path,
    required_path,
    section,
    set_seed,
)


def model_config(cfg: dict[str, Any]) -> dict[str, Any]:
    values = section(cfg, "model")
    return {
        "proj_dim": int(values.get("proj_dim", DF_IRIS_MODEL_DEFAULTS["proj_dim"])),
        "k_local": int(values.get("k_local", DF_IRIS_MODEL_DEFAULTS["k_local"])),
        "num_tokens": int(values.get("num_tokens", DF_IRIS_MODEL_DEFAULTS["num_tokens"])),
        "dropout_p": float(values.get("dropout", DF_IRIS_MODEL_DEFAULTS["dropout_p"])),
        "fusion_branch_dropout": float(values.get("fusion_branch_dropout", DF_IRIS_MODEL_DEFAULTS["fusion_branch_dropout"])),
        "diversity_weight": float(values.get("diversity_weight", DF_IRIS_MODEL_DEFAULTS["diversity_weight"])),
        "local_bg_loss_weight": float(values.get("local_bg_loss_weight", DF_IRIS_MODEL_DEFAULTS["local_bg_loss_weight"])),
        "local_patch_radius": int(values.get("local_patch_radius", DF_IRIS_MODEL_DEFAULTS["local_patch_radius"])),
        "local_feature_stage": str(values.get("local_feature_stage", "stage3")),
        "token_prior_radius": values.get("token_prior_radius", None),
        "cdm_negative_topq": int(values.get("cdm_negative_topq", DF_IRIS_MODEL_DEFAULTS["cdm_negative_topq"])),
        "nms_radius": int(values.get("nms_radius", DF_IRIS_MODEL_DEFAULTS["nms_radius"])),
        "saliency_intersection_weight": float(
            values.get("saliency_intersection_weight", DF_IRIS_MODEL_DEFAULTS["saliency_intersection_weight"])
        ),
        "arcface_scale": float(values.get("arcface_scale", DF_IRIS_MODEL_DEFAULTS["arcface_scale"])),
        "arcface_margin": float(values.get("arcface_margin", DF_IRIS_MODEL_DEFAULTS["arcface_margin"])),
        "arcface_warmup_epochs": int(values.get("arcface_warmup_epochs", DF_IRIS_MODEL_DEFAULTS["arcface_warmup_epochs"])),
    }


@torch.no_grad()
def init_local_heads_from_global(model: MultiBranchResNet) -> None:
    model.head_local.weight.copy_(model.head_global.weight)
    model.head_integrated.weight.copy_(model.head_global.weight)


def classifier_for_epoch(epoch: int, arcface_start_epoch: int) -> str:
    return "arcface" if epoch >= arcface_start_epoch else "softmax"


def build_optimizer(model: nn.Module, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    return torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay)


def build_scheduler(optimizer: torch.optim.Optimizer, epochs: int, min_lr: float) -> torch.optim.lr_scheduler.LRScheduler:
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs), eta_min=min_lr)


def train_one_epoch(
    model: MultiBranchResNet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    amp_enabled: bool,
    grad_clip: float,
    classifier: str,
    classifier_epoch: int,
) -> dict[str, float]:
    model.train()
    loss_sum = 0.0
    correct = 0
    total = 0
    steps = 0
    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs}", leave=False)
    for images, labels, _, _ in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled and device.type == "cuda"):
            loss, logits, _ = model(images, labels, current_epoch=classifier_epoch, total_epochs=total_epochs, classifier=classifier)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        before = float(scaler.get_scale())
        scaler.step(optimizer)
        scaler.update()
        if not scaler.is_enabled() or float(scaler.get_scale()) >= before:
            steps += 1
        metric_labels = labels if logits.size(0) == labels.size(0) else labels.repeat_interleave(logits.size(0) // labels.size(0))
        loss_sum += float(loss.item()) * metric_labels.size(0)
        correct += int((logits.argmax(dim=1) == metric_labels).sum().item())
        total += int(metric_labels.size(0))
        pbar.set_postfix(loss=f"{loss_sum / max(total, 1):.4f}", acc=f"{correct / max(total, 1):.3f}")
    return {"loss": loss_sum / max(total, 1), "accuracy": correct / max(total, 1), "optimizer_steps": steps}


def evaluate_split(
    model: MultiBranchResNet,
    dataset: DFIrisDataset,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> dict[str, Any]:
    previous_stage = model.train_stage
    model.train_stage = "full"
    try:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, **dataloader_kwargs(num_workers, train=False))
        features, labels, _ = extract_embeddings(model, loader, device)
        return compute_metrics(features, labels)
    finally:
        model.train_stage = previous_stage


@torch.no_grad()
def extract_embeddings(model: MultiBranchResNet, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    features: list[np.ndarray] = []
    labels: list[str] = []
    paths: list[str] = []
    for images, _, identities, image_paths in tqdm(loader, desc="Extract", leave=False):
        images = images.to(device, non_blocking=True)
        out = model(images, labels=None)
        features.append(F.normalize(out["integrated_feat"], dim=1).detach().cpu().numpy().astype(np.float32))
        labels.extend(list(identities))
        paths.extend(list(image_paths))
    return np.concatenate(features, axis=0), np.asarray(labels), np.asarray(paths)


def save_features(
    model: MultiBranchResNet,
    dataset: DFIrisDataset,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    output_path: Path,
    split: str,
) -> Path:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, **dataloader_kwargs(num_workers, train=False))
    features, labels, paths = extract_embeddings(model, loader, device)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        features=features,
        labels=labels,
        image_paths=paths,
        split=split,
        model="DF-Iris",
        embedding_name="integrated_feat",
    )
    return output_path


def train(args: argparse.Namespace) -> None:
    cfg = load_config(Path(args.config))
    data = section(cfg, "data")
    output = section(cfg, "output")
    training = section(cfg, "training")
    output_dir = required_path(output.get("output_dir"), "output.output_dir")
    manifest = optional_path(data.get("manifest"))
    data_root = optional_path(data.get("data_root"))
    if manifest is None and data_root is None:
        raise ValueError("data.data_root or data.manifest must be set.")
    seed = int(data.get("seed", 20260521))
    set_seed(seed)
    image_size = int(training.get("image_size", 224))
    if manifest is not None:
        protocol = build_protocol_from_manifest(manifest, output_dir, image_size)
    else:
        protocol = build_protocol(
            data_root,
            output_dir,
            data.get("max_subjects"),
            seed,
            image_size,
            str(data.get("data_layout", "subject_eye")),
        )
    samples = protocol["samples"]
    assert_subject_disjoint(samples)
    train_samples = samples_for(samples, "train")
    val_samples = samples_for(samples, "val")
    test_samples = samples_for(samples, "test")
    labels = train_label_map(train_samples)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    mcfg = model_config(cfg)
    model = MultiBranchResNet(num_classes=len(labels), train_stage="global", **mcfg).to(device)
    load_optional_checkpoint(model, optional_path(section(cfg, "model").get("pretrained_checkpoint")), device)
    train_dataset = DFIrisDataset(
        train_samples,
        labels,
        train=True,
        augment=AugmentConfig(str(training.get("augment_level", "baseline"))),
        image_size=image_size,
        cache_images=bool(training.get("cache_images", False)),
    )
    val_dataset = DFIrisDataset(val_samples, None, train=False, image_size=image_size, cache_images=bool(training.get("cache_images", False)))
    test_dataset = DFIrisDataset(test_samples, None, train=False, image_size=image_size, cache_images=bool(training.get("cache_images", False)))
    batch_size = int(training.get("batch_size", 128))
    eval_batch_size = int(training.get("eval_batch_size", batch_size))
    num_workers = int(training.get("num_workers", 0))
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        **dataloader_kwargs(num_workers, train=True),
    )
    stage_plan = (
        ("global", int(training.get("global_epochs", 20))),
        ("local", int(training.get("local_epochs", 0))),
        ("integrated", int(training.get("integrated_epochs", 0))),
        ("full", int(training.get("full_epochs", 80))),
    )
    total_epochs = sum(epochs for _, epochs in stage_plan)
    if total_epochs <= 0:
        raise ValueError("At least one training epoch is required.")
    lr = float(training.get("lr", 1e-4))
    min_lr = float(training.get("min_lr", 1e-6))
    weight_decay = float(training.get("weight_decay", 1e-4))
    grad_clip = float(training.get("grad_clip", 5.0))
    arcface_start_epoch = int(training.get("arcface_start_epoch", 21))
    arcface_lr = float(training.get("arcface_lr", 5e-5))
    arcface_min_lr = float(training.get("arcface_min_lr", 1e-6))
    arcface_weight_decay = float(training.get("arcface_weight_decay", weight_decay))
    eval_every = int(training.get("eval_every", 10))
    amp_enabled = not bool(training.get("no_amp", False))
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled and device.type == "cuda", init_scale=float(training.get("amp_init_scale", 4096.0)))
    best = {"eer": math.inf, "epoch": 0, "stage": "", "classifier": ""}
    history: list[dict[str, Any]] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    history_jsonl = output_dir / "history.jsonl"
    history_jsonl.write_text("", encoding="utf-8")
    epoch_counter = 0
    heads_initialized = False
    for stage, epochs in stage_plan:
        if epochs <= 0:
            continue
        if stage != "global" and not heads_initialized and int(training.get("global_epochs", 20)) > 0:
            init_local_heads_from_global(model)
            heads_initialized = True
        model.train_stage = stage
        optimizer = build_optimizer(model, lr, weight_decay)
        scheduler = build_scheduler(optimizer, epochs, min_lr)
        optimizer_mode = "softmax"
        for local_epoch in range(1, epochs + 1):
            epoch_counter += 1
            classifier = classifier_for_epoch(epoch_counter, arcface_start_epoch)
            classifier_epoch = epoch_counter if classifier == "softmax" else epoch_counter - arcface_start_epoch + 1
            if classifier == "arcface" and optimizer_mode != "arcface":
                optimizer = build_optimizer(model, arcface_lr, arcface_weight_decay)
                scheduler = build_scheduler(optimizer, epochs - local_epoch + 1, arcface_min_lr)
                optimizer_mode = "arcface"
            stats = train_one_epoch(model, train_loader, optimizer, scaler, device, epoch_counter, total_epochs, amp_enabled, grad_clip, classifier, classifier_epoch)
            scheduler.step()
            row: dict[str, Any] = {
                "epoch": epoch_counter,
                "stage": stage,
                "classifier": classifier,
                "lr": scheduler.get_last_lr()[0],
                "train_loss": stats["loss"],
                "train_accuracy": stats["accuracy"],
                "train_optimizer_steps": stats["optimizer_steps"],
            }
            if eval_every > 0 and (epoch_counter % eval_every == 0 or epoch_counter == total_epochs):
                metrics = evaluate_split(model, val_dataset, device, eval_batch_size, num_workers)
                row.update({f"val_{k}": v for k, v in metrics.items()})
                if metrics["eer"] < best["eer"]:
                    best = {"eer": metrics["eer"], "epoch": epoch_counter, "stage": stage, "classifier": classifier}
                    torch.save(
                        {
                            "epoch": epoch_counter,
                            "stage": stage,
                            "classifier": classifier,
                            "state_dict": model.state_dict(),
                            "label_map": labels,
                            "model_config": mcfg,
                            "config": cfg,
                        },
                        output_dir / "best.pt",
                    )
            torch.save(
                {
                    "epoch": epoch_counter,
                    "stage": stage,
                    "classifier": classifier,
                    "state_dict": model.state_dict(),
                    "label_map": labels,
                    "model_config": mcfg,
                    "config": cfg,
                },
                output_dir / "last.pt",
            )
            history.append(row)
            with history_jsonl.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
    best_path = output_dir / "best.pt"
    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device, weights_only=False)["state_dict"])
    else:
        torch.save({"state_dict": model.state_dict(), "label_map": labels, "model_config": mcfg, "config": cfg}, best_path)
    features_path = save_features(model, test_dataset, device, eval_batch_size, num_workers, output_dir / "features_test.npz", "test")
    test_data = np.load(features_path, allow_pickle=False)
    result = {
        "protocol": protocol["summary"],
        "best": best,
        "features": str(features_path),
        "embedding_name": "integrated_feat",
        "model": "DF-Iris",
        "augmentation": {"level": str(training.get("augment_level", "baseline")), "image_size": image_size},
        "stage_plan": [{"stage": stage, "epochs": epochs} for stage, epochs in stage_plan],
        "test_metrics": compute_metrics(test_data["features"].astype(np.float32), test_data["labels"].astype(str)),
    }
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (output_dir / "history.csv").open("w", newline="", encoding="utf-8") as f:
        fields = sorted({key for row in history for key in row})
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(history)
    print(json.dumps(result, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser("DF-Iris")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
