from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from df_iris.dataset import ImagePathDataset, collect_image_paths
from df_iris.metrics import compute_metrics
from df_iris.model import MultiBranchResNet


def eval_features(args: argparse.Namespace) -> None:
    data = np.load(Path(args.features), allow_pickle=False)
    metrics = compute_metrics(data["features"].astype(np.float32), data["labels"].astype(str))
    output_path = Path(args.output) if args.output else Path(args.features).with_name("eval_metrics.json")
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)


@torch.no_grad()
def infer(args: argparse.Namespace) -> None:
    checkpoint = torch.load(Path(args.checkpoint), map_location="cpu", weights_only=False)
    label_map = checkpoint.get("label_map")
    if not label_map:
        raise ValueError("Checkpoint must contain label_map.")
    model_config = checkpoint.get("model_config", {})
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = MultiBranchResNet(num_classes=len(label_map), train_stage="full", **model_config).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    image_paths = collect_image_paths(Path(args.input))
    if not image_paths:
        raise ValueError("No supported images found.")
    dataset = ImagePathDataset(image_paths, image_size=int(args.image_size))
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=torch.cuda.is_available(),
    )
    features: list[np.ndarray] = []
    paths: list[str] = []
    for images, image_path_batch in tqdm(loader, desc="Infer", leave=False):
        images = images.to(device, non_blocking=True)
        out = model(images, labels=None)
        features.append(F.normalize(out["integrated_feat"], dim=1).detach().cpu().numpy().astype(np.float32))
        paths.extend(list(image_path_batch))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, features=np.concatenate(features, axis=0), image_paths=np.asarray(paths), model="DF-Iris")
    print(json.dumps({"output": str(output), "num_images": len(paths)}, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser("DF-Iris")
    sub = parser.add_subparsers(dest="command", required=True)
    eval_parser = sub.add_parser("eval-features")
    eval_parser.add_argument("--features", required=True)
    eval_parser.add_argument("--output", default="")
    eval_parser.set_defaults(func=eval_features)
    infer_parser = sub.add_parser("infer")
    infer_parser.add_argument("--checkpoint", required=True)
    infer_parser.add_argument("--input", required=True)
    infer_parser.add_argument("--output", required=True)
    infer_parser.add_argument("--image-size", type=int, default=224)
    infer_parser.add_argument("--batch-size", type=int, default=128)
    infer_parser.add_argument("--num-workers", type=int, default=0)
    infer_parser.add_argument("--cpu", action="store_true")
    infer_parser.set_defaults(func=infer)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
