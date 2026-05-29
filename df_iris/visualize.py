from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from df_iris.dataset import build_eval_transform, collect_image_paths
from df_iris.model import MultiBranchResNet

VISUALIZATION_FILES = (
    "Input Eye Image.png",
    "Global Spatial Prior.png",
    "Target-Identity CAM.png",
    "Class-Difference Map.png",
    "Selected Texture Patches.png",
    "Local Texture Response.png",
)


def _to_uint8(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32)
    values = values - float(values.min())
    denom = float(values.max())
    if denom > 1e-12:
        values = values / denom
    return np.clip(values * 255.0, 0, 255).astype(np.uint8)


def _flat_map_to_image(values: torch.Tensor, image_size: int) -> Image.Image:
    arr = values.detach().cpu().float().numpy()
    side = int(round(math.sqrt(arr.size)))
    if side * side != arr.size:
        raise ValueError(f"Expected a square map, got {arr.size} values.")
    image = Image.fromarray(_to_uint8(arr.reshape(side, side)), mode="L")
    return image.resize((image_size, image_size), Image.Resampling.BILINEAR)


def _heatmap(gray: Image.Image, response: Image.Image, alpha: float = 0.45) -> Image.Image:
    base = gray.convert("RGB")
    response = response.resize(base.size, Image.Resampling.BILINEAR)
    r = np.asarray(response, dtype=np.float32) / 255.0
    heat = np.zeros((*r.shape, 3), dtype=np.uint8)
    heat[..., 0] = np.clip(255 * r, 0, 255)
    heat[..., 1] = np.clip(255 * np.maximum(0.0, 1.0 - np.abs(r - 0.55) * 2.0), 0, 255)
    heat[..., 2] = np.clip(255 * (1.0 - r), 0, 255)
    return Image.blend(base, Image.fromarray(heat, mode="RGB"), alpha)


def _draw_selected_patches(image: Image.Image, indices: torch.Tensor, map_size: int, radius: int) -> Image.Image:
    out = image.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    cell_w = out.width / float(map_size)
    cell_h = out.height / float(map_size)
    patch = max(radius, 0) + 0.5
    for idx in indices.detach().cpu().numpy().tolist():
        row, col = divmod(int(idx), map_size)
        left = max(0.0, (col - patch) * cell_w)
        top = max(0.0, (row - patch) * cell_h)
        right = min(float(out.width - 1), (col + patch + 1.0) * cell_w)
        bottom = min(float(out.height - 1), (row + patch + 1.0) * cell_h)
        draw.rectangle((left, top, right, bottom), outline=(255, 220, 0), width=3)
    return out


def _sample_dir_name(path: Path) -> str:
    return path.stem.replace(" ", "_")


def _save_sample_outputs(
    sample_dir: Path,
    gray: Image.Image,
    global_prior: Image.Image,
    cam: Image.Image,
    cdm: Image.Image,
    selected: Image.Image,
    local_response: Image.Image,
) -> dict[str, str]:
    sample_dir.mkdir(parents=True, exist_ok=True)
    images = {
        "Input Eye Image.png": gray.convert("RGB"),
        "Global Spatial Prior.png": _heatmap(gray, global_prior),
        "Target-Identity CAM.png": _heatmap(gray, cam),
        "Class-Difference Map.png": _heatmap(gray, cdm),
        "Selected Texture Patches.png": selected,
        "Local Texture Response.png": _heatmap(gray, local_response),
    }
    outputs = {}
    for name in VISUALIZATION_FILES:
        output_path = sample_dir / name
        images[name].resize(gray.size, Image.Resampling.BILINEAR).save(output_path)
        outputs[name] = str(output_path)
    return outputs


@torch.no_grad()
def visualize(args: argparse.Namespace) -> None:
    checkpoint = torch.load(Path(args.checkpoint), map_location="cpu", weights_only=False)
    label_map = checkpoint.get("label_map")
    if not label_map:
        raise ValueError("Checkpoint must contain label_map.")
    model_config = checkpoint.get("model_config", {})
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = MultiBranchResNet(num_classes=len(label_map), train_stage="full", **model_config).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_paths = collect_image_paths(Path(args.input))[: int(args.max_images)]
    if not image_paths:
        raise ValueError("No supported images found.")

    transform = build_eval_transform(int(args.image_size))
    for path in image_paths:
        gray = Image.open(path).convert("L").resize((int(args.image_size), int(args.image_size)), Image.Resampling.BILINEAR)
        tensor = transform(Image.open(path).convert("L")).unsqueeze(0).to(device)
        out = model(tensor, labels=None)

        global_prior = _flat_map_to_image(out["global_prior"][0], int(args.image_size))
        cam = _flat_map_to_image(out["cam"][0], int(args.image_size))
        cdm = _flat_map_to_image(out["cdm"][0], int(args.image_size))
        local_response = _flat_map_to_image(out["local_saliency"][0], int(args.image_size))
        map_size = int(round(math.sqrt(out["local_saliency"].shape[1])))
        selected = _draw_selected_patches(gray, out["local_indices"][0], map_size, int(model.local_patch_radius))

        sample_dir = output_dir / _sample_dir_name(path)
        _save_sample_outputs(sample_dir, gray, global_prior, cam, cdm, selected, local_response)

    print(f"Saved {len(image_paths)} visualization samples to {output_dir}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser("DF-Iris visualization")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--max-images", type=int, default=16)
    parser.add_argument("--cpu", action="store_true")
    visualize(parser.parse_args())


if __name__ == "__main__":
    main()
