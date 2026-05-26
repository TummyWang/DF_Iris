from __future__ import annotations

import csv
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


@dataclass(frozen=True)
class IrisSample:
    image_path: str
    subject: str
    eye: str
    identity: str
    split: str


@dataclass(frozen=True)
class AugmentConfig:
    level: str = "baseline"


class CenterSquareCrop:
    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        side = min(width, height)
        left = max(0, (width - side) // 2)
        top = max(0, (height - side) // 2)
        return image.crop((left, top, left + side, top + side))


class AddGaussianNoise:
    def __init__(self, std: float = 0.02, p: float = 0.25) -> None:
        self.std = std
        self.p = p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(()) >= self.p:
            return x
        return (x + torch.randn_like(x) * self.std).clamp(0.0, 1.0)


class DFIrisDataset(Dataset):
    def __init__(
        self,
        samples: list[IrisSample],
        label_map: dict[str, int] | None,
        train: bool,
        augment: AugmentConfig | None = None,
        image_size: int = 224,
        cache_images: bool = False,
    ) -> None:
        self.samples = samples
        self.label_map = label_map
        self._image_cache = [_load_base_image(s.image_path, image_size) for s in samples] if cache_images else None
        if train:
            self.tf = build_train_transform(augment or AugmentConfig(), image_size, include_base_steps=not cache_images)
        elif cache_images:
            self.tf = transforms.ToTensor()
        else:
            self.tf = build_eval_transform(image_size)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image = self._image_cache[index].copy() if self._image_cache is not None else Image.open(sample.image_path).convert("L")
        x = self.tf(image)
        y = -1 if self.label_map is None else self.label_map[sample.identity]
        return x, torch.tensor(y, dtype=torch.long), sample.identity, sample.image_path


class ImagePathDataset(Dataset):
    def __init__(self, image_paths: list[Path], image_size: int = 224) -> None:
        self.image_paths = image_paths
        self.tf = build_eval_transform(image_size)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        path = self.image_paths[index]
        return self.tf(Image.open(path).convert("L")), str(path)


def _rotation(degrees: float) -> transforms.RandomRotation:
    return transforms.RandomRotation(degrees, interpolation=transforms.InterpolationMode.BILINEAR, fill=128)


def _affine(**kwargs) -> transforms.RandomAffine:
    return transforms.RandomAffine(interpolation=transforms.InterpolationMode.BILINEAR, fill=128, **kwargs)


def _base_steps(image_size: int) -> list[object]:
    return [
        transforms.Grayscale(num_output_channels=1),
        CenterSquareCrop(),
        transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BILINEAR),
    ]


def _load_base_image(path: str, image_size: int) -> Image.Image:
    image = Image.open(path).convert("L")
    image = CenterSquareCrop()(image)
    return image.resize((image_size, image_size), Image.Resampling.BILINEAR)


def build_eval_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose([*_base_steps(image_size), transforms.ToTensor()])


def build_train_transform(
    augment: AugmentConfig,
    image_size: int = 224,
    *,
    include_base_steps: bool = True,
) -> transforms.Compose:
    base = _base_steps(image_size) if include_base_steps else []
    if augment.level == "none":
        return transforms.Compose([*base, transforms.ToTensor()])
    if augment.level == "baseline":
        return transforms.Compose(
            [
                *base,
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply([_rotation(15)], p=0.25),
                transforms.RandomApply([_affine(degrees=0, translate=(0.08, 0.08))], p=0.25),
                transforms.RandomApply([transforms.ColorJitter(brightness=0.15, contrast=0.15)], p=0.25),
                transforms.ToTensor(),
                AddGaussianNoise(std=0.02, p=0.25),
            ]
        )
    raise ValueError(f"Unknown augmentation level: {augment.level}")


def _iter_images(directory: Path) -> list[Path]:
    return sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def collect_image_paths(path: Path) -> list[Path]:
    if path.is_file():
        return [path] if path.suffix.lower() in IMAGE_EXTENSIONS else []
    return sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def _subject_has_images(subject_dir: Path, data_layout: str) -> bool:
    if data_layout == "subject_eye":
        return any(_iter_images(subject_dir / eye) for eye in ("L", "R") if (subject_dir / eye).is_dir())
    if data_layout == "subject_flat":
        return bool(_iter_images(subject_dir))
    raise ValueError(f"Unknown data layout: {data_layout}")


def collect_subjects(data_root: Path, max_subjects: int | None, seed: int, data_layout: str) -> list[Path]:
    subjects = sorted(p for p in data_root.iterdir() if p.is_dir() and _subject_has_images(p, data_layout))
    random.Random(seed).shuffle(subjects)
    if max_subjects is not None:
        subjects = subjects[:max_subjects]
    if len(subjects) < 3:
        raise ValueError("At least 3 subjects are required.")
    return subjects


def split_subjects(subjects: list[Path], min_eval_subjects: int = 1) -> dict[str, set[str]]:
    n = len(subjects)
    n_train = max(1, int(round(n * 0.70)))
    n_val = max(min_eval_subjects, int(round(n * 0.10)))
    if n_train + n_val >= n:
        n_train = max(1, n - min_eval_subjects * 2)
        n_val = min_eval_subjects
    if n_train < 1 or n_train + n_val >= n:
        raise ValueError("Not enough subjects for train, validation, and test splits.")
    return {
        "train": {p.name for p in subjects[:n_train]},
        "val": {p.name for p in subjects[n_train : n_train + n_val]},
        "test": {p.name for p in subjects[n_train + n_val :]},
    }


def build_protocol(
    data_root: Path,
    output_dir: Path,
    max_subjects: int | None,
    seed: int,
    image_size: int,
    data_layout: str,
) -> dict[str, Any]:
    subjects = collect_subjects(data_root, max_subjects, seed, data_layout)
    splits = split_subjects(subjects, min_eval_subjects=2 if data_layout == "subject_flat" else 1)
    samples: list[IrisSample] = []
    for subject_dir in subjects:
        subject = subject_dir.name
        split = next(name for name, values in splits.items() if subject in values)
        if data_layout == "subject_eye":
            for eye in ("L", "R"):
                eye_dir = subject_dir / eye
                if eye_dir.is_dir():
                    for image_path in _iter_images(eye_dir):
                        samples.append(IrisSample(str(image_path), subject, eye, f"{subject}_{eye}", split))
        elif data_layout == "subject_flat":
            for image_path in _iter_images(subject_dir):
                samples.append(IrisSample(str(image_path), subject, "U", subject, split))
        else:
            raise ValueError(f"Unknown data layout: {data_layout}")
    if not samples:
        raise ValueError(f"No supported images found under {data_root}.")
    return write_protocol(samples, output_dir, image_size, {"data_root": str(data_root), "split_policy": "subject_disjoint_70_10_20"})


def build_protocol_from_manifest(manifest_path: Path, output_dir: Path, image_size: int) -> dict[str, Any]:
    with manifest_path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        required = {"image_path", "subject", "eye", "identity", "split"}
        if reader.fieldnames is None or not required <= set(reader.fieldnames):
            raise ValueError(f"Manifest must contain columns {sorted(required)}.")
        samples = [IrisSample(row["image_path"], row["subject"], row["eye"], row["identity"], row["split"]) for row in reader]
    if not samples:
        raise ValueError("Manifest contains no samples.")
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "manifest.csv"
    if manifest_path.resolve() != target.resolve():
        shutil.copyfile(manifest_path, target)
    return write_protocol(samples, output_dir, image_size, {"source_manifest": str(manifest_path)})


def write_protocol(samples: list[IrisSample], output_dir: Path, image_size: int, extra: dict[str, Any]) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "subject", "eye", "identity", "split"])
        writer.writeheader()
        for sample in samples:
            writer.writerow(sample.__dict__)
    identities = {split: sorted({s.identity for s in samples if s.split == split}) for split in ("train", "val", "test")}
    if not all(identities.values()):
        raise ValueError("Each split must contain at least one identity.")
    summary = {
        **extra,
        "manifest": str(manifest_path),
        "samples": {split: sum(s.split == split for s in samples) for split in ("train", "val", "test")},
        "identities": {split: len(ids) for split, ids in identities.items()},
        "input": f"1x{image_size}x{image_size}",
        "embedding": "integrated_feat",
    }
    (output_dir / "protocol.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {"samples": samples, "summary": summary}


def samples_for(samples: list[IrisSample], split: str) -> list[IrisSample]:
    return [s for s in samples if s.split == split]


def train_label_map(train_samples: list[IrisSample]) -> dict[str, int]:
    return {identity: i for i, identity in enumerate(sorted({s.identity for s in train_samples}))}


def assert_subject_disjoint(samples: list[IrisSample]) -> None:
    split_subjects = {split: {s.subject for s in samples if s.split == split} for split in ("train", "val", "test")}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = split_subjects[left] & split_subjects[right]
        if overlap:
            raise AssertionError(f"Subject overlap between {left} and {right}: {sorted(overlap)}")
