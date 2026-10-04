"""Shared dataset, model, preprocessing, and multilabel evaluation code."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision import models, transforms
import yaml
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             multilabel_confusion_matrix)

PREPROCESS = {
    "size": 224, "resize": "aspect_preserving_fit", "padding": "centered",
    "fill_rgb": [255, 255, 255], "interpolation": "bilinear",
    "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
    "orientation": "EXIF transpose", "color": "RGB",
}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def parse_label(path, nclasses):
    target = [0] * nclasses
    for number, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            values = [float(v) for v in line.split()]
            if len(values) != 5 or not np.isfinite(values).all():
                raise ValueError("expected five finite values")
            c, x, y, w, h = values
            if not c.is_integer() or not 0 <= c < nclasses:
                raise ValueError("invalid class ID")
            if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                raise ValueError("invalid normalized box")
            if min(x-w/2, y-h/2) < -0.001 or max(x+w/2, y+h/2) > 1.001:
                raise ValueError("box extends outside image")
            target[int(c)] = 1
        except ValueError as exc:
            raise ValueError(f"{path}:{number}: {exc}") from exc
    return target


def check_overlap(records):
    """Reject source-ID or decoded-pixel overlap across partitions."""
    for key in ("source_id", "pixel_sha256"):
        groups = defaultdict(set)
        for row in records:
            groups[row[key]].add(row["split"])
        leaked = [k for k, splits in groups.items() if len(splits) > 1]
        if leaked:
            raise ValueError(f"Cross-split overlap by {key}: {leaked[:5]}")


def audit_dataset(root):
    root = Path(root)
    config = yaml.safe_load((root / "data.yaml").read_text())
    classes = config["names"]
    if isinstance(classes, dict):
        classes = [classes[i] for i in range(len(classes))]
    if not classes or len(set(classes)) != len(classes) or config["nc"] != len(classes):
        raise ValueError("Invalid class mapping in data.yaml")
    records = []
    for split in ("train", "valid", "test"):
        folder = root / split
        images = sorted(p for p in (folder / "images").iterdir()
                        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"})
        labels = {p.stem: p for p in (folder / "labels").glob("*.txt")}
        if not images or len({p.stem for p in images}) != len(images):
            raise ValueError(f"Empty split or repeated image stems: {split}")
        if {p.stem for p in images} != set(labels):
            raise ValueError(f"Image/label pairing mismatch in {split}")
        for p in images:
            target = parse_label(labels[p.stem], len(classes))
            with Image.open(p) as im:
                im = ImageOps.exif_transpose(im).convert("RGB")
                im.load()
                digest = hashlib.sha256(str(im.size).encode() + im.tobytes()).hexdigest()
                size = list(im.size)
            records.append({"path": p.relative_to(root).as_posix(), "split": split,
                            "target": target, "source_id": p.stem.split(".rf.")[0],
                            "pixel_sha256": digest, "size": size})
        print(f"Audited {split}: {len(images)} images", flush=True)
    check_overlap(records)
    report = {"classes": classes, "splits": {}, "cross_split_duplicates": 0,
              "within_split_duplicate_groups": sum(v > 1 for v in Counter(
                  (r["split"], r["pixel_sha256"]) for r in records).values()),
              "empty_annotation_meaning": "No annotated target condition; not verified healthy",
              "limitations": ["Absent annotations are assumed negative for the four target conditions.",
                              "Plant-level independence is unknown; source IDs and pixels were checked.",
                              "Field photos, other species, and overall plant health are not validated."]}
    for split in ("train", "valid", "test"):
        rows = [r for r in records if r["split"] == split]
        targets = np.array([r["target"] for r in rows])
        report["splits"][split] = {
            "images": len(rows), "positive_images": dict(zip(classes, targets.sum(0).tolist())),
            "empty_labels": int((targets.sum(1) == 0).sum()),
            "label_combinations": dict(Counter(" + ".join(c for c, present in zip(classes, r["target"])
                                                        if present) or "none annotated" for r in rows))}
        if (targets.sum(0) == 0).any() or (targets.sum(0) == len(rows)).any():
            raise ValueError(f"Every class needs positive and negative examples in {split}")
    return classes, records, report


def fit_image(image, config=PREPROCESS):
    image = ImageOps.exif_transpose(image).convert("RGB")
    size = config["size"]
    w, h = image.size
    scale = min(size / w, size / h)
    resized = image.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (size, size), tuple(config["fill_rgb"]))
    canvas.paste(resized, ((size-resized.width)//2, (size-resized.height)//2))
    return canvas


def tensor_transform(training=False, config=PREPROCESS):
    steps = []
    if training:
        steps += [transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
                  transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02)]
    return transforms.Compose(steps + [transforms.ToTensor(),
                                      transforms.Normalize(config["mean"], config["std"])])


class LeafDataset(Dataset):
    def __init__(self, root, records, training=False):
        self.records = records
        self.transform = tensor_transform(training)
        # Keep only resized RGB images in RAM to avoid decoding 2048px JPEGs each epoch.
        self.images = []
        for row in records:
            with Image.open(Path(root) / row["path"]) as im:
                self.images.append(fit_image(im))
        self.targets = torch.tensor([r["target"] for r in records], dtype=torch.float32)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return self.transform(self.images[index]), self.targets[index]


def make_model(nclasses, pretrained=False):
    weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.mobilenet_v3_small(weights=weights, width_mult=1.0)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, nclasses)
    return model


def calculate_metrics(targets, scores, classes, threshold=0.5):
    targets = np.asarray(targets, dtype=int)
    predicted = (np.asarray(scores) >= threshold).astype(int)
    precision, recall, f1, support = precision_recall_fscore_support(
        targets, predicted, average=None, zero_division=0)
    result = {"exact_match_accuracy": float(accuracy_score(targets, predicted)),
              "mean_label_accuracy": float((targets == predicted).mean()),
              "threshold": threshold, "samples": len(targets), "per_class": {}}
    for average in ("macro", "micro", "weighted"):
        result[f"f1_{average}"] = float(precision_recall_fscore_support(
            targets, predicted, average=average, zero_division=0)[2])
    matrices = multilabel_confusion_matrix(targets, predicted)
    for i, name in enumerate(classes):
        result["per_class"][name] = {"precision": float(precision[i]), "recall": float(recall[i]),
            "f1": float(f1[i]), "support": int(support[i]),
            "accuracy": float((targets[:, i] == predicted[:, i]).mean()),
            "confusion_matrix": matrices[i].tolist()}
    return result
