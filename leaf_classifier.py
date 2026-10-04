"""Data and decisions for the replaceable coffee-leaf classification stage."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random

import numpy as np
from PIL import Image, ImageOps
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from coffee import check_overlap

LEAF_TASK = "coffee_leaf_classifier"
CONDITION_TASK = "coffee_condition_classifier"
LEAF_CLASSES = ["not_coffee_leaf", "coffee_leaf"]
LEAF_THRESHOLDS = {"reject": 0.2, "accept": 0.8}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def leaf_status(score, thresholds=LEAF_THRESHOLDS):
    if not np.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("Leaf score must be finite and in [0,1]")
    if not 0 <= thresholds["reject"] < thresholds["accept"] <= 1:
        raise ValueError("Expected 0 <= reject < accept <= 1")
    if score >= thresholds["accept"]:
        return "coffee_leaf"
    if score <= thresholds["reject"]:
        return "not_coffee_leaf"
    return "uncertain"


def split_coco(paths, seed=42):
    paths = sorted(paths)
    if len(paths) != 128:
        raise ValueError("Automatic COCO split expects 128 images; use --manifest for other sources/counts")
    random.Random(seed).shuffle(paths)
    return {"train": paths[:90], "valid": paths[90:109], "test": paths[109:]}


def image_paths(folder):
    paths = sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    if not paths:
        raise ValueError(f"No images in {folder}")
    return paths


def build_leaf_records(positive_root, negative_root, seed=42):
    records = []
    for split in ("train", "valid", "test"):
        for p in image_paths(Path(positive_root) / split / "images"):
            records.append({"path": str(p.resolve()), "label": "coffee_leaf", "split": split,
                            "source": "bracol", "group_id": f"bracol:{p.stem.split('.rf.')[0]}"})
    for split, paths in split_coco(image_paths(Path(negative_root) / "coco128"), seed).items():
        for p in paths:
            records.append({"path": str(p.resolve()), "label": "not_coffee_leaf", "split": split,
                            "source": "coco128", "group_id": f"coco128:{p.stem}"})
    for p in image_paths(Path(negative_root) / "phone"):
        records.append({"path": str(p.resolve()), "label": "not_coffee_leaf", "split": "phone",
                        "source": "phone", "group_id": "phone_capture_session"})
    return records


def audit_leaf_records(records):
    """Validate arbitrary sources, keeping explicit capture groups in one partition."""
    audited, groups, paths = [], defaultdict(set), set()
    for original in records:
        row = dict(original)
        if row.get("label") not in LEAF_CLASSES or row.get("split") not in {"train", "valid", "test", "phone"}:
            raise ValueError("Manifest requires a supported label and train/valid/test/phone split")
        if not row.get("source"):
            raise ValueError("Manifest rows need a source")
        if row["source"] == "phone" and row["split"] != "phone":
            raise ValueError("Phone photos must stay in the independent phone evaluation partition")
        p = Path(row["path"]).resolve()
        if str(p).casefold() in paths:
            raise ValueError(f"Repeated image path: {p}")
        paths.add(str(p).casefold())
        row["path"] = str(p)
        row["group_id"] = row.get("group_id") or f"{row['source']}:{p.stem}"
        groups[row["group_id"]].add(row["split"])
        with Image.open(p) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            im.load()
            row["size"] = list(im.size)
            row["pixel_sha256"] = hashlib.sha256(str(im.size).encode() + im.tobytes()).hexdigest()
        row["source_id"] = row["group_id"]
        row["target"] = [int(c == row["label"]) for c in LEAF_CLASSES]
        audited.append(row)
    if any(len(splits) > 1 for splits in groups.values()):
        raise ValueError("Capture group overlaps multiple partitions")
    validate_composite_sources(audited)
    check_overlap(audited)
    splits = {}
    for split in ("train", "valid", "test", "phone"):
        rows = [r for r in audited if r["split"] == split]
        counts = Counter(r["label"] for r in rows)
        if split != "phone" and any(counts[c] == 0 for c in LEAF_CLASSES):
            raise ValueError(f"Both classes must be represented in {split}")
        splits[split] = {"images": len(rows), "class_counts": dict(counts),
                         "sources": dict(Counter(r["source"] for r in rows))}
    phone = [r for r in audited if r["split"] == "phone"]
    if not phone or any(r["label"] != "not_coffee_leaf" for r in phone):
        raise ValueError("The phone evaluation partition must contain only held-out negatives")
    report = {"task": LEAF_TASK, "classes": LEAF_CLASSES, "splits": splits,
              "cross_partition_overlap": 0,
              "within_partition_duplicate_groups": sum(n > 1 for n in Counter(
                  (r["split"], r["pixel_sha256"]) for r in audited).values()),
              "limitations": ["BRACOL/plain-background versus COCO/scene background bias remains possible.",
                              "Phone evaluation contains negatives only and cannot measure phone-leaf recall.",
                              "Other-species rejection and image quality are not separately trained.",
                              "Filename/group checks do not establish plant-level independence."]}
    print(json.dumps(splits, indent=2), flush=True)
    return audited, report


def validate_composite_sources(records):
    """Require both composite dependencies to be audited training originals."""
    by_path = {str(Path(r["path"]).resolve()).casefold(): r for r in records}
    for row in records:
        if row["source"] == "background_crop":
            parent = by_path.get(str(Path(row.get("parent_path", "")).resolve()).casefold())
            if (row["split"] != "train" or row["label"] != "not_coffee_leaf" or parent is None
                    or parent["split"] != "train" or parent["label"] != "not_coffee_leaf"
                    or parent["source"] in {"background_crop", "bracol_composite"}):
                raise ValueError("Background crops require original training negatives")
            if (row["group_id"] != parent["group_id"]
                    or row.get("parent_group_id") != parent["group_id"]
                    or row.get("parent_pixel_sha256") != parent["pixel_sha256"]):
                raise ValueError("Background crop provenance does not match its source")
            continue
        if row["source"] != "bracol_composite":
            continue
        if row["split"] != "train" or row["label"] != "coffee_leaf":
            raise ValueError("Composites must be training coffee-leaf positives")
        for prefix, label in (("parent", "coffee_leaf"), ("background", "not_coffee_leaf")):
            dependency = by_path.get(str(Path(row.get(prefix + "_path", "")).resolve()).casefold())
            if dependency is None or dependency["split"] != "train" or dependency["label"] != label:
                raise ValueError(f"Composite {prefix} must be an audited training {label} image")
            if dependency["source"] == "bracol_composite":
                raise ValueError("Composite dependencies must be original images")
            if any(row.get(prefix + "_" + key) != dependency[key] for key in ("group_id", "pixel_sha256")):
                raise ValueError(f"Composite {prefix} provenance does not match its source")
            if prefix == "parent" and row["group_id"] != dependency["group_id"]:
                raise ValueError("Composite must keep its parent's capture group")
        mask_path = Path(row.get("mask_path", ""))
        if not mask_path.is_file() or hashlib.sha256(mask_path.read_bytes()).hexdigest() != row.get("mask_sha256"):
            raise ValueError("Composite mask provenance does not match")


def load_leaf_manifest(path):
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload["classes"] != LEAF_CLASSES:
        raise ValueError(f"Expected manifest classes {LEAF_CLASSES}")
    rows = payload["records"]
    for row in rows:
        for key in ("path", "parent_path", "background_path", "mask_path"):
            if key in row:
                p = Path(row[key])
                row[key] = str(p if p.is_absolute() else (path.parent / p).resolve())
    return rows


def leaf_metrics(targets, scores, thresholds=LEAF_THRESHOLDS):
    targets, scores = np.asarray(targets), np.asarray(scores)
    actual = targets.argmax(1)
    predicted = scores.argmax(1)
    precision, recall, f1, support = precision_recall_fscore_support(
        actual, predicted, labels=[0, 1], zero_division=0)
    matrix = confusion_matrix(actual, predicted, labels=[0, 1])
    statuses = np.array([leaf_status(float(v), thresholds) for v in scores[:, 1]])
    result = {"samples": len(actual), "accuracy": float(accuracy_score(actual, predicted)),
              "f1_macro": float(f1.mean()), "confusion_matrix": matrix.tolist(), "per_class": {},
              "operating_thresholds": thresholds,
              "operating_counts": dict(Counter(statuses.tolist())),
              "negative_images_accepted": int(((actual == 0) & (statuses == "coffee_leaf")).sum()),
              "positive_images_rejected": int(((actual == 1) & (statuses == "not_coffee_leaf")).sum())}
    for i, c in enumerate(LEAF_CLASSES):
        result["per_class"][c] = {"precision": float(precision[i]), "recall": float(recall[i]),
                                   "f1": float(f1[i]), "support": int(support[i])}
    if (actual == 1).sum() == 0:
        result["interpretation"] = "Negative-only evaluation: macro-F1 is not a balanced overall performance estimate"
        result["negative_rejection_rate"] = float((statuses == "not_coffee_leaf").mean())
        result["negative_acceptance_rate"] = float((statuses == "coffee_leaf").mean())
    return result
