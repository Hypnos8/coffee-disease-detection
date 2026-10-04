"""Cache BRACOL leaf masks and build grouped, training-only background composites."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import random

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps

from coffee import parse_label, write_json
from leaf_classifier import LEAF_CLASSES, audit_leaf_records, image_paths, load_leaf_manifest

ALGORITHM = "leaf_outline_grabcut_v4"


def rgb_image(path):
    with Image.open(path) as im:
        return ImageOps.exif_transpose(im).convert("RGB")


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pixel_hash(im):
    return hashlib.sha256(str(im.size).encode() + im.tobytes()).hexdigest()


def leaf_mask(rgb, max_side=1024, iterations=3):
    """Pale-border initialization; graph cut retains non-green lesion pixels too."""
    h, w = rgb.shape[:2]
    scale = min(1., max_side / max(h, w))
    small = cv2.resize(rgb, (round(w*scale), round(h*scale)), interpolation=cv2.INTER_AREA)
    sh, sw = small.shape[:2]
    lab = cv2.cvtColor(small, cv2.COLOR_RGB2LAB).astype(np.float32)
    border = np.concatenate((lab[0], lab[-1], lab[:, 0], lab[:, -1]))
    # Brightness varies across the pale backdrop and its shadows. Use chroma for
    # foreground seeds so glossy pale-green areas and brown lesions remain leaf.
    chroma = np.linalg.norm(lab[:, :, 1:] - np.median(border[:, 1:], axis=0), axis=2)
    noise = np.percentile(np.linalg.norm(border[:, 1:] - np.median(border[:, 1:], axis=0), axis=1), 95)
    hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
    green = ((hsv[:, :, 0] >= 25) & (hsv[:, :, 0] <= 100) & (hsv[:, :, 1] > 25)
             & (small[:, :, 1].astype(int) > small[:, :, 0].astype(int)+1))
    brown = (hsv[:, :, 1] > 100) & (chroma > max(14., float(noise)+4.))
    candidates = (green | brown).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(candidates)
    if n < 2:
        raise ValueError("No foreground candidate")
    candidate = (labels == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])).astype(np.uint8)
    # Keep enclosed glossy areas and necrosis as part of the leaf silhouette.
    # Open edge notches remain open; enclosed physical holes require visual review.
    flooded = np.pad(candidate, 1)
    cv2.floodFill(flooded, None, (0, 0), 1)
    candidate |= (flooded[1:-1, 1:-1] == 0).astype(np.uint8)
    seed = cv2.erode(candidate, np.ones((3, 3), np.uint8)) > 0
    if seed.sum() < 20:
        raise ValueError("Too few reliable foreground pixels")
    probable = cv2.dilate(candidate, np.ones((5, 5), np.uint8)) > 0
    gc = np.where(probable, cv2.GC_PR_FGD, cv2.GC_PR_BGD).astype(np.uint8)
    # Do not let a neutral cast shadow elsewhere in the photograph grow into leaf.
    gc[~probable] = cv2.GC_BGD
    gc[seed] = cv2.GC_FGD
    gc[[0, -1], :] = cv2.GC_BGD
    gc[:, [0, -1]] = cv2.GC_BGD
    cv2.grabCut(small, gc, None, np.zeros((1, 65)), np.zeros((1, 65)), iterations, cv2.GC_INIT_WITH_MASK)
    binary = np.isin(gc, [cv2.GC_FGD, cv2.GC_PR_FGD]).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    if n < 2:
        raise ValueError("Graph cut found no leaf")
    binary = (labels == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])).astype(np.uint8)
    # Refine a narrow edge band at native resolution, rather than keeping a jagged resized mask.
    binary = cv2.resize(binary, (w, h), interpolation=cv2.INTER_NEAREST)
    if scale < 1:
        kernel = np.ones((7, 7), np.uint8)
        inner, outer = cv2.erode(binary, kernel), cv2.dilate(binary, kernel)
        gc = np.where(binary, cv2.GC_PR_FGD, cv2.GC_PR_BGD).astype(np.uint8)
        gc[outer == 0], gc[inner == 1] = cv2.GC_BGD, cv2.GC_FGD
        cv2.grabCut(rgb, gc, None, np.zeros((1, 65)), np.zeros((1, 65)), 1, cv2.GC_INIT_WITH_MASK)
        binary = np.isin(gc, [cv2.GC_FGD, cv2.GC_PR_FGD]).astype(np.uint8)
    return binary * 255


def mask_quality(mask, label_path=None):
    fg = mask > 0
    issues = []
    fraction = float(fg.mean())
    if not .08 <= fraction <= .80:
        issues.append("foreground_area_out_of_range")
    if fg[0].any() or fg[-1].any() or fg[:, 0].any() or fg[:, -1].any():
        issues.append("foreground_touches_border")
    n, _, stats, _ = cv2.connectedComponentsWithStats(fg.astype(np.uint8))
    if n > 1 and stats[1:, cv2.CC_STAT_AREA].max() / max(1, fg.sum()) < .98:
        issues.append("fragmented_foreground")
    contours, hierarchy = cv2.findContours(fg.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    hole_area = 0
    if hierarchy is not None:
        hole_area = sum(cv2.contourArea(c) for c, link in zip(contours, hierarchy[0]) if link[3] >= 0)
    if hole_area > max(100, fg.sum() * .002):
        issues.append("interior_holes_need_review")
    coverages = []
    if label_path and Path(label_path).exists():
        h, w = mask.shape
        for line in Path(label_path).read_text().splitlines():
            if not line.strip():
                continue
            _, x, y, bw, bh = map(float, line.split())
            x0, x1 = max(0, int((x-bw/2)*w)), min(w, int(np.ceil((x+bw/2)*w)))
            y0, y1 = max(0, int((y-bh/2)*h)), min(h, int(np.ceil((y+bh/2)*h)))
            coverage = float(fg[y0:y1, x0:x1].mean())
            coverages.append(coverage)
            if coverage < .65:
                issues.append("annotated_symptom_region_mostly_missing")
    return {"foreground_fraction": fraction, "symptom_box_coverages": coverages,
            "issues": sorted(set(issues)), "automatic_checks_passed": not issues}


def composite(leaf, mask, background):
    if mask.size != leaf.size or set(np.unique(np.array(mask))) - {0, 255}:
        raise ValueError("Mask must be binary and match the original dimensions")
    bg = ImageOps.fit(background.convert("RGB"), leaf.size, method=Image.Resampling.LANCZOS)
    result = Image.composite(leaf, bg, mask)
    keep = np.array(mask) == 255
    if not np.array_equal(np.array(result)[keep], np.array(leaf)[keep]):
        raise AssertionError("Foreground pixels changed")
    return result


def label_path(row):
    p = Path(row["path"])
    return p.parent.parent / "labels" / (p.stem + ".txt")


def add_matching_background_negatives(records, backgrounds, sizes, output):
    """Same fitted canvas without a leaf: removes an aspect-ratio/padding cue."""
    folder = Path(output) / "background_negatives"
    folder.mkdir()
    added = []
    for index, row in enumerate(backgrounds):
        for w, h in sorted(sizes):
            fitted = ImageOps.fit(rgb_image(row["path"]), (w, h), method=Image.Resampling.LANCZOS)
            p = folder / f"background_{index:03d}_{w}x{h}.png"
            fitted.save(p)
            added.append({"path": str(p.resolve()), "label": LEAF_CLASSES[0], "split": "train",
                          "source": "background_crop", "group_id": row["group_id"],
                          "parent_path": row["path"], "parent_group_id": row["group_id"],
                          "parent_pixel_sha256": row["pixel_sha256"],
                          "crop": "centered aspect-fill Lanczos", "size": [w, h]})
    records.extend(added)
    return len(added)


def select_leaves(rows, limit, seed):
    # Round-robin across no annotation and the four symptom classes for a diverse pilot.
    buckets = [[] for _ in range(5)]
    for row in sorted(rows, key=lambda r: r["path"]):
        target = parse_label(label_path(row), 4)
        category = next((i+1 for i, present in enumerate(target) if present), 0)
        buckets[category].append(row)
    rng = random.Random(seed)
    for bucket in buckets:
        rng.shuffle(bucket)
    ordered = []
    while any(buckets):
        for bucket in buckets:
            if bucket:
                ordered.append(bucket.pop())
    return ordered[:limit] if limit else ordered


def contact_sheets(items, folder, prefix, rows_per_sheet=10):
    """Original, alpha overlay, and composite; labels mark exclusions."""
    folder.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(items), rows_per_sheet):
        batch = items[start:start+rows_per_sheet]
        sheet = Image.new("RGB", (1200, 180*len(batch)), "white")
        draw = ImageDraw.Draw(sheet)
        for i, item in enumerate(batch):
            original = rgb_image(item["parent_path"])
            mask = Image.open(item["mask_path"]).convert("L")
            overlay = Image.composite(original, Image.new("RGB", original.size, (220, 40, 160)), mask)
            previews = [rgb_image(p) for p in item.get("preview_paths", [])[:2]]
            previews += [overlay] * (2-len(previews))
            for j, im in enumerate((original, overlay, *previews)):
                sheet.paste(ImageOps.contain(im, (300, 150)), (j*300, i*180))
            draw.text((4, i*180+151), Path(item["parent_path"]).stem.split(".rf.")[0] +
                      " | " + (", ".join(item["quality"]["issues"]) or "automatic checks passed"), fill="black")
        sheet.save(folder / f"{prefix}_{start//rows_per_sheet+1:03d}.jpg", quality=90)


def prepare_mask(row, cache, overrides, max_side, seed):
    p = Path(row["path"])
    im = rgb_image(p)
    key = p.stem
    target = cache / (key + ".png")
    info_path = cache / (key + ".json")
    override = overrides / (key + ".png") if overrides else None
    fingerprint = {"parent_sha256": file_hash(p), "algorithm": ALGORITHM, "max_side": max_side,
                   "seed": seed, "override_sha256": file_hash(override) if override and override.exists() else None}
    previous = json.loads(info_path.read_text()) if info_path.exists() else {}
    cached = (target.exists() and previous.get("inputs") == fingerprint
              and previous.get("mask_sha256") == file_hash(target))
    if cached:
        mask = Image.open(target).convert("L")
    else:
        if override and override.exists():
            mask = Image.open(override).convert("L")
        else:
            cv2.setRNGSeed(seed)
            mask = Image.fromarray(leaf_mask(np.array(im), max_side))
        if mask.size != im.size or set(np.unique(np.array(mask))) - {0, 255}:
            raise ValueError(f"Invalid mask for {p}")
        mask.save(target)
        write_json(info_path, {"inputs": fingerprint, "mask_sha256": file_hash(target)})
    rgba = im.convert("RGBA")
    rgba.putalpha(mask)
    cutout = cache / (key + ".cutout.png")
    if not cutout.exists() or not cached:
        rgba.save(cutout)
    return {"parent_path": str(p), "parent_group_id": row["group_id"],
            "parent_pixel_sha256": pixel_hash(im), "mask_path": str(target.resolve()),
            "mask_sha256": file_hash(target), "cutout_path": str(cutout.resolve()),
            "quality": mask_quality(np.array(mask), label_path(row))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("outputs/coffee_leaf_classifier/split_manifest.json"))
    parser.add_argument("--backgrounds", type=Path, action="append", default=[], help="Repeatable extra background folders, declared training-only negatives")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mask-cache", type=Path, default=Path("outputs/bracol_masks"))
    parser.add_argument("--mask-overrides", type=Path)
    parser.add_argument("--exclude-groups", type=Path, help="JSON list of capture group IDs to omit from compositing after visual review")
    parser.add_argument("--limit", type=int, default=0, help="0 means all training leaves")
    parser.add_argument("--variants-per-leaf", type=int, default=2)
    parser.add_argument("--mask-max-side", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output directory")
    if args.limit < 0 or args.variants_per_leaf < 1 or args.workers < 1 or args.mask_max_side < 128:
        parser.error("Invalid limit, variants, workers or mask size")
    records, _ = audit_leaf_records(load_leaf_manifest(args.manifest))
    # This script consumes original BRACOL leaves, never its own synthetic descendants.
    positives = [r for r in records if r["split"] == "train" and r["source"] == "bracol" and r["label"] == LEAF_CLASSES[1]]
    backgrounds = [r for r in records if r["split"] == "train" and r["label"] == LEAF_CLASSES[0]]
    existing_paths = {r["path"].casefold() for r in records}
    heldout_hashes = {r["pixel_sha256"] for r in records if r["split"] != "train"}
    for folder in args.backgrounds:
        for p in image_paths(folder):
            resolved = str(p.resolve())
            digest = pixel_hash(rgb_image(p))
            if digest in heldout_hashes:
                raise ValueError(f"Extra background duplicates a held-out image: {p}")
            if resolved.casefold() in existing_paths:
                raise ValueError(f"Extra background already appears in the manifest: {p}")
            row = {"path": resolved, "split": "train", "label": LEAF_CLASSES[0],
                   "source": "background_pool", "group_id": "background:" + digest,
                   "pixel_sha256": digest}
            records.append(row); backgrounds.append(row); existing_paths.add(resolved.casefold())
    if not backgrounds or not positives:
        raise ValueError("Need original training leaves and training backgrounds")
    backgrounds.sort(key=lambda r: r["path"])
    args.output.mkdir(parents=True)
    args.mask_cache.mkdir(parents=True, exist_ok=True)
    composites_dir = args.output / "composites"
    composites_dir.mkdir()
    matched_negatives = add_matching_background_negatives(
        records, backgrounds, {tuple(r["size"]) for r in positives}, args.output)
    cv2.setNumThreads(1)
    leaves = select_leaves(positives, args.limit, args.seed)
    excluded_groups = set(json.loads(args.exclude_groups.read_text())) if args.exclude_groups else set()
    if excluded_groups - {r["group_id"] for r in positives}:
        raise ValueError("Exclusion list contains an unknown original training group")
    reports, errors = [], []
    rng = random.Random(args.seed)
    config = {k: str(v) if isinstance(v, Path) else [str(p) for p in v] if k == "backgrounds" else v for k, v in vars(args).items()}
    write_json(args.output / "config.json", config)

    def process(row):
        try:
            result = prepare_mask(row, args.mask_cache, args.mask_overrides, args.mask_max_side, args.seed)
            if row["group_id"] in excluded_groups:
                result["quality"]["issues"].append("excluded_by_visual_review")
                result["quality"]["automatic_checks_passed"] = False
            return result
        except (ValueError, cv2.error) as exc:
            return {"error": str(exc), "parent_path": row["path"]}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for index, (row, report) in enumerate(zip(leaves, pool.map(process, leaves)), 1):
            if "error" in report:
                errors.append(report)
                continue
            reports.append(report)
            if report["quality"]["automatic_checks_passed"]:
                leaf, mask = rgb_image(row["path"]), Image.open(report["mask_path"]).convert("L")
                for variant in range(args.variants_per_leaf):
                    # Alternate existing scene backgrounds and extra background-only assets.
                    source = "background_pool" if variant % 2 else "existing"
                    choices = [b for b in backgrounds if (b["source"] == "background_pool") == (source == "background_pool")]
                    bg = rng.choice(choices or backgrounds)
                    result = composite(leaf, mask, rgb_image(bg["path"]))
                    path = composites_dir / f"{Path(row['path']).stem}_v{variant+1:02d}.png"
                    result.save(path)
                    report.setdefault("preview_paths", []).append(str(path.resolve()))
                    records.append({"path": str(path.resolve()), "label": LEAF_CLASSES[1], "split": "train",
                                    "source": "bracol_composite", "group_id": row["group_id"],
                                    "parent_path": row["path"], "parent_group_id": row["group_id"],
                                    "parent_pixel_sha256": report["parent_pixel_sha256"],
                                    "background_path": bg["path"], "background_group_id": bg["group_id"],
                                    "background_pixel_sha256": bg["pixel_sha256"],
                                    "mask_path": report["mask_path"], "mask_sha256": report["mask_sha256"],
                                    "augmentation_seed": args.seed, "variant": variant+1,
                                    "foreground_pixels_verified_unchanged": True})
            if index % 20 == 0 or index == len(leaves):
                print(f"Processed {index}/{len(leaves)} leaves; composites={sum(r['source']=='bracol_composite' for r in records)}", flush=True)
                write_json(args.output / "mask_report.json", {"masks": reports, "errors": errors})
    contact_sheets(reports, args.output / "review", "masks")
    flagged = [r for r in reports if not r["quality"]["automatic_checks_passed"]]
    if flagged:
        contact_sheets(flagged, args.output / "review", "excluded")
    audited, audit = audit_leaf_records(records)
    write_json(args.output / "split_manifest.json", {"classes": LEAF_CLASSES, "records": audited})
    write_json(args.output / "dataset_report.json", audit)
    summary = {"selected_leaves": len(leaves), "usable_masks": len(reports)-len(flagged),
               "excluded_masks": len(flagged), "mask_errors": len(errors),
               "composites": sum(r["source"] == "bracol_composite" for r in records),
               "backgrounds": len(backgrounds), "foreground_pixels_verified_unchanged": True,
               "matching_background_negatives": matched_negatives,
               "heldout_partitions_unchanged": True,
               "review": "Automatic checks are screening only; inspect contact sheets before training.",
               "excluded_groups": [r["parent_group_id"] for r in flagged]}
    write_json(args.output / "mask_report.json", {"masks": reports, "errors": errors})
    write_json(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
