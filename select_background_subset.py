"""Add a reproducible small subset of composites to an unchanged leaf baseline."""
import argparse
from collections import Counter, defaultdict
import math
from pathlib import Path
import random

from coffee import write_json
from leaf_classifier import LEAF_CLASSES, load_leaf_manifest, validate_composite_sources


def select_subset(baseline, augmented, fraction=0.2, seed=42):
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    by_path = {str(Path(r["path"]).resolve()).casefold(): r for r in baseline}
    positives = [r for r in baseline if r["split"] == "train" and r["label"] == LEAF_CLASSES[1]]
    candidates = defaultdict(list)
    for row in augmented:
        if row["source"] != "bracol_composite" or row["split"] != "train":
            continue
        parent_key = str(Path(row["parent_path"]).resolve()).casefold()
        background_key = str(Path(row["background_path"]).resolve()).casefold()
        parent, background = by_path.get(parent_key), by_path.get(background_key)
        if (parent and parent["split"] == "train" and parent["label"] == LEAF_CLASSES[1]
                and background and background["split"] == "train"
                and background["label"] == LEAF_CLASSES[0]):
            candidates[parent_key].append(row)
    count = math.floor(len(positives) * fraction + 0.5)
    if count < 1 or count > len(candidates):
        raise ValueError(f"Requested {count} parents, but only {len(candidates)} eligible parents available")
    rng = random.Random(seed)
    selected_keys = rng.sample(sorted(candidates), count)
    selected = [dict(rng.choice(sorted(candidates[key], key=lambda r: r["path"]))) for key in selected_keys]
    records = [dict(r) for r in baseline] + selected
    validate_composite_sources(records)
    report = {
        "selection_seed": seed, "requested_fraction_of_original_training_leaves": fraction,
        "original_training_leaves": len(positives), "eligible_parents": len(candidates),
        "selected_parents": count, "added_composites": len(selected),
        "fraction_of_original_training_leaves": count / len(positives),
        "fraction_of_training_positives_that_are_composites": count / (len(positives) + count),
        "original_records_preserved": records[:len(baseline)] == baseline,
        "one_variant_per_parent": len({r["parent_path"] for r in selected}) == count,
        "background_policy": "Only original baseline training negatives; no new background negatives or crops",
        "selection_policy": "Uniform random sample of eligible training leaves; one eligible variant per leaf",
        "background_sources": dict(Counter(by_path[str(Path(r["background_path"]).resolve()).casefold()]["source"]
                                           for r in selected)),
        "selected_parent_groups": [r["parent_group_id"] for r in selected],
        "splits": dict(Counter(r["split"] for r in records)),
        "training_sources": dict(Counter(r["source"] for r in records if r["split"] == "train")),
    }
    return records, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=Path("outputs/coffee_leaf_classifier/split_manifest.json"))
    parser.add_argument("--augmented", type=Path, default=Path("outputs/background_augmented/split_manifest.json"))
    parser.add_argument("--fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("outputs/background_subset_20pct"))
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output directory")
    records, report = select_subset(load_leaf_manifest(args.baseline), load_leaf_manifest(args.augmented),
                                    args.fraction, args.seed)
    args.output.mkdir(parents=True)
    report.update(baseline_manifest=str(args.baseline.resolve()), augmented_manifest=str(args.augmented.resolve()))
    write_json(args.output / "split_manifest.json", {"classes": LEAF_CLASSES, "records": records})
    write_json(args.output / "selection_report.json", report)
    print(f"Selected {report['added_composites']} variants; partitions: {report['splits']}")


if __name__ == "__main__":
    main()
