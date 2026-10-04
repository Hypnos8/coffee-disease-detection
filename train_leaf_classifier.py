"""Train the coffee-leaf stage with negatives and independent phone evaluation."""
import argparse
import csv
from collections import Counter
from importlib.metadata import version
import random
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from coffee import PREPROCESS, LeafDataset, make_model, write_json
from leaf_classifier import (LEAF_TASK, LEAF_CLASSES, LEAF_THRESHOLDS, audit_leaf_records,
                             build_leaf_records, leaf_metrics, leaf_status, load_leaf_manifest)
from train import run_epoch


def class_weights(targets):
    counts = targets.sum(0)
    if (counts <= 0).any():
        raise ValueError("Training needs both classes")
    return len(targets) / (2 * counts)


def background_sampling_weights(rows):
    """Equal class probability; negative backgrounds reflect composite usage."""
    positive_count = sum(r["label"] == LEAF_CLASSES[1] for r in rows)
    negatives = [r for r in rows if r["label"] == LEAF_CLASSES[0]]
    if not positive_count or not negatives:
        raise ValueError("Sampling needs both classes")
    usage = Counter(r["background_group_id"] for r in rows if r["source"] == "bracol_composite")
    copies = Counter(r["group_id"] for r in negatives)
    negative_weights = {group: max(1, usage[group])/count for group, count in copies.items()}
    normalizer = sum(negative_weights[r["group_id"]] for r in negatives)
    return torch.tensor([.5/positive_count if r["label"] == LEAF_CLASSES[1]
                         else .5*negative_weights[r["group_id"]]/normalizer for r in rows], dtype=torch.double)


def evaluate_partition(out, split, rows, targets, scores):
    result = leaf_metrics(targets, scores)
    write_json(out / f"{split}_metrics.json", result)
    with (out / f"{split}_predictions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "source", "actual", *LEAF_CLASSES, "predicted", "leaf_status"])
        for row, y, score in zip(rows, targets, scores):
            writer.writerow([row["path"], row["source"], LEAF_CLASSES[int(y.argmax())],
                             *score.tolist(), LEAF_CLASSES[int(score.argmax())], leaf_status(float(score[1]))])
    matrix = np.array(result["confusion_matrix"])
    with (out / f"{split}_confusion.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows([["actual / predicted", *LEAF_CLASSES],
                         [LEAF_CLASSES[0], *matrix[0]], [LEAF_CLASSES[1], *matrix[1]]])
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(matrix, cmap="Blues")
    for (i, j), n in np.ndenumerate(matrix):
        ax.text(j, i, str(n), ha="center", va="center", color="white" if n > matrix.max()/2 else "black")
    ax.set(title=f"Coffee-leaf classifier: {split}", xlabel="Predicted", ylabel="Actual",
           xticks=[0, 1], yticks=[0, 1], xticklabels=["Not coffee leaf", "Coffee leaf"],
           yticklabels=["Not coffee leaf", "Coffee leaf"])
    fig.tight_layout()
    fig.savefig(out / f"{split}_confusion.png", dpi=160)
    plt.close(fig)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positive-data", type=Path, default=Path("BRACOL_REVIEWED"))
    parser.add_argument("--negative-data", type=Path, default=Path("not_coffee_leaves"))
    parser.add_argument("--manifest", type=Path, help="Explicit labeled sources and grouped partitions")
    parser.add_argument("--output", type=Path, default=Path("outputs/coffee_leaf_classifier"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--balance-backgrounds", action="store_true",
                        help="Equal class sampling with negative background exposure matched to composite usage; unweighted CE")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output directory")
    if args.epochs <= args.warmup_epochs or args.warmup_epochs < 0 or args.batch_size < 2 or args.patience < 1:
        parser.error("epochs > warmup >= 0, batch-size >= 2, and patience >= 1 required")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    records = load_leaf_manifest(args.manifest) if args.manifest else build_leaf_records(
        args.positive_data, args.negative_data, args.seed)
    records, audit = audit_leaf_records(records)
    args.output.mkdir(parents=True)
    write_json(args.output / "split_manifest.json", {"classes": LEAF_CLASSES, "records": records})
    write_json(args.output / "dataset_report.json", audit)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(task=LEAF_TASK, device=str(device), warmup_lr=0.001, finetune_lr=0.0001,
                  weight_decay=0.0001, preprocessing=PREPROCESS, thresholds=LEAF_THRESHOLDS)
    packages = ("torch", "torchvision", "numpy", "Pillow", "PyYAML", "scikit-learn", "matplotlib")
    (args.output / "environment.txt").write_text("\n".join(f"{p}=={version(p)}" for p in packages)+"\n")
    rows = {s: [r for r in records if r["split"] == s] for s in ("train", "valid", "test", "phone")}
    loaders = {}
    for split in ("train", "valid"):
        dataset = LeafDataset(Path("."), rows[split], training=split == "train")
        sampler = None
        if split == "train" and args.balance_backgrounds:
            sampler = WeightedRandomSampler(background_sampling_weights(rows[split]), len(dataset), replacement=True,
                                            generator=torch.Generator().manual_seed(args.seed))
        loaders[split] = DataLoader(dataset, batch_size=args.batch_size, shuffle=split == "train" and sampler is None,
                                    sampler=sampler)
    weights = class_weights(loaders["train"].dataset.targets)
    if args.balance_backgrounds:
        weights = torch.ones(2)
    config["class_weights"] = weights.tolist()
    config["sampling"] = "equal_classes_background_usage" if args.balance_backgrounds else "shuffled_class_weighted_loss"
    write_json(args.output / "config.json", config)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights.to(device))
    torch.hub.set_dir(str(Path(".cache/torch").resolve()))
    model = make_model(2, pretrained=True).to(device)
    for p in model.features.parameters():
        p.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=0.001, weight_decay=0.0001)
    best, stale, history = -1, 0, []
    checkpoint = args.output / f"{LEAF_TASK}.pt"
    started = time.perf_counter()
    for epoch in range(1, args.epochs+1):
        frozen = epoch <= args.warmup_epochs
        if epoch == args.warmup_epochs+1:
            for p in model.parameters():
                p.requires_grad_(True)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.0001)
        tl, yt, pt = run_epoch(model, loaders["train"], loss_fn, device, optimizer, frozen, "binary")
        vl, yv, pv = run_epoch(model, loaders["valid"], loss_fn, device, classification="binary")
        tm, vm = leaf_metrics(yt, pt), leaf_metrics(yv, pv)
        entry = {"epoch": epoch, "train_loss": tl, "valid_loss": vl,
                 "train_f1_macro": tm["f1_macro"], "valid_f1_macro": vm["f1_macro"],
                 "train_accuracy": tm["accuracy"], "valid_accuracy": vm["accuracy"]}
        history.append(entry)
        write_json(args.output / "history.json", history)
        with (args.output / "history.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(entry)); writer.writeheader(); writer.writerows(history)
        if vm["f1_macro"] > best:
            best, stale = vm["f1_macro"], 0
            torch.save({"task": LEAF_TASK, "activation": "softmax", "state_dict": model.state_dict(),
                        "classes": LEAF_CLASSES, "architecture": "mobilenet_v3_small", "width_mult": 1.0,
                        "preprocessing": PREPROCESS, "threshold": 0.5, "leaf_thresholds": LEAF_THRESHOLDS,
                        "epoch": epoch, "validation_macro_f1": best, "config": config}, checkpoint)
        elif not frozen:
            stale += 1
        print(f"Coffee-leaf epoch {epoch:02d}/{args.epochs}: loss={tl:.4f}, "
              f"val_macro_F1={vm['f1_macro']:.4f}, val_accuracy={vm['accuracy']:.4f}", flush=True)
        if not frozen and stale >= args.patience:
            print("Early stopping", flush=True)
            break
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(saved["state_dict"])
    reloaded = make_model(2).to(device)
    reloaded.load_state_dict(saved["state_dict"])
    model.eval(); reloaded.eval()
    x, _ = next(iter(loaders["valid"]))
    with torch.inference_mode():
        torch.testing.assert_close(model(x.to(device)), reloaded(x.to(device)), atol=0, rtol=0)
    del reloaded
    results = {}
    for split in ("valid", "test", "phone"):
        if split not in loaders:
            loaders[split] = DataLoader(LeafDataset(Path("."), rows[split]), batch_size=args.batch_size)
        _, y, scores = run_epoch(model, loaders[split], loss_fn, device, classification="binary")
        results[split] = evaluate_partition(args.output, split, rows[split], y, scores)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, metric in zip(axes, ("loss", "f1_macro", "accuracy")):
        for split in ("train", "valid"):
            ax.plot([h["epoch"] for h in history], [h[f"{split}_{metric}"] for h in history], label=split)
        ax.set(xlabel="Epoch", title=metric); ax.legend()
    fig.tight_layout(); fig.savefig(args.output / "training_curves.png", dpi=160); plt.close(fig)
    summary = {"task": LEAF_TASK, "best_epoch": saved["epoch"], "epochs_completed": len(history),
               "parameter_count": sum(p.numel() for p in model.parameters()),
               "checkpoint_bytes": checkpoint.stat().st_size, "checkpoint_reload_verified": True,
               "elapsed_training_and_evaluation_seconds": time.perf_counter()-started, "metrics": results}
    write_json(args.output / "summary.json", summary)
    print(f"Saved {checkpoint}; test accuracy={results['test']['accuracy']:.4f}; "
          f"phone counts={results['phone']['operating_counts']}", flush=True)


if __name__ == "__main__":
    main()
