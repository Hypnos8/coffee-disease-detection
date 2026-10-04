"""Train a four-output MobileNetV3-Small and save reproducible evaluation artifacts."""
import argparse
import csv
from importlib.metadata import version
import json
import random
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from coffee import (PREPROCESS, LeafDataset, audit_dataset, calculate_metrics,
                    make_model, write_json)


def run_epoch(model, loader, loss_fn, device, optimizer=None, frozen=False, classification="multilabel"):
    model.train(optimizer is not None)
    if frozen:
        model.features.eval()  # Freeze batch-norm statistics along with backbone weights.
    loss_total, targets, scores = 0.0, [], []
    with torch.set_grad_enabled(optimizer is not None):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if optimizer:
                optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y.argmax(1)) if classification == "binary" else loss_fn(logits, y)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss")
            if optimizer:
                loss.backward()
                optimizer.step()
            loss_total += loss.item() * len(x)
            targets.append(y.detach().cpu().numpy())
            probabilities = logits.detach().softmax(1) if classification == "binary" else logits.detach().sigmoid()
            scores.append(probabilities.cpu().numpy())
    return loss_total / len(loader.dataset), np.concatenate(targets), np.concatenate(scores)


def save_evaluation(out, split, rows, targets, scores, classes):
    metrics = calculate_metrics(targets, scores, classes)
    write_json(out / f"{split}_metrics.json", metrics)
    with (out / f"{split}_predictions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["path"] + [f"true_{c}" for c in classes] +
                        [f"score_{c}" for c in classes] + [f"pred_{c}" for c in classes])
        for row, y, score in zip(rows, targets, scores):
            writer.writerow([row["path"]] + y.astype(int).tolist() + score.tolist() +
                            (score >= 0.5).astype(int).tolist())
    fig, axes = plt.subplots(2, 2, figsize=(9, 8))
    for ax, name in zip(axes.flat, classes):
        matrix = np.array(metrics["per_class"][name]["confusion_matrix"])
        with (out / f"{split}_confusion_{name}.csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows([["actual / predicted", "absent", "present"],
                             ["absent", *matrix[0]], ["present", *matrix[1]]])
        ax.imshow(matrix, cmap="Blues")
        for (i, j), value in np.ndenumerate(matrix):
            ax.text(j, i, str(value), ha="center", va="center",
                    color="white" if value > matrix.max()/2 else "black")
        ax.set(title=name, xlabel="Predicted", ylabel="Actual", xticks=[0, 1], yticks=[0, 1],
               xticklabels=["Absent", "Present"], yticklabels=["Absent", "Present"])
    fig.suptitle(f"{split.title()} confusion matrices — threshold 0.5")
    fig.tight_layout()
    fig.savefig(out / f"{split}_confusion_matrices.png", dpi=160)
    plt.close(fig)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("BRACOL_REVIEWED"))
    parser.add_argument("--output", type=Path, default=Path("outputs/coffee_condition_classifier"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    if args.epochs <= args.warmup_epochs or args.warmup_epochs < 0:
        parser.error("epochs must exceed nonnegative warmup-epochs")
    if args.batch_size < 2 or args.patience < 1:
        parser.error("batch-size must be >=2 and patience >=1")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("output directory must be new or empty; choose another --output")
    args.output.mkdir(parents=True, exist_ok=True)
    packages = ("torch", "torchvision", "numpy", "Pillow", "PyYAML", "scikit-learn", "matplotlib")
    (args.output / "environment.txt").write_text(
        "\n".join(f"{name}=={version(name)}" for name in packages) + "\n", encoding="utf-8")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu")
                          if args.device == "auto" else args.device)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update({"device": str(device), "architecture": "mobilenet_v3_small", "width_mult": 1.0,
                   "pretrained_weights": "IMAGENET1K_V1", "torch_version": str(torch.__version__),
                   "warmup_lr": 0.001, "finetune_lr": 0.0001, "weight_decay": 0.0001,
                   "preprocessing": PREPROCESS, "threshold": 0.5})
    write_json(args.output / "config.json", config)
    classes, records, report = audit_dataset(args.data)
    if len(classes) != 4:
        raise ValueError("This workflow expects four condition classes")
    write_json(args.output / "dataset_report.json", report)
    write_json(args.output / "split_manifest.json", records)
    print(json.dumps(report["splits"], indent=2), flush=True)
    if args.audit_only:
        return
    # Cache downloaded weights inside the workspace, not the user's global torch cache.
    torch.hub.set_dir(str(Path(".cache/torch").resolve()))
    model = make_model(len(classes), pretrained=True).to(device)
    rows = {s: [r for r in records if r["split"] == s] for s in ("train", "valid", "test")}
    loaders = {}
    for split in ("train", "valid"):
        dataset = LeafDataset(args.data, rows[split], training=split == "train")
        loaders[split] = DataLoader(dataset, batch_size=args.batch_size, shuffle=split == "train",
                                    num_workers=0, drop_last=False)
        print(f"Cached {split} resized images", flush=True)
    positives = loaders["train"].dataset.targets.sum(0)
    positive_weights = (len(rows["train"]) - positives) / positives
    config["positive_weights"] = positive_weights.tolist()
    write_json(args.output / "config.json", config)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=positive_weights.to(device))
    for p in model.features.parameters():
        p.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=0.001, weight_decay=0.0001)
    best, best_epoch, stale, history = -1.0, 0, 0, []
    started = time.perf_counter()
    checkpoint = args.output / "coffee_condition_classifier.pt"
    for epoch in range(1, args.epochs + 1):
        frozen = epoch <= args.warmup_epochs
        if epoch == args.warmup_epochs + 1:
            for p in model.parameters():
                p.requires_grad_(True)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.0001)
        epoch_start = time.perf_counter()
        train_loss, yt, pt = run_epoch(model, loaders["train"], loss_fn, device, optimizer, frozen)
        val_loss, yv, pv = run_epoch(model, loaders["valid"], loss_fn, device)
        tm, vm = calculate_metrics(yt, pt, classes), calculate_metrics(yv, pv, classes)
        entry = {"epoch": epoch, "phase": "warmup" if frozen else "finetune",
                 "train_loss": train_loss, "valid_loss": val_loss,
                 "train_f1_macro": tm["f1_macro"], "valid_f1_macro": vm["f1_macro"],
                 "train_exact_match_accuracy": tm["exact_match_accuracy"],
                 "valid_exact_match_accuracy": vm["exact_match_accuracy"],
                 "seconds": time.perf_counter()-epoch_start}
        history.append(entry)
        write_json(args.output / "history.json", history)
        with (args.output / "history.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(entry))
            writer.writeheader()
            writer.writerows(history)
        if vm["f1_macro"] > best:
            best, best_epoch, stale = vm["f1_macro"], epoch, 0
            torch.save({"task": "coffee_condition_classifier", "activation": "sigmoid",
                        "state_dict": model.state_dict(), "classes": classes,
                        "architecture": "mobilenet_v3_small", "width_mult": 1.0,
                        "preprocessing": PREPROCESS, "threshold": 0.5,
                        "epoch": epoch, "validation_macro_f1": best, "config": config}, checkpoint)
        elif not frozen:
            stale += 1
        print(f"Epoch {epoch:02d}/{args.epochs} {entry['phase']}: loss={train_loss:.4f} "
              f"val_loss={val_loss:.4f} val_macro_F1={vm['f1_macro']:.4f} "
              f"val_exact_acc={vm['exact_match_accuracy']:.4f} ({entry['seconds']:.1f}s)", flush=True)
        if not frozen and stale >= args.patience:
            print("Early stopping", flush=True)
            break
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(saved["state_dict"])
    # Verify reload against a separately instantiated model before final evaluation.
    reloaded = make_model(len(classes)).to(device)
    reloaded.load_state_dict(saved["state_dict"])
    model.eval()
    reloaded.eval()
    x, _ = next(iter(loaders["valid"]))
    with torch.no_grad():
        torch.testing.assert_close(model(x.to(device)), reloaded(x.to(device)), rtol=0, atol=0)
    del reloaded
    metrics = {}
    loaders["test"] = DataLoader(LeafDataset(args.data, rows["test"]), batch_size=args.batch_size)
    for split in ("valid", "test"):
        loss, targets, scores = run_epoch(model, loaders[split], loss_fn, device)
        metrics[split] = save_evaluation(args.output, split, rows[split], targets, scores, classes)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (suffix, title) in zip(axes, [("loss", "Loss"), ("f1_macro", "Macro F1"),
                                         ("exact_match_accuracy", "Exact-match accuracy")]):
        for split in ("train", "valid"):
            ax.plot([h["epoch"] for h in history], [h[f"{split}_{suffix}"] for h in history], label=split)
        ax.set(title=title, xlabel="Epoch")
        ax.legend()
    fig.tight_layout()
    fig.savefig(args.output / "training_curves.png", dpi=160)
    plt.close(fig)
    summary = {"best_epoch": best_epoch, "epochs_completed": len(history),
               "elapsed_training_and_evaluation_seconds": time.perf_counter()-started,
               "parameter_count": sum(p.numel() for p in model.parameters()),
               "checkpoint_bytes": checkpoint.stat().st_size,
               "checkpoint_reload_verified": True, "metrics": metrics}
    write_json(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
