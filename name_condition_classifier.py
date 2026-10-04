"""Publish a task-named copy of the historical condition classifier without retraining."""
import argparse
import hashlib
from pathlib import Path
import torch

from coffee import write_json
from leaf_classifier import CONDITION_TASK
from model_io import load_classifier


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/bracol_small/best_model.pt"))
    parser.add_argument("--output", type=Path, default=Path("outputs/coffee_condition_classifier"))
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output directory must be new")
    original, saved = load_classifier(args.checkpoint, CONDITION_TASK)
    args.output.mkdir(parents=True)
    path = args.output / f"{CONDITION_TASK}.pt"
    saved["provenance"] = {"source_checkpoint": str(args.checkpoint),
                           "source_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                           "weights_retrained": False}
    torch.save(saved, path)
    renamed, verified = load_classifier(path, CONDITION_TASK)
    for key, value in original.state_dict().items():
        torch.testing.assert_close(value, renamed.state_dict()[key], atol=0, rtol=0)
    torch.set_num_threads(4)
    x = torch.randn(2, 3, 224, 224, generator=torch.Generator().manual_seed(42))
    with torch.inference_mode():
        torch.testing.assert_close(original(x), renamed(x), atol=0, rtol=0)
    write_json(args.output / "migration_report.json", dict(saved["provenance"],
               task=CONDITION_TASK, state_dict_identical=True, predictions_identical=True,
               checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    print(f"Published {path}; weights and predictions verified identical")


if __name__ == "__main__":
    main()
