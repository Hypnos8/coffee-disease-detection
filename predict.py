"""Predict annotated coffee-leaf conditions from a trusted trained checkpoint."""
import argparse
import json
from pathlib import Path
import torch
from model_io import PhotoClassifier
from leaf_classifier import CONDITION_TASK


def predict(checkpoint, image, device="cpu"):
    classifier = PhotoClassifier(checkpoint, CONDITION_TASK, device)
    return condition_result(classifier, image)


def condition_result(classifier, image):
    saved = classifier.metadata
    scores = classifier.scores(image).tolist()
    detected = [c for c, score in zip(saved["classes"], scores) if score >= saved["threshold"]]
    return {"task": CONDITION_TASK, "image": str(image), "scores": dict(zip(saved["classes"], scores)),
            "threshold": saved["threshold"], "conditions": detected,
            "result": ", ".join(detected) if detected else "No target condition detected",
            "scope": "Isolated coffee leaf; scores are not calibrated probabilities or proof of health."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("outputs/coffee_condition_classifier/coffee_condition_classifier.pt"))
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(4)
    print(json.dumps(predict(args.checkpoint, args.image, args.device), indent=2))


if __name__ == "__main__":
    main()
