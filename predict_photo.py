"""Check for a coffee leaf, then classify supported conditions if accepted."""
import argparse
import json
from pathlib import Path
import torch

from leaf_classifier import CONDITION_TASK, LEAF_TASK, leaf_status
from model_io import PhotoClassifier
from predict import condition_result


class PhotoPipeline:
    """Replace leaf_stage with a detector/quality stage exposing scores and metadata."""
    def __init__(self, leaf_stage, condition_stage):
        self.leaf_stage = leaf_stage
        self.condition_stage = condition_stage

    def predict(self, image):
        scores = self.leaf_stage.scores(image)
        metadata = self.leaf_stage.metadata
        score = float(scores[metadata["classes"].index("coffee_leaf")])
        thresholds = metadata["leaf_thresholds"]
        status = leaf_status(score, thresholds)
        result = {"image": str(image), "leaf": {"task": LEAF_TASK, "score": score,
                   "status": status, "thresholds": thresholds},
                  "condition_inference_ran": False, "conditions": None, "condition_scores": None}
        if status == "not_coffee_leaf":
            result["result"] = "No suitable coffee leaf detected"
        elif status == "uncertain":
            result["result"] = "Please retake the photo"
        else:
            condition = condition_result(self.condition_stage, image)
            result.update(condition_inference_ran=True, conditions=condition["conditions"],
                          condition_scores=condition["scores"], result=condition["result"])
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--leaf-checkpoint", type=Path,
                        default=Path("outputs/coffee_leaf_classifier/coffee_leaf_classifier.pt"))
    parser.add_argument("--condition-checkpoint", type=Path,
                        default=Path("outputs/coffee_condition_classifier/coffee_condition_classifier.pt"))
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--reject-threshold", type=float)
    parser.add_argument("--accept-threshold", type=float)
    args = parser.parse_args()
    torch.set_num_threads(4)
    leaf = PhotoClassifier(args.leaf_checkpoint, LEAF_TASK, args.device)
    condition = PhotoClassifier(args.condition_checkpoint, CONDITION_TASK, args.device)
    thresholds = dict(leaf.metadata["leaf_thresholds"])
    if args.reject_threshold is not None:
        thresholds["reject"] = args.reject_threshold
    if args.accept_threshold is not None:
        thresholds["accept"] = args.accept_threshold
    leaf_status(0.5, thresholds)  # Validate overrides before inference.
    leaf.metadata["leaf_thresholds"] = thresholds
    print(json.dumps(PhotoPipeline(leaf, condition).predict(args.image), indent=2))


if __name__ == "__main__":
    main()
