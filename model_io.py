"""Task-aware checkpoint loading shared by Python prediction and Android export."""
import numpy as np
from PIL import Image
import torch

from coffee import PREPROCESS, fit_image, make_model, tensor_transform
from leaf_classifier import CONDITION_TASK, LEAF_TASK, LEAF_CLASSES, LEAF_THRESHOLDS

CONDITION_CLASSES = ["Cercospora", "Miner", "Phoma", "Rust"]


def load_classifier(path, expected_task=None):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved.get("architecture") != "mobilenet_v3_small" or saved.get("width_mult") != 1.0:
        raise ValueError("Expected width-1.0 MobileNetV3-Small checkpoint")
    classes = saved.get("classes")
    # Historical checkpoints without task metadata are supported only for the known condition task.
    task = saved.get("task")
    if task is None and classes == CONDITION_CLASSES:
        task = CONDITION_TASK
    if task not in (CONDITION_TASK, LEAF_TASK) or (expected_task and task != expected_task):
        raise ValueError(f"Checkpoint task {task!r} does not match expected task {expected_task!r}")
    expected_classes = CONDITION_CLASSES if task == CONDITION_TASK else LEAF_CLASSES
    activation = "sigmoid" if task == CONDITION_TASK else "softmax"
    if classes != expected_classes or saved.get("activation", activation) != activation:
        raise ValueError(f"Invalid class order or activation for {task}")
    if saved.get("preprocessing") != PREPROCESS:
        raise ValueError("Unsupported checkpoint preprocessing")
    threshold = saved.get("threshold")
    if not isinstance(threshold, (int, float)) or not 0 < threshold < 1:
        raise ValueError("Checkpoint threshold must be between zero and one")
    saved = dict(saved, task=task, activation=activation)
    if task == LEAF_TASK:
        thresholds = saved.get("leaf_thresholds", LEAF_THRESHOLDS)
        if not 0 <= thresholds["reject"] < thresholds["accept"] <= 1:
            raise ValueError("Invalid leaf operating thresholds")
        saved["leaf_thresholds"] = thresholds
    model = make_model(len(classes))
    model.load_state_dict(saved["state_dict"], strict=True)
    return model.eval(), saved


class PhotoClassifier:
    """Load once; classify many images with the checkpoint's own preprocessing."""
    def __init__(self, checkpoint, task, device="cpu"):
        self.model, self.metadata = load_classifier(checkpoint, task)
        self.device = device
        self.model.to(device)

    def scores(self, image):
        config = self.metadata["preprocessing"]
        with Image.open(image) as im:
            x = tensor_transform(config=config)(fit_image(im, config)).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            logits = self.model(x)
            output = logits.softmax(1) if self.metadata["activation"] == "softmax" else logits.sigmoid()
        scores = output[0].cpu().numpy()
        if not np.isfinite(scores).all():
            raise RuntimeError("Non-finite classifier output")
        return scores
