"""Export MobileNetV3-Small to ONNX for ONNX Runtime Android CPU inference."""
import argparse
import csv
import hashlib
from importlib.metadata import version
import inspect
import json
from pathlib import Path
import random

import numpy as np
from PIL import Image
import torch
from torch import nn

from coffee import (PREPROCESS, audit_dataset, calculate_metrics, fit_image,
                    make_model, tensor_transform, write_json)
from leaf_classifier import (CONDITION_TASK, LEAF_TASK, LEAF_CLASSES, LEAF_THRESHOLDS,
                             audit_leaf_records, leaf_metrics, load_leaf_manifest)
from model_io import load_classifier


class AndroidModel(nn.Module):
    """Export scores with the activation belonging to the checkpoint's task."""
    def __init__(self, model, activation="sigmoid"):
        super().__init__()
        self.model = model
        self.activation = activation

    def forward(self, image):
        logits = self.model(image)
        return logits.softmax(1) if self.activation == "softmax" else logits.sigmoid()


def load_checkpoint(path, expected_task=None):
    model, saved = load_classifier(path, expected_task)
    return AndroidModel(model, saved["activation"]).eval(), saved


def image_tensor(path, preprocessing=PREPROCESS):
    with Image.open(path) as im:
        return tensor_transform(config=preprocessing)(fit_image(im, preprocessing)).unsqueeze(0)


def select_calibration(records, limit, seed):
    """Only training images may determine quantization ranges."""
    rows = sorted((r for r in records if r["split"] == "train"), key=lambda r: r["path"])
    if not rows or limit <= 0:
        raise ValueError("Calibration requires training images and a positive sample limit")
    random.Random(seed).shuffle(rows)
    return rows[:limit]


def export_fp32(model, destination):
    import onnx
    sample = torch.zeros(1, 3, 224, 224)
    kwargs = {}
    # Pin the tested legacy exporter even when newer torch versions default to dynamo.
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        kwargs["dynamo"] = False
    torch.onnx.export(model, (sample,), str(destination), input_names=["image"],
                      output_names=["scores"], opset_version=17,
                      do_constant_folding=True, **kwargs)
    onnx.checker.check_model(onnx.load(str(destination)), full_check=True)


def make_session(path):
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])


def quantize_model(source, destination, calibration, root):
    import onnx
    from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod, QuantFormat,
                                          QuantType, quantize_static)
    from onnxruntime.quantization.shape_inference import quant_pre_process

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.rows = iter(calibration)

        def get_next(self):
            row = next(self.rows, None)
            if row is None:
                return None
            return {"image": image_tensor(root / row["path"]).numpy()}

    # Preprocess shapes separately; do not fuse the public FP32 reference during quantization.
    intermediate = destination.parent / "quantization_preprocessed.onnx"
    quant_pre_process(str(source), str(intermediate), skip_optimization=True)
    quantize_static(str(intermediate), str(destination), Reader(),
                    quant_format=QuantFormat.QDQ, activation_type=QuantType.QInt8,
                    weight_type=QuantType.QInt8, per_channel=True,
                    calibrate_method=CalibrationMethod.MinMax,
                    op_types_to_quantize=["Conv", "Gemm", "MatMul"],
                    extra_options={"ActivationSymmetric": True, "WeightSymmetric": True})
    onnx.checker.check_model(onnx.load(str(destination)), full_check=True)


def quantization_quality(reference, candidate, max_macro_drop, max_class_drop):
    macro_drop = reference["f1_macro"] - candidate["f1_macro"]
    drops = {c: reference["per_class"][c]["f1"] - candidate["per_class"][c]["f1"]
             for c in reference["per_class"]}
    return {"macro_f1_drop": macro_drop, "per_class_f1_drop": drops,
            "max_allowed_macro_f1_drop": max_macro_drop,
            "max_allowed_per_class_f1_drop": max_class_drop,
            "passed": macro_drop <= max_macro_drop and all(d <= max_class_drop for d in drops.values())}


def verify_models(model, sessions, records, root, classes, threshold, output,
                  task=CONDITION_TASK, leaf_thresholds=LEAF_THRESHOLDS):
    validation = [r for r in records if r["split"] == "valid"]
    if not validation:
        raise ValueError("No validation images available for export verification")
    values = {"pytorch": []} | {name: [] for name in sessions}
    with torch.inference_mode():
        for i, row in enumerate(validation):
            tensor = image_tensor(root / row["path"])
            values["pytorch"].append(model(tensor).numpy()[0])
            for name, session in sessions.items():
                scores = session.run(["scores"], {"image": tensor.numpy()})[0]
                if scores.shape != (1, len(classes)) or not np.isfinite(scores).all():
                    raise ValueError(f"Invalid output from {name}")
                values[name].append(scores[0])
            if i == 0:
                # Little-endian float fixture usable directly by an Android ByteBuffer.
                tensor.numpy().astype("<f4").tofile(output / f"{task}.verification_input.f32")
                write_json(output / f"{task}.verification_expected.json", {
                    "source_image": row["path"], "shape": [1, 3, 224, 224],
                    "dtype": "float32", "byte_order": "little_endian",
                    "scores": {name: rows[0].tolist() for name, rows in values.items()}})
            if (i + 1) % 100 == 0:
                print(f"Verified {i + 1}/{len(validation)} validation images", flush=True)
    values = {k: np.asarray(v) for k, v in values.items()}
    np.testing.assert_allclose(values["fp32"], values["pytorch"], rtol=1e-4, atol=1e-4,
                               err_msg="ONNX FP32 export differs from PyTorch")
    targets = [r["target"] for r in validation]
    report = {"split": "valid", "images": len(validation), "threshold": threshold,
              "fp32_tolerance": {"rtol": 1e-4, "atol": 1e-4}, "models": {}}
    for name, scores in values.items():
        report["models"][name] = {
            "metrics": leaf_metrics(targets, scores, leaf_thresholds) if task == LEAF_TASK
                       else calculate_metrics(targets, scores, classes, threshold),
            "max_absolute_score_difference_from_pytorch": float(abs(scores-values["pytorch"]).max()),
            "label_decisions_differing_from_pytorch": int(
                ((scores >= threshold) != (values["pytorch"] >= threshold)).sum())}
        if task == LEAF_TASK:
            report["models"][name]["operating_decisions_differing_from_pytorch"] = int(
                ((scores[:, 1] >= leaf_thresholds["accept"]) !=
                 (values["pytorch"][:, 1] >= leaf_thresholds["accept"])).sum() +
                ((scores[:, 1] <= leaf_thresholds["reject"]) !=
                 (values["pytorch"][:, 1] <= leaf_thresholds["reject"])).sum())
    with (output / "validation_predictions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["path"] + [f"true_{c}" for c in classes] +
                        [f"{name}_{c}" for name in values for c in classes])
        for i, row in enumerate(validation):
            writer.writerow([row["path"], *row["target"],
                             *(float(values[name][i, c]) for name in values for c in range(len(classes)))])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=[CONDITION_TASK, LEAF_TASK], default=CONDITION_TASK)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--data", type=Path, default=Path("BRACOL_REVIEWED"))
    parser.add_argument("--manifest", type=Path, help="Leaf split manifest; defaults to checkpoint sibling")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--int8", action="store_true", help="Also export a calibrated QDQ INT8 candidate")
    parser.add_argument("--calibration-samples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-f1-drop", type=float, default=0.02,
                        help="Maximum absolute validation macro-F1 drop to recommend INT8")
    parser.add_argument("--max-class-f1-drop", type=float, default=0.05)
    args = parser.parse_args()
    args.checkpoint = args.checkpoint or Path(f"outputs/{args.task}/{args.task}.pt")
    args.output = args.output or Path(f"outputs/android/{args.task}")
    if args.calibration_samples < 1 or not 0 <= args.max_f1_drop <= 1 or not 0 <= args.max_class_f1_drop <= 1:
        parser.error("Calibration samples must be positive; F1 drop limits must be in [0,1]")
    try:
        import onnx
        import onnxruntime
    except ImportError:
        parser.error("Install export dependencies: python -m pip install -r requirements-export.txt")
    if args.output.exists():
        parser.error("Output must not exist. Choose a new directory to preserve previous exports.")
    torch.set_num_threads(4)
    model, saved = load_checkpoint(args.checkpoint, args.task)
    if args.task == LEAF_TASK:
        manifest = args.manifest or args.checkpoint.parent / "split_manifest.json"
        records, audit = audit_leaf_records(load_leaf_manifest(manifest))
        classes = LEAF_CLASSES
        args.data = Path(".")  # Manifest contains audited absolute image paths.
    else:
        if args.manifest:
            parser.error("--manifest is supported for the leaf task only")
        classes, records, audit = audit_dataset(args.data)
    if classes != saved["classes"]:
        raise ValueError("Dataset class order differs from checkpoint")
    args.output.mkdir(parents=True)
    write_json(args.output / "dataset_report.json", audit)
    fp32 = args.output / f"{args.task}.onnx"
    export_fp32(model, fp32)
    print(f"Exported FP32: {fp32.stat().st_size:,} bytes", flush=True)
    paths = {"fp32": fp32}
    calibration = []
    if args.int8:
        calibration = select_calibration(records, args.calibration_samples, args.seed)
        write_json(args.output / "calibration_manifest.json", calibration)
        paths["int8"] = args.output / f"{args.task}.int8.onnx"
        quantize_model(fp32, paths["int8"], calibration, args.data)
        print(f"Exported INT8 candidate: {paths['int8'].stat().st_size:,} bytes", flush=True)
    sessions = {name: make_session(path) for name, path in paths.items()}
    report = verify_models(model, sessions, records, args.data, classes, saved["threshold"], args.output,
                           args.task, saved.get("leaf_thresholds", LEAF_THRESHOLDS))
    recommended = "fp32"
    if args.int8:
        report["int8_quality_gate"] = quantization_quality(
            report["models"]["fp32"]["metrics"], report["models"]["int8"]["metrics"],
            args.max_f1_drop, args.max_class_f1_drop)
        if report["int8_quality_gate"]["passed"]:
            recommended = "int8"
    write_json(args.output / "validation_report.json", report)
    metadata = {
        "schema_version": 2, "task": args.task, "runtime": "ONNX Runtime Android", "opset": 17,
        "android_dependency": f"com.microsoft.onnxruntime:onnxruntime-android:{version('onnxruntime')}",
        "architecture": saved["architecture"], "classes": classes, "threshold": saved["threshold"],
        "input": {"name": "image", "dtype": "float32", "shape": [1, 3, 224, 224], "layout": "NCHW"},
        "output": {"name": "scores", "dtype": "float32", "shape": [1, len(classes)],
                   "activation": f"{saved['activation']}_included",
                   "classification": "binary" if args.task == LEAF_TASK else "multilabel"},
        "preprocessing": saved["preprocessing"],
        "models": {name: {"file": path.name, "bytes": path.stat().st_size,
                           "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                   for name, path in paths.items()},
        "recommended_model": paths[recommended].name,
        "recommendation_scope": "Validation quantization quality only; not diagnostic approval or device speed",
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "calibration": {"split": "train", "samples": len(calibration), "seed": args.seed} if args.int8 else None,
        "quantization": "QDQ signed INT8 Conv/Gemm/MatMul; other operations and public IO float32" if args.int8 else None,
        "versions": {p: version(p) for p in ("torch", "torchvision", "onnx", "onnxruntime", "numpy", "Pillow")},
        "desktop_verified": True, "android_device_tested": False,
        "no_detection_text": "No target condition detected",
    }
    if args.task == LEAF_TASK:
        metadata["leaf_thresholds"] = saved["leaf_thresholds"]
        metadata["no_detection_text"] = "No suitable coffee leaf detected"
        metadata["uncertain_text"] = "Please retake the photo"
    write_json(args.output / f"{args.task}.metadata.json", metadata)
    (args.output / f"{args.task}.labels.txt").write_text("\n".join(classes) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "recommended_model": metadata["recommended_model"],
                      "validation_macro_f1": {k: v["metrics"]["f1_macro"] for k, v in report["models"].items()},
                      "int8_quality_gate": report.get("int8_quality_gate"),
                      "android_device_tested": False}, indent=2), flush=True)


if __name__ == "__main__":
    main()
