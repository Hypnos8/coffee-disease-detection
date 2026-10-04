# Coffee photo screening

The app bundles two ONNX models in `app/src/main/assets/`, with a separate metadata and label file for each:

- `coffee_leaf_classifier.onnx` checks whether the image shows a coffee leaf.
- `coffee_condition_classifier.onnx` screens accepted coffee leaves for Cercospora, Miner, Phoma, and Rust.

The leaf model runs first. Its softmax scores are ordered `not_coffee_leaf`, then `coffee_leaf`. The coffee-leaf score at or below the metadata reject threshold (0.20) returns “not recognized as a coffee leaf”; scores at or above the accept threshold (0.80) continue to condition screening; intermediate scores ask for a retake. The disease model runs only after acceptance. Its independent sigmoid scores are already activated; each class at or above the metadata threshold (0.50) is reported. Do not add a softmax or sigmoid.

Both exports use float32 RGB NCHW input `[1, 3, 224, 224]`, with aspect-preserving bilinear fit, centered white padding, and ImageNet channel normalization. Preprocessing and thresholds are read from the bundled metadata. The model sessions are retained for the screen lifetime and inference runs off the UI thread.

An accepted leaf with no disease score at threshold is reported as “No health issue detected in this photo,” followed by a note that screening can miss problems and does not confirm the plant is healthy. This is not a diagnosis. Non-leaf, retake, no-issue, and detected-condition outcomes are stored distinctly in the existing scan result field; technical failures are not saved as predictions.

When an existing version-1 database is upgraded, its scan history is cleared because previous app versions recorded placeholder predictions. Profile records remain intact.

Android integration fixtures for both models live in `app/src/androidTest/assets/`. The connected model tests compare device ONNX scores against the exported expected scores. JVM tests cover leaf threshold boundaries. The Room migration test checks that old scans are removed while profiles are preserved.
