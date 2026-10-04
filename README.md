# CoffeeGuardian

An Android hackathon prototype for screening coffee-leaf photos, with offline
inference using two MobileNetV3-Small models: coffee-leaf recognition, followed
by multilabel screening for **Cercospora, Miner, Phoma, and Rust**.

**[Download the v0.1 release](https://github.com/Hypnos8/coffee-disease-detection/releases/tag/v0)**

## Android app

Open the project in Android Studio, or build with an installed Android SDK:

```sh
./gradlew assembleDebug
```

See [MODEL_INTEGRATION.md](MODEL_INTEGRATION.md) for the app's model pipeline.

## Training and prediction

Install Python 3.10+ and matching PyTorch/torchvision packages. Dataset folders
and trained checkpoints are supplied separately and excluded from Git.

```sh
python -m pip install -r requirements.txt
python train.py --data BRACOL_REVIEWED --output outputs/condition_run
python train_leaf_classifier.py --output outputs/leaf_run
python predict_photo.py --image path/to/photo.jpg
```

Prediction requires checkpoints; use `--condition-checkpoint` and
`--leaf-checkpoint` for custom training outputs.

## Android integration contract

See [ANDROID_EXPORT.md](ANDROID_EXPORT.md) for ONNX export and preprocessing,
[LEAF_CLASSIFIER.md](LEAF_CLASSIFIER.md) for leaf classification, and
[BACKGROUND_AUGMENTATION.md](BACKGROUND_AUGMENTATION.md) for augmentation experiments.

This is an experimental screening tool. No detected target condition does not
establish that a plant is healthy; performance on real-world coffee-leaf phone
photos remains unvalidated.
