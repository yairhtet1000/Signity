import argparse
import os
import random
import site
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight


def configure_cuda_library_path():
    """Expose pip-installed NVIDIA CUDA libraries before TensorFlow imports."""
    candidate_roots = []
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        candidate_roots.append(
            Path(conda_prefix) / "lib" / "python3.10" / "site-packages" / "nvidia"
        )
    for site_dir in site.getsitepackages():
        candidate_roots.append(Path(site_dir) / "nvidia")

    lib_dirs = []
    for root in candidate_roots:
        if not root.exists():
            continue
        for package in [
            "cublas",
            "cuda_cupti",
            "cuda_nvrtc",
            "cuda_runtime",
            "cudnn",
            "cufft",
            "curand",
            "cusolver",
            "cusparse",
            "nccl",
            "nvjitlink",
        ]:
            lib_dir = root / package / "lib"
            if lib_dir.exists():
                lib_dirs.append(str(lib_dir))

    if lib_dirs:
        existing = os.environ.get("LD_LIBRARY_PATH", "")
        existing_parts = [part for part in existing.split(":") if part]
        os.environ["LD_LIBRARY_PATH"] = ":".join(
            list(dict.fromkeys(lib_dirs + existing_parts))
        )


configure_cuda_library_path()

import tensorflow as tf

from model import build_lstm_model
from utils import SEQUENCE_LENGTH, load_landmark_dataset, sort_label

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.h5"
LABELS_PATH = BASE_DIR / "labels.npy"
BATCH_SIZE = 32
EPOCHS = 80
RANDOM_STATE = 42
LOG_PATH = BASE_DIR / "training_log.csv"


def parse_args():
    parser = argparse.ArgumentParser(description="Train the Signity ASL LSTM model.")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument(
        "--early-stopping",
        action="store_true",
        help="Stop before --epochs when validation loss stops improving.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=12,
        help="Early-stopping patience. Used only with --early-stopping.",
    )
    parser.add_argument(
        "--validation-split",
        type=float,
        default=0.15,
        help="Fraction of samples reserved for validation.",
    )
    parser.add_argument(
        "--no-class-weights",
        action="store_true",
        help="Disable balanced class weights during training.",
    )
    parser.add_argument(
        "--no-words",
        action="store_true",
        help="Train only on the alphabet/digit image dataset.",
    )
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Re-extract MediaPipe landmarks from the image dataset.",
    )
    parser.add_argument(
        "--max-images-per-class",
        type=int,
        default=100,
        help="Limit image samples per alphabet/digit class. Use 0 for all images.",
    )
    parser.add_argument(
        "--max-word-samples-per-class",
        type=int,
        default=35,
        help="Limit word sequence samples per class. Use 0 for all word samples.",
    )
    return parser.parse_args()


def configure_tensorflow():
    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    tf.keras.utils.set_random_seed(RANDOM_STATE)

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"TensorFlow GPU devices: {gpus}")
    else:
        print("TensorFlow GPU devices: none. Training will use CPU.")


def main():
    args = parse_args()
    configure_tensorflow()
    print("Loading dataset...")
    X_seq, y = load_landmark_dataset(
        sequence_length=SEQUENCE_LENGTH,
        include_words=not args.no_words,
        refresh_cache=args.refresh_cache,
        max_images_per_class=args.max_images_per_class or None,
        max_word_samples_per_class=args.max_word_samples_per_class or None,
    )
    print(f"Loaded {len(X_seq)} sequence samples.")

    # Convert string labels into numeric indices for training.
    encoder = LabelEncoder()
    encoder.fit(sorted(set(y), key=sort_label))
    y_encoded = encoder.transform(y)
    num_classes = len(encoder.classes_)
    print(f"Found {num_classes} classes.")
    print("Sample counts for the first 20 classes:")
    counts = Counter(y)
    for label, count in counts.most_common(20):
        print(f"  {label}: {count}")
    print(
        f"Smallest class has {min(counts.values())} samples; "
        f"largest class has {max(counts.values())} samples."
    )

    y_cat = tf.keras.utils.to_categorical(y_encoded, num_classes=num_classes)
    class_weight_dict = None
    if not args.no_class_weights:
        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=np.unique(y_encoded),
            y=y_encoded,
        )
        class_weight_dict = {
            int(cls): float(weight)
            for cls, weight in zip(np.unique(y_encoded), class_weights)
        }

    X_train, X_val, y_train, y_val = train_test_split(
        X_seq,
        y_cat,
        test_size=args.validation_split,
        random_state=RANDOM_STATE,
        stratify=y_encoded,
    )

    model = build_lstm_model(
        sequence_length=SEQUENCE_LENGTH,
        feature_dim=X_seq.shape[2],
        num_classes=num_classes,
    )
    model.summary()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(MODEL_PATH),
            save_best_only=True,
            monitor="val_accuracy",
            mode="max",
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(str(LOG_PATH)),
    ]
    if args.early_stopping:
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=args.patience,
                restore_best_weights=True,
            )
        )

    print("Starting training...")
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=2,
    )

    best_epoch = int(np.argmax(history.history["val_accuracy"])) + 1
    best_val_accuracy = float(np.max(history.history["val_accuracy"]))
    print(
        f"Best validation accuracy during this run: "
        f"{best_val_accuracy:.4f} at epoch {best_epoch}."
    )

    # Save the label mapping so inference knows the class text.
    print("Saving label classes...")
    np.save(LABELS_PATH, encoder.classes_)

    print("Training finished.")
    best_model = tf.keras.models.load_model(str(MODEL_PATH))
    scores = best_model.evaluate(X_val, y_val, verbose=0)
    print(
        f"Saved best model validation loss: {scores[0]:.4f}, "
        f"accuracy: {scores[1]:.4f}, top-3: {scores[2]:.4f}"
    )


if __name__ == "__main__":
    main()
