import argparse
import math
import os
import random
from collections import Counter
from pathlib import Path

# Keep ordinary TensorFlow startup information out of training logs while
# retaining warnings and errors that are useful when diagnosing GPU issues.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from cuda_config import prepare_tensorflow_cuda

prepare_tensorflow_cuda()

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf

from model import build_lstm_model
from utils import SEQUENCE_LENGTH, load_landmark_dataset, sort_label

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.keras"
LABELS_PATH = BASE_DIR / "labels.npy"
LOG_PATH = BASE_DIR / "training_log.csv"
REPORTS_DIR = BASE_DIR / "reports" / "training_evaluation"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 80
EPOCHS = 80
LEARNING_RATE = 1e-3
RANDOM_STATE = 42


def parse_args():
    parser = argparse.ArgumentParser(description="Train the Signity ASL LSTM model.")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    early_stopping_group = parser.add_mutually_exclusive_group()
    early_stopping_group.add_argument(
        "--early-stopping",
        dest="early_stopping",
        action="store_true",
        help="Enable early stopping (the default).",
    )
    early_stopping_group.add_argument(
        "--no-early-stopping",
        dest="early_stopping",
        action="store_false",
        help="Always train for the full number of epochs.",
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
    parser.set_defaults(early_stopping=True)
    return parser.parse_args()


def configure_tensorflow():
    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    tf.keras.utils.set_random_seed(RANDOM_STATE)


def _get_device_context():
    gpus = tf.config.list_physical_devices("GPU")
    return tf.device("/GPU:0") if gpus else tf.device("/CPU:0")


def export_training_plots(history, output_dir=REPORTS_DIR):
    """Save accuracy, loss, and combined metric plots from Keras history."""
    output_dir.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history.history["loss"]) + 1)
    acc = history.history.get("accuracy", [])
    val_acc = history.history.get("val_accuracy", [])
    loss = history.history.get("loss", [])
    val_loss = history.history.get("val_loss", [])

    accuracy_path = output_dir / "training_accuracy_curve.png"
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, acc, label="Training Accuracy", marker="o", linewidth=2)
    ax.plot(epochs, val_acc, label="Validation Accuracy", marker="o", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Training vs. Validation Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.5)
    fig.tight_layout()
    fig.savefig(str(accuracy_path), dpi=300)
    plt.close(fig)
    print(f"Saved accuracy curve to: {accuracy_path}")

    loss_path = output_dir / "training_loss_curve.png"
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, loss, label="Training Loss", marker="o", linewidth=2)
    ax.plot(epochs, val_loss, label="Validation Loss", marker="o", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training vs. Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.5)
    fig.tight_layout()
    fig.savefig(str(loss_path), dpi=300)
    plt.close(fig)
    print(f"Saved loss curve to: {loss_path}")

    summary_path = output_dir / "training_metrics_summary.png"
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(epochs, acc, label="Training", marker="o", linewidth=2)
    axes[0].plot(epochs, val_acc, label="Validation", marker="o", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.5)
    axes[1].plot(epochs, loss, label="Training", marker="o", linewidth=2)
    axes[1].plot(epochs, val_loss, label="Validation", marker="o", linewidth=2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.5)
    fig.suptitle("Training Metrics Summary", fontsize=14)
    fig.tight_layout()
    fig.savefig(str(summary_path), dpi=300)
    plt.close(fig)
    print(f"Saved metrics summary to: {summary_path}")


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

    train_ds = (
        tf.data.Dataset.from_tensor_slices((X_train, y_train))
        .shuffle(buffer_size=len(X_train))
        .batch(args.batch_size)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )
    val_ds = (
        tf.data.Dataset.from_tensor_slices((X_val, y_val))
        .batch(args.batch_size)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

    steps_per_epoch = math.ceil(len(X_train) / args.batch_size)
    decay_steps = max(1, args.epochs * steps_per_epoch)

    device_context = _get_device_context()
    with device_context:
        print(f"Building and training model on: {device_context}")
        model = build_lstm_model(
            sequence_length=SEQUENCE_LENGTH,
            feature_dim=X_seq.shape[2],
            num_classes=num_classes,
            learning_rate=args.learning_rate,
            decay_steps=decay_steps,
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
            tf.keras.callbacks.TerminateOnNaN(),
        ]
        if args.early_stopping:
            callbacks.append(
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=args.patience,
                    restore_best_weights=True,
                    verbose=1,
                )
            )

        print("Starting training...")
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.epochs,
            class_weight=class_weight_dict,
            callbacks=callbacks,
            verbose=2,
        )

        test_tensor = tf.constant([[1.0, 2.0]])
        print(f"TensorFlow device placement verified: {test_tensor.device}")

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
    val_eval_ds = (
        tf.data.Dataset.from_tensor_slices((X_val, y_val))
        .batch(args.batch_size)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )
    scores = best_model.evaluate(val_eval_ds, verbose=0)
    print(
        f"Saved best model validation loss: {scores[0]:.4f}, "
        f"accuracy: {scores[1]:.4f}, top-3: {scores[2]:.4f}"
    )

    export_training_plots(history)


if __name__ == "__main__":
    main()
