#!/usr/bin/env python3
"""Evaluation pipeline for the Signity ASL LSTM classifier."""

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from cuda_config import prepare_tensorflow_cuda

prepare_tensorflow_cuda()

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    top_k_accuracy_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from utils import FEATURE_DIM, SEQUENCE_LENGTH, load_landmark_dataset, sort_label

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.keras"
LABELS_PATH = BASE_DIR / "labels.npy"
REPORTS_DIR = BASE_DIR / "reports" / "evaluation"

RANDOM_STATE = 42
TEST_SIZE = 0.15


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate the Signity ASL LSTM model on a held-out test split."
    )
    parser.add_argument(
        "--no-words",
        action="store_true",
        help="Evaluate only on the alphabet/digit image dataset.",
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
        help="Limit image samples per alphabet/digit class. Use 0 for all.",
    )
    parser.add_argument(
        "--max-word-samples-per-class",
        type=int,
        default=35,
        help="Limit word sequence samples per class. Use 0 for all word samples.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(REPORTS_DIR),
        help="Directory to save generated report PNGs.",
    )
    return parser.parse_args()


def configure_tensorflow():
    """Set seeds for reproducible evaluation."""
    np.random.seed(RANDOM_STATE)
    tf.keras.utils.set_random_seed(RANDOM_STATE)


def _get_device_context():
    gpus = tf.config.list_physical_devices("GPU")
    return tf.device("/GPU:0") if gpus else tf.device("/CPU:0")


def _classify_label(label):
    """Classify a label string into alphabet, digit, or word category."""
    value = str(label)
    if value.isdigit():
        return "digits"
    if len(value) == 1 and value.isalpha():
        return "alphabet"
    return "words"


def _save_confusion_matrix(cm, labels, title, output_path, mask_zero=False):
    """Plot and save a confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(10, 8))
    mask = cm == 0 if mask_zero else None
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        mask=mask,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=200)
    plt.close("all")
    print(f"Saved {output_path}")


def _save_top_confused_pairs(cm, labels, output_path, top_k=10):
    """Plot the top confused pairs as a horizontal bar chart."""
    cm_copy = cm.copy()
    np.fill_diagonal(cm_copy, 0)

    flat_indices = np.argsort(cm_copy.ravel())[::-1]
    pairs = []
    for idx in flat_indices:
        if len(pairs) >= top_k:
            break
        true_idx, pred_idx = np.unravel_index(idx, cm_copy.shape)
        count = int(cm_copy[true_idx, pred_idx])
        if count == 0:
            break
        pairs.append((str(labels[true_idx]), str(labels[pred_idx]), count))

    if not pairs:
        print("No misclassifications found for top confused pairs.")
        return

    fig, ax = plt.subplots(figsize=(10, 8))
    y_pos = np.arange(len(pairs))
    counts = [p[2] for p in pairs]
    pair_labels = [f"True: '{p[0]}' -> Pred: '{p[1]}'" for p in pairs]

    ax.barh(y_pos, counts, color="steelblue")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(pair_labels)
    ax.invert_yaxis()
    ax.set_xlabel("Misclassification Count")
    ax.set_title("Top Confused Pairs")
    ax.grid(True, alpha=0.5, axis="x")
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=200)
    plt.close("all")
    print(f"Saved {output_path}")


def main():
    args = parse_args()
    configure_tensorflow()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not MODEL_PATH.exists():
        print(f"ERROR: Model not found at {MODEL_PATH}. Run python train.py first.")
        sys.exit(1)
    if not LABELS_PATH.exists():
        print(f"ERROR: Labels not found at {LABELS_PATH}. Run python train.py first.")
        sys.exit(1)

    print("Loading dataset...")
    X_seq, y = load_landmark_dataset(
        sequence_length=SEQUENCE_LENGTH,
        include_words=not args.no_words,
        refresh_cache=args.refresh_cache,
        max_images_per_class=args.max_images_per_class or None,
        max_word_samples_per_class=args.max_word_samples_per_class or None,
    )
    print(f"Loaded {len(X_seq)} total sequence samples.")

    encoder = LabelEncoder()
    encoder.fit(sorted(set(y), key=sort_label))
    y_encoded = encoder.transform(y)
    num_classes = len(encoder.classes_)
    print(f"Found {num_classes} classes.")

    X_train, X_test, y_train_enc, y_test_enc = train_test_split(
        X_seq,
        y_encoded,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_encoded,
    )
    print(f"Test split size: {len(X_test)} samples.")

    device_context = _get_device_context()
    with device_context:
        print(f"Loading model on: {device_context}")
        model = tf.keras.models.load_model(str(MODEL_PATH), compile=False)
        label_classes = np.load(str(LABELS_PATH), allow_pickle=True)

        model_output = int(model.output_shape[-1])
        if model_output != len(label_classes):
            print(
                f"ERROR: Model output has {model_output} classes, but labels.npy has "
                f"{len(label_classes)} labels. Re-run python train.py."
            )
            sys.exit(1)

        expected_feature_dim = (
            int(model.input_shape[-1])
            if model.input_shape[-1] is not None
            else FEATURE_DIM
        )
        if expected_feature_dim != FEATURE_DIM:
            print(
                f"ERROR: Model expects {expected_feature_dim} features per frame, but "
                f"the current pipeline produces {FEATURE_DIM}. Re-run python train.py."
            )
            sys.exit(1)

        test_tensor = tf.constant([[1.0, 2.0]])
        print(f"TensorFlow device placement verified: {test_tensor.device}")

    with device_context:
        print("Running inference on test set...")
        y_proba = model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_proba, axis=1)

    acc = accuracy_score(y_test_enc, y_pred)
    top3_acc = top_k_accuracy_score(
        y_test_enc, y_proba, k=3, labels=range(model_output)
    )

    report_dict = classification_report(
        y_test_enc,
        y_pred,
        labels=range(model_output),
        target_names=[str(c) for c in label_classes],
        output_dict=True,
        zero_division=0,
    )
    macro_f1 = float(report_dict["macro avg"]["f1-score"])

    print("\n" + "=" * 50)
    print(f"{'Metric':<30} {'Value':>10}")
    print("-" * 50)
    print(f"{'Overall Categorical Accuracy':<30} {acc:>10.4f}")
    print(f"{'Top-3 Accuracy':<30} {top3_acc:>10.4f}")
    print(f"{'Macro F1-Score':<30} {macro_f1:>10.4f}")
    print("=" * 50)

    print("\nPer-class Precision, Recall, and F1-Score:")
    print(
        classification_report(
            y_test_enc,
            y_pred,
            labels=range(model_output),
            target_names=[str(c) for c in label_classes],
            zero_division=0,
        )
    )

    category_indices = {"alphabet": [], "digits": [], "words": []}
    category_labels = {"alphabet": [], "digits": [], "words": []}
    for i, label in enumerate(label_classes):
        cat = _classify_label(label)
        category_indices[cat].append(i)
        category_labels[cat].append(str(label))

    full_cm = confusion_matrix(y_test_enc, y_pred, labels=range(model_output))

    for cat_name in ["alphabet", "digits", "words"]:
        indices = category_indices[cat_name]
        if not indices:
            continue
        labels = category_labels[cat_name]

        if cat_name == "words":
            word_counts = [(i, int(np.sum(y_test_enc == i))) for i in indices]
            if not any(count for _, count in word_counts):
                continue
            word_counts.sort(key=lambda item: item[1], reverse=True)
            selected = [idx for idx, _ in word_counts[:25]]
            labels = [str(label_classes[i]) for i in selected]
            sub_cm = full_cm[np.ix_(selected, selected)]
        else:
            sub_cm = full_cm[np.ix_(indices, indices)]

        title = f"Confusion Matrix - {cat_name.capitalize()}"
        output_path = output_dir / f"cm_{cat_name}.png"
        _save_confusion_matrix(sub_cm, labels, title, output_path, mask_zero=True)

    top_confused_path = output_dir / "top_confused_pairs.png"
    _save_top_confused_pairs(
        full_cm,
        [str(c) for c in label_classes],
        top_confused_path,
        top_k=10,
    )

    print(f"\nGenerated report artifacts in: {output_dir}")
    for cat_name in ["alphabet", "digits", "words"]:
        if category_indices[cat_name]:
            print(f"  - {output_dir / f'cm_{cat_name}.png'}")
    print(f"  - {output_dir / 'top_confused_pairs.png'}")


if __name__ == "__main__":
    main()
