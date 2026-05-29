import argparse
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf

from model import build_lstm_model
from utils import SEQUENCE_LENGTH, load_landmark_dataset, sort_label

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.h5"
LABELS_PATH = BASE_DIR / "labels.npy"
BATCH_SIZE = 32
EPOCHS = 80
RANDOM_STATE = 42


def parse_args():
    parser = argparse.ArgumentParser(description="Train the Signity ASL LSTM model.")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
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


def main():
    args = parse_args()
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

    y_cat = tf.keras.utils.to_categorical(y_encoded, num_classes=num_classes)
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
        test_size=0.15,
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
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(
            str(MODEL_PATH), save_best_only=True, monitor="val_accuracy"
        ),
    ]

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

    # Save the label mapping so inference knows the class text.
    print("Saving label classes...")
    np.save(LABELS_PATH, encoder.classes_)

    print("Training finished.")
    scores = model.evaluate(X_val, y_val, verbose=0)
    print(f"Validation loss: {scores[0]:.4f}, accuracy: {scores[1]:.4f}")


if __name__ == "__main__":
    main()
