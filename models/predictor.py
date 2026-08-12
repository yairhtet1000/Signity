"""Keras model loading, sequence preparation, and thread-safe inference."""

import threading
from pathlib import Path

import numpy as np
import tensorflow as tf

from config import (
    INFERENCE_DEBUG,
    LABELS_PATH,
    LEGACY_MODEL_PATH,
    MIN_VALID_FRAME_RATIO,
    MODEL_PATH,
)
from models.landmark_extractor import hands
from utils import (
    FEATURE_DIM,
    decode_base64_image,
    extract_landmarks,
    prepare_sequence,
    dataset_summary,
)

inference_lock = threading.Lock()

model = None
label_classes = None


def available_model_path():
    if MODEL_PATH.exists():
        return MODEL_PATH
    if LEGACY_MODEL_PATH.exists():
        return LEGACY_MODEL_PATH
    return None


def prediction_sequence_length():
    if model is None:
        from utils import SEQUENCE_LENGTH
        return SEQUENCE_LENGTH
    expected_length = model.input_shape[1]
    return int(expected_length) if expected_length is not None else 20


def load_model_and_labels(app_logger=None):
    global model, label_classes

    if model is None:
        model_path = available_model_path()
        if model_path is None:
            raise FileNotFoundError(
                "Trained model not found. Run python train.py first."
            )
        model = tf.keras.models.load_model(str(model_path), compile=False)
        if any(layer.__class__.__name__ == "Masking" for layer in model.layers):
            if app_logger:
                app_logger.warning(
                    "Loaded model contains a legacy Masking layer before Conv1D. "
                    "Retrain with the current pipeline for reliable live inference."
                )

    if label_classes is None:
        if not LABELS_PATH.exists():
            raise FileNotFoundError(
                "Label classes not found. Run python train.py first."
            )
        label_classes = np.load(str(LABELS_PATH), allow_pickle=True)

    model_output = int(model.output_shape[-1])
    if model_output != len(label_classes):
        raise ValueError(
            f"Model output has {model_output} classes, but labels.npy has "
            f"{len(label_classes)} labels. Re-run python train.py."
        )

    expected_shape = model.input_shape
    if len(expected_shape) == 3:
        _, expected_sequence_length, expected_feature_dim = expected_shape
        if expected_feature_dim != FEATURE_DIM:
            raise ValueError(
                f"Model expects {expected_feature_dim} features per frame, but "
                f"the current MediaPipe hand pipeline produces {FEATURE_DIM}."
            )


def model_assets_ready(summary=None):
    if available_model_path() is None or not LABELS_PATH.exists():
        return False
    if summary is None:
        summary = dataset_summary()

    expected_classes = len(summary["rows"])
    if summary["words"]["present"]:
        expected_classes += max(
            (item["classes"] for item in summary["words"]["sets"]), default=0
        )

    try:
        labels = np.load(str(LABELS_PATH), allow_pickle=True)
    except Exception:
        return False

    return expected_classes == 0 or len(labels) == expected_classes


def sequence_from_live_frames(image_sequence, expected_length):
    if not isinstance(image_sequence, list):
        image_sequence = []

    if len(image_sequence) < expected_length:
        image_sequence = image_sequence + [None] * (expected_length - len(image_sequence))
    elif len(image_sequence) > expected_length:
        image_sequence = image_sequence[:expected_length]

    landmarks_sequence = []
    for element in image_sequence:
        if element is None:
            continue
        try:
            frame = decode_base64_image(element)
        except Exception:
            continue
        if frame is None:
            continue
        with inference_lock:
            lm = extract_landmarks(frame, hands)
        if lm is not None and lm.shape == (FEATURE_DIM,):
            landmarks_sequence.append(lm)

    minimum_valid_frames = max(4, int(np.ceil(expected_length * MIN_VALID_FRAME_RATIO)))
    if len(landmarks_sequence) < minimum_valid_frames:
        return None, len(landmarks_sequence)

    sequence = prepare_sequence(
        np.stack(landmarks_sequence), sequence_length=expected_length
    )
    if sequence.shape != (expected_length, FEATURE_DIM):
        raise ValueError(f"Invalid sequence shape: {sequence.shape}.")
    return sequence, len(landmarks_sequence)


def log_inference(sequence, probabilities, predicted_label, confidence, valid_frames, app_logger=None):
    if INFERENCE_DEBUG:
        msg = (
            "inference valid_frames=%s landmarks=%s sequence=%s probabilities=%s label=%s confidence=%.4f",
            valid_frames,
            (FEATURE_DIM,),
            sequence.shape,
            np.array2string(probabilities, precision=4, threshold=probabilities.size),
            predicted_label,
            confidence,
        )
        if app_logger:
            app_logger.info(*msg)
        else:
            print(msg)


def run_prediction(account, image_sequence, app_logger=None):
    expected_length = prediction_sequence_length()
    if expected_length > 32:
        return api_response_error("The loaded model has an unsupported sequence length.", 503)

    try:
        sequence, valid_frames = sequence_from_live_frames(image_sequence, expected_length)
    except ValueError as exc:
        return api_response_error(str(exc), 422)

    if sequence is None:
        return api_response_success({
            "label": "Unsure",
            "raw_label": "Unsure",
            "confidence": 0.0,
            "is_confident": False,
            "has_clear_margin": False,
            "top": [],
            "sequence_shape": [1, expected_length, FEATURE_DIM],
            "valid_frames": valid_frames,
        })

    try:
        input_tensor = np.expand_dims(sequence, axis=0).astype(np.float32)
        with inference_lock:
            prediction = model.predict(input_tensor, verbose=0)[0]
    except Exception as exc:
        if app_logger:
            app_logger.exception("Prediction failed")
        return api_response_error("Prediction failed.", 500)

    best_i = int(np.argmax(prediction))
    top_indices = np.argsort(prediction)[-3:][::-1]
    top = [
        {
            "label": str(label_classes[index]),
            "confidence": float(prediction[index]),
        }
        for index in top_indices
    ]
    second_confidence = (
        float(prediction[top_indices[1]]) if len(top_indices) > 1 else 0.0
    )
    confidence = float(prediction[best_i])
    meets_confidence = confidence >= 0.50
    has_clear_margin = (confidence - second_confidence) >= 0.02

    result = {
        "label": str(label_classes[best_i]) if meets_confidence else "Unsure",
        "raw_label": str(label_classes[best_i]),
        "confidence": confidence,
        "is_confident": meets_confidence,
        "has_clear_margin": has_clear_margin,
        "top": top,
        "sequence_shape": list(input_tensor.shape),
        "valid_frames": valid_frames,
    }
    log_inference(sequence, prediction, result["raw_label"], confidence, valid_frames, app_logger)
    return api_response_success(result)


def api_response_success(data=None, status=200):
    from shared import api_response
    return api_response(data, status)


def api_response_error(message, status=400):
    from shared import api_error
    return api_error(message, status)
