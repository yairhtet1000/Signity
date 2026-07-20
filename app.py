import base64
from io import BytesIO
from pathlib import Path
import sys

# import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, render_template, request, send_file
from gtts import gTTS

from utils import (
    FEATURE_DIM,
    SEQUENCE_LENGTH,
    decode_base64_image,
    dataset_summary,
    extract_landmarks,
    prepare_sequence,
)

# Paths for the trained model and the saved label name mapping.
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.h5"
LABELS_PATH = BASE_DIR / "labels.npy"
MIN_CONFIDENCE = 0.40
MIN_MARGIN = 0.03

app = Flask(__name__)

# Cache the model and labels in memory so repeated requests are fast.
model = None
label_classes = None

# We prefer the classic `mp.solutions.hands` API when available because
# our utilities were written against it. If it's not available (newer
# mediapipe versions use the Tasks API), we attempt a Tasks-based fallback
# that requires a local model file at `models/hand_landmarker.task`.
hands = None
try:
    # Try the legacy API first
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
except Exception:
    # Fall back to the Tasks API if present.
    try:
        from mediapipe.tasks.python import vision as mp_tasks_vision
        from mediapipe.tasks.python.core import base_options as mp_base_options
        from mediapipe.tasks.python.vision.core import (
            image as mp_image_module,
            vision_task_running_mode as mp_running_mode,
        )
    except Exception:
        print(
            "MediaPipe is installed but neither the `solutions` nor the Tasks API is usable.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Look for a downloaded task model under ./models/hand_landmarker.task
    MODEL_TASK_PATH = BASE_DIR / "models" / "hand_landmarker.task"
    if not MODEL_TASK_PATH.exists():
        print(
            "MediaPipe Tasks API detected but the hand landmarker model is missing.",
            file=sys.stderr,
        )
        print("Download the model and place it at:", file=sys.stderr)
        print(str(MODEL_TASK_PATH), file=sys.stderr)
        print(
            "Suggested model URL: https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker.task",
            file=sys.stderr,
        )
        print(
            "After downloading, re-run the app. If you want, I can download the model automatically.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Create a Tasks-based HandLandmarker and wrap it to expose a `process()`
    # method compatible with the classic `mp.solutions` results format.
    options = mp_tasks_vision.HandLandmarkerOptions(
        base_options=mp_base_options.BaseOptions(model_asset_path=str(MODEL_TASK_PATH)),
        running_mode=mp_running_mode.VisionTaskRunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    tasks_landmarker = mp_tasks_vision.HandLandmarker.create_from_options(options)

    class TasksHandsWrapper:
        """Adapter that exposes a `process(image_rgb)` method returning an
        object with `multi_hand_landmarks`, mimicking `mp.solutions.hands`.
        """

        def __init__(self, landmarker, mp_tasks_vision):
            self.landmarker = landmarker
            self.mp_vis = mp_tasks_vision

        def process(self, image_rgb):
            # Convert numpy RGB image to a mediapipe.tasks Image.
            # The installed Tasks API expects an Image constructed from the
            # numpy array and a format enum.
            mp_image = mp_image_module.Image(
                mp_image_module.ImageFormat.SRGB,
                image_rgb,
            )
            result = self.landmarker.detect(mp_image)

            # Build a compatibility object where each hand provides a
            # `.landmark` iterable with items that have `.x/.y/.z`.
            class _Hand:
                def __init__(self, landmarks):
                    self.landmark = landmarks

            class _Result:
                def __init__(self, hands_list):
                    self.multi_hand_landmarks = hands_list

            hands_list = []
            if hasattr(result, "hand_landmarks") and result.hand_landmarks:
                for hand_kp in result.hand_landmarks:
                    hands_list.append(_Hand(hand_kp))

            return _Result(hands_list)

    hands = TasksHandsWrapper(tasks_landmarker, mp_tasks_vision)


def load_model_and_labels():
    """Load the trained TensorFlow model and its recorded label names.

    The model is loaded only once and reused for later requests. The label
    names are saved by `train.py` so we can map model indices back to text.
    """
    global model, label_classes

    if model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                "Trained model not found. Run python train.py first."
            )
        model = tf.keras.models.load_model(str(MODEL_PATH))

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
        if expected_sequence_length not in (None, SEQUENCE_LENGTH):
            raise ValueError(
                f"Model expects {expected_sequence_length} frames, but the app "
                f"is configured for {SEQUENCE_LENGTH}. Re-run python train.py."
            )
        if expected_feature_dim != FEATURE_DIM:
            raise ValueError(
                f"Model expects {expected_feature_dim} features per frame, but "
                f"the current MediaPipe hand pipeline produces {FEATURE_DIM}."
            )


def model_assets_ready(summary=None):
    """Return True when saved model files match the detected dataset classes."""
    if not MODEL_PATH.exists() or not LABELS_PATH.exists():
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


@app.route("/")
def index():
    """Render the home page with dataset summary and model readiness state."""
    summary = dataset_summary()
    has_model = model_assets_ready(summary)
    return render_template("index.html", summary=summary, has_model=has_model)


@app.route("/live")
def live():
    """Render the live webcam interface for prediction."""
    has_model = model_assets_ready()
    return render_template("live.html", has_model=has_model)


@app.route("/predict", methods=["POST"])
def predict():
    """Receive a webcam snapshot, detect hand landmarks, and return a class."""
    if not request.is_json:
        return jsonify({"error": "Expected application/json body."}), 400

    payload = request.get_json()
    image_data = payload.get("image")
    image_sequence = payload.get("images")

    if not image_data and not image_sequence:
        return jsonify({"error": "Missing image data."}), 400

    load_model_and_labels()

    if image_sequence and isinstance(image_sequence, list):
        landmarks_sequence = []
        for element in image_sequence:
            try:
                frame = decode_base64_image(element)
            except Exception:
                continue
            landmarks = extract_landmarks(frame, hands)
            if landmarks is not None:
                landmarks_sequence.append(landmarks)

        if not landmarks_sequence:
            return jsonify(
                {"error": "No hand landmarks detected in the image sequence."}
            ), 200

        sequence = prepare_sequence(
            np.stack(landmarks_sequence, axis=0),
            sequence_length=SEQUENCE_LENGTH,
        )
    else:
        try:
            image = decode_base64_image(image_data)
        except Exception as exc:
            return jsonify({"error": f"Unable to decode image: {exc}"}), 400

        landmarks = extract_landmarks(image, hands)
        if landmarks is None:
            return jsonify({"error": "No hand landmarks detected."}), 200

        sequence = prepare_sequence(landmarks, sequence_length=SEQUENCE_LENGTH)

    try:
        prediction = model.predict(np.array([sequence], dtype=np.float32), verbose=0)[0]
    except Exception as exc:
        return jsonify({"error": f"Prediction failed: {exc}"}), 500
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
    is_confident = (
        confidence >= MIN_CONFIDENCE and (confidence - second_confidence) >= MIN_MARGIN
    )

    result = {
        "label": str(label_classes[best_i]) if is_confident else "Unsure",
        "raw_label": str(label_classes[best_i]),
        "confidence": confidence,
        "is_confident": is_confident,
        "top": top,
        "all": [float(x) for x in prediction.tolist()],
    }
    return jsonify(result)


@app.route("/tts")
def tts():
    """Generate spoken audio for the given text query parameter."""
    text = request.args.get("text", "")
    if not text:
        return jsonify({"error": "Missing text query parameter."}), 400

    tts_audio = gTTS(text=text, lang="en")
    mp3_buffer = BytesIO()
    tts_audio.write_to_fp(mp3_buffer)
    mp3_buffer.seek(0)
    return send_file(mp3_buffer, mimetype="audio/mpeg")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5500, debug=True)
