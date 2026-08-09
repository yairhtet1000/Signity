import os
import secrets
import sqlite3
import sys
from io import BytesIO
from pathlib import Path

# Keep ordinary TensorFlow startup information out of the server log while
# retaining warnings and errors that are useful when diagnosing GPU issues.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# import cv2
import mediapipe as mp
import numpy as np

from cuda_config import prepare_tensorflow_cuda

# This must happen before importing TensorFlow so its pip-installed CUDA
# libraries are discoverable on Pop!_OS/Linux.
prepare_tensorflow_cuda()

import tensorflow as tf
from flask import flash, jsonify, redirect, render_template, request, send_file, session, url_for
from gtts import gTTS
from werkzeug.security import check_password_hash, generate_password_hash

from database import get_connection, initialize_database

from utils import (
    FEATURE_DIM,
    SEQUENCE_LENGTH,
    dataset_summary,
    decode_base64_image,
    extract_landmarks,
    prepare_sequence,
)

# Paths for the trained model and the saved label name mapping.
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.keras"
LEGACY_MODEL_PATH = BASE_DIR / "model.h5"
LABELS_PATH = BASE_DIR / "labels.npy"
MIN_CONFIDENCE = 0.40
MIN_MARGIN = 0.03

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SIGNITY_SECRET_KEY", secrets.token_urlsafe(32))
initialize_database()

# Cache the model and labels in memory so repeated requests are fast.
model = None
label_classes = None


def current_account():
    role, account_id = session.get("role"), session.get("account_id")
    if role not in {"admin", "user"} or not account_id:
        return None
    table = "Admin" if role == "admin" else "User"
    with get_connection() as connection:
        row = connection.execute(f'SELECT id, name, email FROM "{table}" WHERE id = ?', (account_id,)).fetchone()
        if row is None:
            session.clear()
            return None
        approved = role == "admin" or connection.execute('SELECT 1 FROM "UserApprove" WHERE userId = ?', (account_id,)).fetchone() is not None
    return {**dict(row), "role": role, "approved": approved}


def require_access(api=False, admin_only=False):
    account = current_account()
    allowed = account and (account["role"] == "admin" if admin_only else account["role"] == "admin" or account["approved"])
    if allowed:
        return account
    if api:
        return jsonify({"error": "Admin approval is required." if account else "Sign in to use the interpreter."}), 403
    flash("Your account needs admin approval." if account else "Sign in to continue.", "warning")
    return redirect(url_for("login"))


def record_history(user_id, interpreted_text):
    if session.get("last_interpreted_text") == interpreted_text:
        return
    with get_connection() as connection:
        connection.execute('INSERT INTO "UserHistory" (userId, interpretedTexts) VALUES (?, ?)', (user_id, interpreted_text))
    session["last_interpreted_text"] = interpreted_text

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
        )
        from mediapipe.tasks.python.vision.core import (
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

            class _Classification:
                def __init__(self, label):
                    self.label = label

            class _Handedness:
                def __init__(self, label):
                    self.classification = [_Classification(label)]

            class _Result:
                def __init__(self, hands_list, handedness_list):
                    self.multi_hand_landmarks = hands_list
                    self.multi_handedness = handedness_list

            hands_list = []
            handedness_list = []
            if hasattr(result, "hand_landmarks") and result.hand_landmarks:
                for index, hand_kp in enumerate(result.hand_landmarks):
                    hands_list.append(_Hand(hand_kp))
                    categories = (getattr(result, "handedness", []) or [])
                    category = categories[index][0] if index < len(categories) and categories[index] else None
                    label = getattr(category, "category_name", "")
                    handedness_list.append(_Handedness(label))

            return _Result(hands_list, handedness_list)

    hands = TasksHandsWrapper(tasks_landmarker, mp_tasks_vision)


def available_model_path():
    """Prefer the native Keras model, while retaining legacy .h5 support."""
    if MODEL_PATH.exists():
        return MODEL_PATH
    if LEGACY_MODEL_PATH.exists():
        return LEGACY_MODEL_PATH
    return None


def prediction_sequence_length():
    """Use the active model's sequence length, including legacy 5-frame models."""
    if model is None:
        return SEQUENCE_LENGTH
    expected_length = model.input_shape[1]
    return int(expected_length) if expected_length is not None else SEQUENCE_LENGTH


def load_model_and_labels():
    """Load the trained TensorFlow model and its recorded label names.

    The model is loaded only once and reused for later requests. The label
    names are saved by `train.py` so we can map model indices back to text.
    """
    global model, label_classes

    if model is None:
        model_path = available_model_path()
        if model_path is None:
            raise FileNotFoundError(
                "Trained model not found. Run python train.py first."
            )
        model = tf.keras.models.load_model(str(model_path))

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
    """Return True when saved model files match the detected dataset classes."""
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


@app.route("/")
def index():
    """Render the home page with dataset summary and model readiness state."""
    summary = dataset_summary()
    has_model = model_assets_ready(summary)
    return render_template("index.html", summary=summary, has_model=has_model, account=current_account())


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name, email, password = request.form.get("name", "").strip(), request.form.get("email", "").strip().lower(), request.form.get("password", "")
        if not name or not email or len(password) < 8:
            flash("Enter a name and email, and use a password with at least 8 characters.", "error")
        else:
            try:
                with get_connection() as connection:
                    connection.execute('INSERT INTO "User" (name, email, password) VALUES (?, ?, ?)', (name, email, generate_password_hash(password)))
            except sqlite3.IntegrityError:
                flash("An account with that email already exists.", "error")
            else:
                flash("Registration received. An admin must approve your account before access is enabled.", "success")
                return redirect(url_for("login"))
    return render_template("register.html", account=None)


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email, password = request.form.get("email", "").strip().lower(), request.form.get("password", "")
        role = request.form.get("role", "user")
        role = role if role in {"user", "admin"} else "user"
        table = "Admin" if role == "admin" else "User"
        with get_connection() as connection:
            account = connection.execute(f'SELECT id, password FROM "{table}" WHERE email = ?', (email,)).fetchone()
        if account is None or not check_password_hash(account["password"], password):
            flash("Invalid email, password, or account type.", "error")
        else:
            session.clear()
            session.update(account_id=account["id"], role=role)
            if role == "user" and not current_account()["approved"]:
                flash("Your account is awaiting admin approval.", "warning")
                return redirect(url_for("index"))
            return redirect(url_for("admin_dashboard" if role == "admin" else "live"))
    return render_template("login.html", account=None)


@app.post("/logout")
def logout():
    session.clear()
    return redirect(url_for("index"))


@app.route("/history")
def history():
    account = require_access()
    if not isinstance(account, dict): return account
    if account["role"] == "admin": return redirect(url_for("admin_dashboard"))
    with get_connection() as connection:
        rows = connection.execute('SELECT id, interpretedTexts FROM "UserHistory" WHERE userId = ? ORDER BY id DESC', (account["id"],)).fetchall()
    return render_template("history.html", account=account, history=rows)


@app.route("/admin")
def admin_dashboard():
    account = require_access(admin_only=True)
    if not isinstance(account, dict): return account
    with get_connection() as connection:
        pending = connection.execute('SELECT u.id, u.name, u.email FROM "User" u LEFT JOIN "UserApprove" p ON p.userId = u.id WHERE p.userId IS NULL ORDER BY u.id').fetchall()
        activity = connection.execute('SELECT a.action, admin.name AS admin_name, user.name AS user_name FROM "AdminActivity" a JOIN "Admin" admin ON admin.id = a.adminId JOIN "User" user ON user.id = a.userId ORDER BY a.id DESC LIMIT 20').fetchall()
    return render_template("admin.html", account=account, pending=pending, activity=activity)


@app.post("/admin/users/<int:user_id>/approve")
def approve_user(user_id):
    account = require_access(admin_only=True)
    if not isinstance(account, dict): return account
    with get_connection() as connection:
        user = connection.execute('SELECT id, name FROM "User" WHERE id = ?', (user_id,)).fetchone()
        if user:
            connection.execute('INSERT OR IGNORE INTO "UserApprove" (adminId, userId) VALUES (?, ?)', (account["id"], user_id))
            connection.execute('INSERT INTO "AdminActivity" (adminId, userId, action) VALUES (?, ?, ?)', (account["id"], user_id, f"Approved {user['name']}"))
            flash(f"Approved {user['name']}.", "success")
    return redirect(url_for("admin_dashboard"))


@app.route("/live")
def live():
    """Render the live webcam interface for prediction."""
    account = require_access()
    if not isinstance(account, dict): return account
    has_model = model_assets_ready()
    return render_template("live.html", has_model=has_model, account=account)


@app.route("/predict", methods=["POST"])
def predict():
    """Receive a webcam snapshot, detect hand landmarks, and return a class."""
    account = require_access(api=True)
    if not isinstance(account, dict): return account
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
                print("Failed to decode one of the images in the sequence. Skipping.")
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
            sequence_length=prediction_sequence_length(),
        )
    else:
        try:
            image = decode_base64_image(image_data)
        except Exception as exc:
            return jsonify({"error": f"Unable to decode image: {exc}"}), 400

        landmarks = extract_landmarks(image, hands)
        if landmarks is None:
            return jsonify({"error": "No hand landmarks detected."}), 200

        sequence = prepare_sequence(
            landmarks, sequence_length=prediction_sequence_length()
        )

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
    if is_confident and account["role"] == "user":
        record_history(account["id"], result["raw_label"])
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
