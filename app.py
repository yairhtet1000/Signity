import os
import secrets
import sqlite3
import sys
import threading
from functools import wraps
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
from flask import (
    Flask,
    flash,
    g,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    session,
    url_for,
)
from gtts import gTTS
from werkzeug.exceptions import RequestEntityTooLarge
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
MIN_CONFIDENCE = float(os.environ.get("SIGNITY_MIN_CONFIDENCE", "0.50"))
MIN_MARGIN = float(os.environ.get("SIGNITY_MIN_MARGIN", "0.02"))
MIN_VALID_FRAME_RATIO = 0.80
MAX_SEQUENCE_FRAMES = 32
INFERENCE_DEBUG = os.environ.get("SIGNITY_INFERENCE_DEBUG") == "1"

app = Flask(__name__)
app.config.update(
    SECRET_KEY=os.environ.get("SIGNITY_SECRET_KEY", secrets.token_urlsafe(32)),
    MAX_CONTENT_LENGTH=12 * 1024 * 1024,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=os.environ.get("SIGNITY_SECURE_COOKIES") == "1",
)
initialize_database()
if INFERENCE_DEBUG:
    app.logger.setLevel("INFO")


@app.errorhandler(RequestEntityTooLarge)
def request_too_large(_error):
    if request.path == "/predict":
        return api_error("Request is too large. Send fewer or smaller frames.", 413)
    return "Request is too large.", 413


# Cache the model and labels in memory so repeated requests are fast.
model = None
label_classes = None
inference_lock = threading.Lock()


def api_response(data=None, status=200):
    return jsonify({"success": True, "data": data}), status


def api_error(message, status=400):
    return jsonify({"success": False, "data": None, "error": message}), status


def current_account():
    role, account_id = session.get("role"), session.get("account_id")
    if role not in {"admin", "user"} or not account_id:
        return None
    table = "Admin" if role == "admin" else "User"
    with get_connection() as connection:
        row = connection.execute(
            f'SELECT id, name, email FROM "{table}" WHERE id = ?', (account_id,)
        ).fetchone()
        if row is None:
            session.clear()
            return None
        status = (
            "approved"
            if role == "admin"
            else connection.execute(
                'SELECT approval_status FROM "User" WHERE id = ?', (account_id,)
            ).fetchone()["approval_status"]
        )
    return {
        **dict(row),
        "role": role,
        "approved": status == "approved",
        "approval_status": status,
    }


def _auth_failure(api, message, status):
    if api:
        return api_error(message, status)
    flash(message, "warning")
    return redirect(url_for("login"))


def login_required(api=False):
    def decorator(view):
        @wraps(view)
        def wrapped(*args, **kwargs):
            account = current_account()
            if account is None:
                return _auth_failure(api, "Sign in to continue.", 401)
            g.account = account
            return view(*args, **kwargs)

        return wrapped

    return decorator


def admin_required(api=False):
    def decorator(view):
        @login_required(api=api)
        @wraps(view)
        def wrapped(*args, **kwargs):
            if g.account["role"] != "admin":
                return _auth_failure(api, "Administrator access is required.", 403)
            return view(*args, **kwargs)

        return wrapped

    return decorator


def approved_user_required(api=False):
    def decorator(view):
        @login_required(api=api)
        @wraps(view)
        def wrapped(*args, **kwargs):
            if g.account["role"] != "admin" and not g.account["approved"]:
                return _auth_failure(
                    api, "Your account is awaiting admin approval.", 403
                )
            return view(*args, **kwargs)

        return wrapped

    return decorator


def csrf_token():
    return session.setdefault("csrf_token", secrets.token_urlsafe(24))


@app.context_processor
def inject_template_globals():
    return {"csrf_token": csrf_token}


def csrf_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if request.method in {"GET", "HEAD", "OPTIONS"}:
            return view(*args, **kwargs)
        token = request.form.get("csrf_token") or request.headers.get("X-CSRF-Token")
        if not token or not secrets.compare_digest(
            token, session.get("csrf_token", "")
        ):
            return (
                api_error("Invalid CSRF token.", 400)
                if request.is_json
                else ("Invalid CSRF token.", 400)
            )
        return view(*args, **kwargs)

    return wrapped


def record_history(user_id, interpreted_text):
    if session.get("last_interpreted_text") == interpreted_text:
        return
    with get_connection() as connection:
        connection.execute(
            'INSERT INTO "UserHistory" (userId, interpretedText) VALUES (?, ?)',
            (user_id, interpreted_text),
        )
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
                    categories = getattr(result, "handedness", []) or []
                    category = (
                        categories[index][0]
                        if index < len(categories) and categories[index]
                        else None
                    )
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
        # Inference does not need optimizer/loss state; skipping compilation
        # makes startup faster and avoids legacy training-object warnings.
        model = tf.keras.models.load_model(str(model_path), compile=False)
        if any(layer.__class__.__name__ == "Masking" for layer in model.layers):
            app.logger.warning(
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


def sequence_from_live_frames(image_sequence, expected_length):
    """Extract a reliable full-window landmark sequence from browser frames."""
    if not isinstance(image_sequence, list) or len(image_sequence) != expected_length:
        raise ValueError(f"Expected exactly {expected_length} captured frames.")

    landmarks_sequence = []
    for element in image_sequence:
        try:
            frame = decode_base64_image(element)
        except Exception:
            continue
        if frame is None:
            continue
        with inference_lock:
            landmarks = extract_landmarks(frame, hands)
        if landmarks is not None and landmarks.shape == (FEATURE_DIM,):
            landmarks_sequence.append(landmarks)

    minimum_valid_frames = max(4, int(np.ceil(expected_length * MIN_VALID_FRAME_RATIO)))
    if len(landmarks_sequence) < minimum_valid_frames:
        raise ValueError(
            f"Hand tracking found only {len(landmarks_sequence)}/{expected_length} usable frames. "
            "Keep your hand visible for the complete capture window."
        )

    sequence = prepare_sequence(
        np.stack(landmarks_sequence), sequence_length=expected_length
    )
    if sequence.shape != (expected_length, FEATURE_DIM):
        raise ValueError(f"Invalid sequence shape: {sequence.shape}.")
    return sequence, len(landmarks_sequence)


def log_inference(sequence, probabilities, predicted_label, confidence, valid_frames):
    if INFERENCE_DEBUG:
        app.logger.info(
            "inference valid_frames=%s landmarks=%s sequence=%s probabilities=%s label=%s confidence=%.4f",
            valid_frames,
            (FEATURE_DIM,),
            sequence.shape,
            np.array2string(probabilities, precision=4, threshold=probabilities.size),
            predicted_label,
            confidence,
        )


@app.route("/")
def index():
    """Render the home page with dataset summary and model readiness state."""
    summary = dataset_summary()
    has_model = model_assets_ready(summary)
    return render_template(
        "index.html", summary=summary, has_model=has_model, account=current_account()
    )


@app.route("/register", methods=["GET", "POST"])
@csrf_required
def register():
    if request.method == "POST":
        name, email, password = (
            request.form.get("name", "").strip(),
            request.form.get("email", "").strip().lower(),
            request.form.get("password", ""),
        )
        if not name or not email or len(password) < 8:
            flash(
                "Enter a name and email, and use a password with at least 8 characters.",
                "error",
            )
        else:
            try:
                with get_connection() as connection:
                    connection.execute(
                        'INSERT INTO "User" (name, email, password, is_approved, approval_status) VALUES (?, ?, ?, 0, "pending")',
                        (name, email, generate_password_hash(password)),
                    )
            except sqlite3.IntegrityError:
                flash("An account with that email already exists.", "error")
            else:
                flash(
                    "Registration received. An admin must approve your account before access is enabled.",
                    "success",
                )
                return redirect(url_for("login"))
    return render_template("register.html", account=None)


@app.route("/login", methods=["GET", "POST"])
@csrf_required
def login():
    if request.method == "POST":
        email, password = (
            request.form.get("email", "").strip().lower(),
            request.form.get("password", ""),
        )
        role = request.form.get("role", "user")
        role = role if role in {"user", "admin"} else "user"
        table = "Admin" if role == "admin" else "User"
        with get_connection() as connection:
            account = connection.execute(
                f'SELECT id, password FROM "{table}" WHERE email = ?', (email,)
            ).fetchone()
        if account is None or not check_password_hash(account["password"], password):
            flash("Invalid email, password, or account type.", "error")
        else:
            session.clear()
            session.update(account_id=account["id"], role=role)
            if role == "user" and not current_account()["approved"]:
                status = current_account()["approval_status"]
                flash(
                    f"Your account is {status}; an admin must approve it before access is enabled.",
                    "warning",
                )
                return redirect(url_for("index"))
            return redirect(url_for("admin_dashboard" if role == "admin" else "live"))
    return render_template("login.html", account=None)


@app.post("/logout")
@csrf_required
def logout():
    session.clear()
    return redirect(url_for("index"))


@app.route("/history")
@approved_user_required()
def history():
    account = g.account
    if account["role"] == "admin":
        return redirect(url_for("admin_dashboard"))
    with get_connection() as connection:
        rows = connection.execute(
            'SELECT id, interpretedText, timestamp FROM "UserHistory" WHERE userId = ? ORDER BY timestamp DESC, id DESC',
            (account["id"],),
        ).fetchall()
    return render_template("history.html", account=account, history=rows)


@app.route("/admin")
@admin_required()
def admin_dashboard():
    account = g.account
    page = max(request.args.get("page", 1, type=int), 1)
    page_size = 20
    offset = (page - 1) * page_size
    with get_connection() as connection:
        pending = connection.execute(
            'SELECT id, name, email, approval_status, created_at FROM "User" WHERE approval_status = "pending" ORDER BY created_at ASC, id ASC'
        ).fetchall()
        activity = connection.execute(
            'SELECT a.action, a.timestamp, admin.name AS admin_name, user.name AS user_name FROM "AdminActivity" a JOIN "Admin" admin ON admin.id = a.adminId JOIN "User" user ON user.id = a.userId ORDER BY a.timestamp DESC, a.id DESC LIMIT ? OFFSET ?',
            (page_size, offset),
        ).fetchall()
        activity_count = connection.execute(
            'SELECT COUNT(*) AS count FROM "AdminActivity"'
        ).fetchone()["count"]
    return render_template(
        "admin.html",
        account=account,
        pending=pending,
        activity=activity,
        page=page,
        has_next=offset + page_size < activity_count,
    )


@app.post("/admin/users/<int:user_id>/approve")
@csrf_required
@admin_required()
def approve_user(user_id):
    account = g.account
    with get_connection() as connection:
        user = connection.execute(
            'SELECT id, name FROM "User" WHERE id = ?', (user_id,)
        ).fetchone()
        if user is None:
            return ("User not found.", 404)
        connection.execute(
            'UPDATE "User" SET is_approved = 1, approval_status = "approved" WHERE id = ?',
            (user_id,),
        )
        connection.execute(
            'INSERT INTO "UserApprove" (adminId, userId, decision) VALUES (?, ?, "approved")',
            (account["id"], user_id),
        )
        connection.execute(
            'INSERT INTO "AdminActivity" (adminId, userId, action) VALUES (?, ?, ?)',
            (account["id"], user_id, f"Approved {user['name']}"),
        )
        flash(f"Approved {user['name']}.", "success")
    return redirect(url_for("admin_dashboard"))


@app.post("/admin/users/<int:user_id>/reject")
@csrf_required
@admin_required()
def reject_user(user_id):
    account = g.account
    with get_connection() as connection:
        user = connection.execute(
            'SELECT id, name FROM "User" WHERE id = ?', (user_id,)
        ).fetchone()
        if user is None:
            return ("User not found.", 404)
        connection.execute(
            'UPDATE "User" SET is_approved = 0, approval_status = "rejected" WHERE id = ?',
            (user_id,),
        )
        connection.execute(
            'INSERT INTO "UserApprove" (adminId, userId, decision) VALUES (?, ?, "rejected")',
            (account["id"], user_id),
        )
        connection.execute(
            'INSERT INTO "AdminActivity" (adminId, userId, action) VALUES (?, ?, ?)',
            (account["id"], user_id, f"Rejected {user['name']}"),
        )
        flash(f"Rejected {user['name']}.", "success")
    return redirect(url_for("admin_dashboard"))


@app.route("/live")
@approved_user_required()
def live():
    """Render the live webcam interface for prediction."""
    account = g.account
    has_model = model_assets_ready()
    return render_template("live.html", has_model=has_model, account=account)


@app.route("/predict", methods=["POST"])
@approved_user_required(api=True)
@csrf_required
def predict():
    """Receive a webcam snapshot, detect hand landmarks, and return a class."""
    account = g.account
    if not request.is_json:
        return api_error("Expected application/json body.", 400)

    payload = request.get_json(silent=True) or {}
    image_data = payload.get("image")
    image_sequence = payload.get("images")

    if not image_sequence:
        return api_error(
            "A complete frame sequence is required for live prediction.", 400
        )
    if image_data:
        return api_error(
            "Single-frame prediction is disabled for the sequence model.", 400
        )

    try:
        load_model_and_labels()
    except (FileNotFoundError, ValueError) as exc:
        app.logger.exception("Model assets are unavailable")
        return api_error(str(exc), 503)

    expected_length = prediction_sequence_length()
    if expected_length > MAX_SEQUENCE_FRAMES:
        return api_error("The loaded model has an unsupported sequence length.", 503)
    try:
        sequence, valid_frames = sequence_from_live_frames(
            image_sequence, expected_length
        )
    except ValueError as exc:
        return api_error(str(exc), 422)

    try:
        input_tensor = np.expand_dims(sequence, axis=0).astype(np.float32)
        with inference_lock:
            prediction = model.predict(input_tensor, verbose=0)[0]
    except Exception as exc:
        app.logger.exception("Prediction failed")
        return api_error("Prediction failed.", 500)
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
    meets_confidence = confidence >= MIN_CONFIDENCE
    has_clear_margin = (confidence - second_confidence) >= MIN_MARGIN

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
    log_inference(sequence, prediction, result["raw_label"], confidence, valid_frames)
    return api_response(result)


@app.post("/history/confirm")
@approved_user_required(api=True)
@csrf_required
def confirm_history():
    """Persist a label only after the browser's temporal stabilizer confirms it."""
    payload = request.get_json(silent=True) or {}
    label = str(payload.get("label", "")).strip()
    if not label:
        return api_error("A confirmed label is required.", 400)
    try:
        load_model_and_labels()
    except (FileNotFoundError, ValueError):
        return api_error("Model assets are unavailable.", 503)
    if label not in {str(value) for value in label_classes}:
        return api_error("Unknown label.", 400)
    if g.account["role"] == "user":
        record_history(g.account["id"], label)
    return api_response({"label": label})


@app.route("/tts")
@approved_user_required(api=True)
def tts():
    """Generate spoken audio for the given text query parameter."""
    text = request.args.get("text", "")
    if not text:
        return api_error("Missing text query parameter.", 400)

    tts_audio = gTTS(text=text, lang="en")
    mp3_buffer = BytesIO()
    tts_audio.write_to_fp(mp3_buffer)
    mp3_buffer.seek(0)
    return send_file(mp3_buffer, mimetype="audio/mpeg")


if __name__ == "__main__":
    app.run(
        host=os.environ.get("SIGNITY_HOST", "127.0.0.1"),
        port=int(os.environ.get("SIGNITY_PORT", "5500")),
        debug=os.environ.get("FLASK_DEBUG") == "1",
    )
