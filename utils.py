import base64
import json
import pickle
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
DATASETS_DIR = BASE_DIR / "datasets"

IMAGE_DATA_DIR_CANDIDATES = [
    DATASETS_DIR / "processed_combine_asl_dataset",
    DATASETS_DIR / "processed_combined_asl_dataset",
]
IMAGE_DATA_DIR = next(
    (path for path in IMAGE_DATA_DIR_CANDIDATES if path.exists()),
    IMAGE_DATA_DIR_CANDIDATES[0],
)
WORD_DATA_DIR = DATASETS_DIR / "asl_word_dataset"
CACHE_DIR = DATASETS_DIR / "_cache"
IMAGE_CACHE_PATH = CACHE_DIR / "processed_asl_image_landmarks.npz"
WORD_CACHE_PATH = CACHE_DIR / "asl_word_sequences.npz"

SEQUENCE_LENGTH = 5
LANDMARK_VALUES = 3
HAND_LANDMARKS = 21
MAX_HANDS = 2
SINGLE_HAND_FEATURE_DIM = HAND_LANDMARKS * LANDMARK_VALUES
FEATURE_DIM = MAX_HANDS * SINGLE_HAND_FEATURE_DIM
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CACHE_VERSION = 3


def sort_label(label):
    """Sort digits before letters, then word labels alphabetically."""
    value = str(label)
    if value.isdigit():
        return (0, int(value))
    if len(value) == 1 and value.isalpha():
        return (1, value.lower())
    return (2, value.upper())


def normalize_hand_landmarks(frame, sort_points=False):
    """Return a wrist-relative, scale-normalized 63-value hand vector."""
    points = np.asarray(frame, dtype=np.float32).reshape(21, 3)
    points = points - np.mean(points, axis=0, keepdims=True)
    scale = np.max(np.linalg.norm(points, axis=1))
    if scale > 1e-6:
        points = points / scale
    if sort_points:
        order = np.lexsort((points[:, 0], points[:, 1]))
        points = points[order]
    return points.reshape(-1).astype(np.float32)


def normalize_hand_pair(left_hand=None, right_hand=None, sort_points=False):
    """Return a normalized 126-value left+right hand vector."""
    hands = []
    for hand in [left_hand, right_hand]:
        if hand is None:
            hands.append(np.zeros((HAND_LANDMARKS, LANDMARK_VALUES), dtype=np.float32))
        else:
            hand_points = np.asarray(hand, dtype=np.float32).reshape(
                HAND_LANDMARKS,
                LANDMARK_VALUES,
            )
            if sort_points:
                order = np.lexsort((hand_points[:, 0], hand_points[:, 1]))
                hand_points = hand_points[order]
            hands.append(hand_points)

    points = np.vstack(hands).astype(np.float32)
    active = np.linalg.norm(points, axis=1) > 1e-8
    if np.any(active):
        center = np.mean(points[active], axis=0, keepdims=True)
        points[active] = points[active] - center
        scale = np.max(np.linalg.norm(points[active], axis=1))
        if scale > 1e-6:
            points[active] = points[active] / scale

    return points.reshape(-1).astype(np.float32)


def extract_skeleton_image_landmarks(image):
    """Read a drawn MediaPipe skeleton image into a normalized point-set vector."""
    if image is None:
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=12,
        param1=50,
        param2=10,
        minRadius=4,
        maxRadius=10,
    )
    if circles is None:
        return None

    detected = np.round(circles[0]).astype(np.float32)
    detected = detected[np.argsort(-detected[:, 2])]
    centers = []
    for x, y, radius in detected:
        point = np.array([x, y], dtype=np.float32)
        if all(np.linalg.norm(point - existing) > 8 for existing in centers):
            centers.append(point)
        if len(centers) == 21:
            break

    if len(centers) < 10:
        return None

    while len(centers) < 21:
        centers.append(centers[-1].copy())

    h, w = image.shape[:2]
    points = np.zeros((21, 3), dtype=np.float32)
    points[:, 0] = np.array([point[0] / max(w, 1) for point in centers])
    points[:, 1] = np.array([point[1] / max(h, 1) for point in centers])
    return normalize_hand_pair(points, None, sort_points=True)


def dataset_summary():
    """Summarize the available alphabet/digit images and word pickle datasets."""
    image_rows = []
    image_total = 0
    if IMAGE_DATA_DIR.exists():
        for class_dir in sorted(
            [path for path in IMAGE_DATA_DIR.iterdir() if path.is_dir()],
            key=lambda p: sort_label(p.name),
        ):
            count = sum(
                1
                for path in class_dir.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            )
            image_rows.append({"label": class_dir.name, "count": count})
            image_total += count

    return {
        "image_dir": str(IMAGE_DATA_DIR.relative_to(BASE_DIR))
        if IMAGE_DATA_DIR.exists()
        else str(IMAGE_DATA_DIR),
        "rows": image_rows,
        "total": image_total,
        "words": word_dataset_summary(),
    }


def word_dataset_summary():
    """Count word classes and pickle samples under datasets/asl_word_dataset."""
    word_info = {"present": False, "sets": [], "total_samples": 0}
    if not WORD_DATA_DIR.exists():
        return word_info

    for keypoints_dir in sorted(WORD_DATA_DIR.glob("keypoints-*")):
        if not keypoints_dir.is_dir():
            continue

        for split in ["train", "test"]:
            split_dir = keypoints_dir / split
            if not split_dir.exists():
                continue

            class_dirs = [path for path in split_dir.iterdir() if path.is_dir()]
            sample_count = sum(
                len(list(class_path.glob("*.pkl"))) for class_path in class_dirs
            )
            word_info["sets"].append(
                {
                    "name": f"{keypoints_dir.name}/{split}",
                    "classes": len(class_dirs),
                    "samples": sample_count,
                }
            )
            word_info["total_samples"] += sample_count

    word_info["present"] = bool(word_info["sets"])
    return word_info


def cache_is_fresh(cache_path, source_dir, extensions=None):
    if not cache_path.exists() or not source_dir.exists():
        return False
    extensions = extensions or IMAGE_EXTENSIONS
    cache_mtime = cache_path.stat().st_mtime
    for path in source_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in extensions:
            if path.stat().st_mtime > cache_mtime:
                return False
    return True


def make_static_hands():
    try:
        return mp.solutions.hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.45,
        )
    except Exception:
        from mediapipe.tasks.python import vision as mp_tasks_vision
        from mediapipe.tasks.python.core import base_options as mp_base_options
        from mediapipe.tasks.python.vision.core import (
            image as mp_image_module,
            vision_task_running_mode as mp_running_mode,
        )

        task_path = BASE_DIR / "models" / "hand_landmarker.task"
        if not task_path.exists():
            raise FileNotFoundError(
                f"MediaPipe Tasks requires {task_path} for hand landmark extraction."
            )

        options = mp_tasks_vision.HandLandmarkerOptions(
            base_options=mp_base_options.BaseOptions(model_asset_path=str(task_path)),
            running_mode=mp_running_mode.VisionTaskRunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=0.45,
        )
        landmarker = mp_tasks_vision.HandLandmarker.create_from_options(options)

        class TasksHandsWrapper:
            def __init__(self, detector):
                self.detector = detector

            def process(self, image_rgb):
                mp_image = mp_image_module.Image(
                    mp_image_module.ImageFormat.SRGB,
                    image_rgb,
                )
                result = self.detector.detect(mp_image)

                class _Hand:
                    def __init__(self, landmarks):
                        self.landmark = landmarks

                class _Result:
                    def __init__(self, hands_list):
                        self.multi_hand_landmarks = hands_list

                hands_list = []
                for hand_landmarks in getattr(result, "hand_landmarks", []) or []:
                    hands_list.append(_Hand(hand_landmarks))
                return _Result(hands_list)

            def close(self):
                self.detector.close()

        return TasksHandsWrapper(landmarker)


def load_image_landmark_dataset(refresh_cache=False, max_images_per_class=None):
    """Load class-folder ASL images and extract/cache MediaPipe hand landmarks."""
    use_cache = max_images_per_class in (None, 0)
    if (
        use_cache
        and cache_is_fresh(IMAGE_CACHE_PATH, IMAGE_DATA_DIR)
        and not refresh_cache
    ):
        cached = np.load(IMAGE_CACHE_PATH, allow_pickle=True)
        cache_version = (
            int(cached["cache_version"]) if "cache_version" in cached.files else 0
        )
        if cache_version != CACHE_VERSION:
            return load_image_landmark_dataset(
                refresh_cache=True,
                max_images_per_class=max_images_per_class,
            )
        return cached["X"].astype(np.float32), cached["y"].astype(str)

    if not IMAGE_DATA_DIR.exists():
        return np.empty((0, FEATURE_DIM), dtype=np.float32), np.array([], dtype=str)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    labels = []
    skipped = 0

    hands = make_static_hands()
    try:
        for class_dir in sorted(
            [path for path in IMAGE_DATA_DIR.iterdir() if path.is_dir()],
            key=lambda p: sort_label(p.name),
        ):
            class_seen = 0
            for image_path in sorted(class_dir.iterdir()):
                if (
                    not image_path.is_file()
                    or image_path.suffix.lower() not in IMAGE_EXTENSIONS
                ):
                    continue
                if max_images_per_class and class_seen >= max_images_per_class:
                    break
                class_seen += 1
                image = cv2.imread(str(image_path))
                landmarks = extract_landmarks(image, hands, normalize=True)
                if landmarks is None:
                    landmarks = extract_skeleton_image_landmarks(image)
                if landmarks is None:
                    skipped += 1
                    continue
                rows.append(landmarks)
                labels.append(class_dir.name)
    finally:
        if hasattr(hands, "close"):
            hands.close()

    X = (
        np.stack(rows, axis=0).astype(np.float32)
        if rows
        else np.empty((0, FEATURE_DIM), dtype=np.float32)
    )
    y = np.array(labels, dtype=str)
    if use_cache:
        np.savez_compressed(
            IMAGE_CACHE_PATH,
            X=X,
            y=y,
            meta=json.dumps({"source": str(IMAGE_DATA_DIR), "skipped": skipped}),
            cache_version=CACHE_VERSION,
        )
    return X, y


def select_hands_from_keypoints(keypoints):
    """Extract left/right 21-landmark hands from common 75x4 holistic frames."""
    frame = np.asarray(keypoints, dtype=np.float32)
    if frame.ndim != 2 or frame.shape[0] < 21 or frame.shape[1] < 3:
        return None, None

    if frame.shape[0] >= 75:
        return frame[33:54, :3], frame[54:75, :3]
    elif frame.shape[0] >= 42:
        return frame[:21, :3], frame[21:42, :3]
    else:
        return frame[:21, :3], None


def selected_word_keypoints_dirs():
    dirs = [path for path in WORD_DATA_DIR.glob("keypoints-*") if path.is_dir()]
    if not dirs:
        return []

    def suffix_number(path):
        try:
            return int(path.name.split("-", 1)[1])
        except (IndexError, ValueError):
            return -1

    return [max(dirs, key=suffix_number)]


def load_word_landmark_sequences(
    sequence_length=SEQUENCE_LENGTH,
    refresh_cache=False,
    max_samples_per_class=None,
):
    """Load pickle word samples and convert them to fixed-length hand sequences."""
    if (
        max_samples_per_class in (None, 0)
        and cache_is_fresh(WORD_CACHE_PATH, WORD_DATA_DIR, extensions={".pkl"})
        and not refresh_cache
    ):
        cached = np.load(WORD_CACHE_PATH, allow_pickle=True)
        cache_version = (
            int(cached["cache_version"]) if "cache_version" in cached.files else 0
        )
        if (
            int(cached["sequence_length"]) == sequence_length
            and cache_version == CACHE_VERSION
        ):
            return cached["X"].astype(np.float32), cached["y"].astype(str)

    sequences = []
    labels = []
    if not WORD_DATA_DIR.exists():
        return np.empty((0, sequence_length, FEATURE_DIM), dtype=np.float32), np.array(
            [], dtype=str
        )

    for keypoints_dir in selected_word_keypoints_dirs():
        for split in ["train", "test"]:
            split_dir = keypoints_dir / split
            if not split_dir.exists():
                continue

            for class_dir in sorted(
                split_dir.iterdir(), key=lambda p: sort_label(p.name)
            ):
                if not class_dir.is_dir():
                    continue

                class_count = 0
                for pkl_path in sorted(class_dir.glob("*.pkl")):
                    if max_samples_per_class and class_count >= max_samples_per_class:
                        break
                    try:
                        with pkl_path.open("rb") as file:
                            sample = pickle.load(file)
                    except Exception:
                        continue

                    keypoints = (
                        sample.get("keypoints") if isinstance(sample, dict) else None
                    )
                    if not isinstance(keypoints, np.ndarray) or keypoints.ndim != 3:
                        continue

                    label = class_dir.name.strip()
                    file_label = (
                        sample.get("class") if isinstance(sample, dict) else None
                    )
                    if isinstance(file_label, str) and file_label.strip():
                        label = file_label.strip()

                    frame_vectors = []
                    for frame in keypoints:
                        left_hand, right_hand = select_hands_from_keypoints(frame)
                        if left_hand is not None or right_hand is not None:
                            frame_vectors.append(
                                normalize_hand_pair(left_hand, right_hand)
                            )

                    if not frame_vectors:
                        continue

                    sequence = prepare_sequence(
                        np.stack(frame_vectors, axis=0),
                        sequence_length=sequence_length,
                    )
                    sequences.append(sequence)
                    labels.append(label)
                    class_count += 1

    if not sequences:
        return np.empty((0, sequence_length, FEATURE_DIM), dtype=np.float32), np.array(
            [], dtype=str
        )

    X = np.stack(sequences, axis=0).astype(np.float32)
    y = np.array(labels, dtype=str)
    if max_samples_per_class in (None, 0):
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            WORD_CACHE_PATH,
            X=X,
            y=y,
            sequence_length=sequence_length,
            cache_version=CACHE_VERSION,
        )
    return X, y


def load_landmark_dataset(
    sequence_length=SEQUENCE_LENGTH,
    include_words=True,
    refresh_cache=False,
    max_images_per_class=None,
    max_word_samples_per_class=None,
):
    """Load alphabet/digit images plus optional word sequences for LSTM training."""
    X_frames, y_frames = load_image_landmark_dataset(
        refresh_cache=refresh_cache,
        max_images_per_class=max_images_per_class,
    )
    if X_frames.size:
        X_sequences = np.tile(X_frames[:, np.newaxis, :], (1, sequence_length, 1))
        y = y_frames.astype(str)
    else:
        X_sequences = np.empty((0, sequence_length, FEATURE_DIM), dtype=np.float32)
        y = np.array([], dtype=str)

    if include_words:
        X_words, y_words = load_word_landmark_sequences(
            sequence_length=sequence_length,
            refresh_cache=refresh_cache,
            max_samples_per_class=max_word_samples_per_class,
        )
        if X_words.size:
            X_sequences = np.vstack((X_sequences, X_words))
            y = np.concatenate((y, y_words))

    if not X_sequences.size:
        raise FileNotFoundError(
            "No usable ASL landmarks found. Check datasets/processed_combine_asl_dataset and datasets/asl_word_dataset."
        )

    return X_sequences.astype(np.float32), y.astype(str)


def extract_landmarks(image, hands, normalize=True):
    if image is None:
        return None

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if not results.multi_hand_landmarks:
        return None

    landmarks = results.multi_hand_landmarks[0]
    detected_hands = []
    for hand_landmarks in results.multi_hand_landmarks[:MAX_HANDS]:
        points = []
        for lm in hand_landmarks.landmark[:HAND_LANDMARKS]:
            points.append([lm.x, lm.y, lm.z])
        if len(points) == HAND_LANDMARKS:
            detected_hands.append(np.array(points, dtype=np.float32))

    if not detected_hands:
        return None

    detected_hands.sort(key=lambda hand: float(np.mean(hand[:, 0])))
    left_hand = detected_hands[0]
    right_hand = detected_hands[1] if len(detected_hands) > 1 else None
    vector = (
        normalize_hand_pair(left_hand, right_hand)
        if normalize
        else np.concatenate(
            [
                left_hand.reshape(-1),
                right_hand.reshape(-1)
                if right_hand is not None
                else np.zeros(SINGLE_HAND_FEATURE_DIM, dtype=np.float32),
            ]
        )
    )
    return vector.astype(np.float32)


def prepare_sequence(frame_vector, sequence_length=SEQUENCE_LENGTH):
    if frame_vector is None:
        return None

    frame_array = np.asarray(frame_vector, dtype=np.float32)
    if frame_array.ndim == 1:
        if frame_array.shape[0] != FEATURE_DIM:
            raise ValueError(
                f"Expected feature dimension {FEATURE_DIM}, got {frame_array.shape[0]}"
            )
        return np.tile(frame_array[np.newaxis, :], (sequence_length, 1))

    if frame_array.ndim == 2:
        if frame_array.shape[1] != FEATURE_DIM:
            raise ValueError(
                f"Expected feature dimension {FEATURE_DIM}, got {frame_array.shape[1]}"
            )
        if frame_array.shape[0] >= sequence_length:
            indices = np.linspace(
                0, frame_array.shape[0] - 1, sequence_length, dtype=int
            )
            return frame_array[indices]

        first_frame = frame_array[0]
        pad_count = sequence_length - frame_array.shape[0]
        padding = np.tile(first_frame[np.newaxis, :], (pad_count, 1))
        return np.vstack((padding, frame_array))

    raise ValueError(
        "prepare_sequence expects either a single frame vector or a sequence of frame vectors."
    )


def decode_base64_image(image_data):
    if "," in image_data:
        image_data = image_data.split(",", 1)[1]
    image_bytes = base64.b64decode(image_data)
    image_np = np.frombuffer(image_bytes, np.uint8)
    return cv2.imdecode(image_np, cv2.IMREAD_COLOR)
