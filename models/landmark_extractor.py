"""MediaPipe hands initialization with legacy and Tasks API fallback."""

import sys

import mediapipe as mp

from config import TASK_MODEL_PATH

hands = None
try:
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
except Exception:
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

    if not TASK_MODEL_PATH.exists():
        print(
            "MediaPipe Tasks API detected but the hand landmarker model is missing.",
            file=sys.stderr,
        )
        print("Download the model and place it at:", file=sys.stderr)
        print(str(TASK_MODEL_PATH), file=sys.stderr)
        print(
            "Suggested model URL: https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker.task",
            file=sys.stderr,
        )
        sys.exit(1)

    options = mp_tasks_vision.HandLandmarkerOptions(
        base_options=mp_base_options.BaseOptions(
            model_asset_path=str(TASK_MODEL_PATH)
        ),
        running_mode=mp_running_mode.VisionTaskRunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    tasks_landmarker = mp_tasks_vision.HandLandmarker.create_from_options(options)

    class TasksHandsWrapper:
        def __init__(self, landmarker, mp_tasks_vision):
            self.landmarker = landmarker
            self.mp_vis = mp_tasks_vision

        def process(self, image_rgb):
            mp_image = mp_image_module.Image(
                mp_image_module.ImageFormat.SRGB,
                image_rgb,
            )
            result = self.landmarker.detect(mp_image)

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
