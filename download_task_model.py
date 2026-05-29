import os
import sys
from pathlib import Path
from urllib.request import urlopen

# Candidate URLs to try (some mediapipe distributions use different paths).
MODEL_URLS = [
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker.task",
    "https://storage.googleapis.com/mediapipe/hand_landmarker.task",
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker.task.bin",
    "https://storage.googleapis.com/mediapipe/models/hand_landmarker/hand_landmarker.task",
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker.task?alt=media",
    # Common path used in MediaPipe samples (float16 variant)
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task?alt=media",
]
BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "models"
OUT_FILE = OUT_DIR / "hand_landmarker.task"

OUT_DIR.mkdir(parents=True, exist_ok=True)

print(
    f"Attempting to download MediaPipe hand_landmarker task model into:\n  {OUT_FILE}\n"
)


def try_url(url):
    try:
        with urlopen(url) as resp, open(OUT_FILE, "wb") as out:
            total = int(resp.getheader("Content-Length", 0))
            downloaded = 0
            chunk_size = 8192
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                out.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded * 100 // total
                    print(f"\r{downloaded}/{total} bytes ({pct}%)", end="", flush=True)
            print()
        return True, None
    except Exception as e:
        if OUT_FILE.exists():
            try:
                OUT_FILE.unlink()
            except Exception:
                pass
        return False, e


for u in MODEL_URLS:
    print(f"Trying: {u}")
    ok, err = try_url(u)
    if ok:
        print("Download complete.")
        print(f"Saved to: {OUT_FILE}")
        sys.exit(0)
    else:
        print(f"Failed: {err}")

print(
    "All candidate URLs failed (404 or network error). Please download the correct task model manually from the MediaPipe model zoo and place it at the path above."
)
sys.exit(2)
