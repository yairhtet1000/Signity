# Signity ASL Interpreter

A modular, Flask-based American Sign Language (ASL) interpreter using OpenCV, MediaPipe, TensorFlow (stacked LSTM), and Web Speech API / gTTS for real-time sign recognition and speech output.

---

## Modular Project Structure

```text
signity/
├── app.py                      # Application factory, blueprint registration & server startup
├── config.py                   # Environment configuration & system thresholds
├── database.py                 # SQLite connection helpers & schema management
├── shared.py                   # Authentication & security decorators (@login_required, @csrf_required)
├── manage.py                   # Management CLI tool (creating admin accounts)
├── train.py                    # Script to train the Keras LSTM model
├── model.keras                 # Trained LSTM model file
├── labels.npy                  # Label mapping dictionary
├── signity.db                  # SQLite database file
├── models/                     # Machine Learning & Computer Vision layer
│   ├── landmark_extractor.py   # MediaPipe Hand landmark tracking & processing
│   └── predictor.py            # Model loading, thread-safe inference & sequence management
├── routes/                     # Application Blueprints
│   ├── main.py                 # Public pages (Home, Dataset info)
│   ├── auth.py                 # Authentication (User/Admin sign-in, registration, logout)
│   ├── profile.py              # Profile & password management
│   ├── predict.py              # Live camera prediction, TTS, & history confirm routes
│   └── admin.py                # Admin dashboard, pending approvals, activity logging
├── static/                     # CSS, client-side JS scripts (app.js), and styling assets
├── templates/                  # Jinja2 HTML templates
└── datasets/                   # ASL Datasets (Image and Word Sequence data)
```

---

## Setup

### 1. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Configure environment variables (optional)

```bash
export SIGNITY_SECRET_KEY="replace-with-a-long-random-secret"
export SIGNITY_MIN_CONFIDENCE=0.50
export SIGNITY_MIN_MARGIN=0.02
export SIGNITY_MIN_VALID_FRAME_RATIO=0.40
export SIGNITY_INFERENCE_DEBUG=1
```

### 4. Initialize the database

```bash
python manage.py init-db
```

### 5. Create the first admin account

```bash
python manage.py create-admin --name "Admin Name" --email admin@example.com
```

### 6. Train the model (if not already trained)

```bash
python train.py
```

---

## Run

```bash
python app.py
```

Open `http://127.0.0.1:5500`.

---

## Admin CLI (`manage.py`)

```bash
# Initialize or migrate the SQLite database
python manage.py init-db

# Create an admin account
python manage.py create-admin --name "Admin Name" --email admin@example.com

# Reset the primary admin password (emergency recovery)
python manage.py reset-admin --email admin@example.com --password admin123
```

---

## Features

### Modular Flask Blueprints
- **Main**: Home page (`/`) and dataset info (`/dataset`)
- **Auth**: User registration (`/register`), user login (`/login/user`), admin login (`/login/admin`), and logout (`/logout`)
- **Profile**: Edit profile (`/profile/edit`) and change password (`/profile/password`)
- **Predict**: Live interpreter (`/live`), prediction API (`/predict`), history confirm (`/history/confirm`), history clear (`/history/clear`), history page (`/history`), and TTS (`/tts`)
- **Admin**: Dashboard (`/admin`), user approval (`/admin/users/<id>/approve`), user rejection (`/admin/users/<id>/reject`)

### Live Prediction Engine
- Frame sequences are padded or trimmed to match the model's expected input length
- Lower valid-frame threshold (`0.40`) so moving hands do not break inference
- Graceful `200 OK` responses with `{"label": "Unsure", "is_confident": false}` when frames are insufficient
- Thread-safe inference lock protects TensorFlow model predictions

### Top-3 Closest Predictions
- The live UI shows the primary recognized sign plus the top 3 closest candidates with confidence percentages
- Updates dynamically on every successful inference frame

### Auto-Speak Toggle
- The speech button toggles between `Auto-Speak: OFF` and `Auto-Speak: ON`
- When enabled, confirmed predictions are spoken automatically via gTTS
- When disabled, clicking the button speaks the current prediction once as a fallback

### History Recording
- Predicted words are stored via `/history/confirm` only after browser-side stabilization
- Users can view their interpretation history at `/history` (ordered by timestamp DESC)
- History can be cleared with the `Clear history` button

### Admin Dashboard
- Review pending user registrations
- Approve or reject user accounts
- View timestamped admin activity logs with pagination

---

## Dataset Layout

The project reads ASL datasets from the `datasets/` folder:

- `datasets/processed_combine_asl_dataset` — Alphabet and digit image classes
- `datasets/asl_word_dataset` — Word-sequence pickle classes

The image dataset is converted to 63 MediaPipe hand-landmark features and cached at `datasets/_cache/processed_asl_image_landmarks.npz`. Use `python train.py --refresh-cache` after changing the image dataset.

The word pickle samples contain 75 holistic keypoints per frame. The loader selects the hand blocks, normalizes them, and resamples each sign to a fixed 20-frame LSTM sequence. Static alphabet/digit samples use their landmark vector repeated across those 20 frames so every class has the same numerical input shape.

### Download the Datasets

The datasets are not included in this repository because of their size.

1. Download **`datasets.rar`** from Google Drive:
   https://drive.google.com/file/d/1SSoq6TQatHlbmkETgLmsi4eCp_hFMp2D/view?usp=sharing

2. Extract the downloaded `datasets.rar` file.

3. Copy the extracted **`datasets`** folder into the project root.

---

## Train

```bash
python train.py
```

Training runs for up to 80 epochs and stops early by default when validation loss no longer improves.

### Full-dataset training

```bash
python train.py --refresh-cache --max-images-per-class 0 --max-word-samples-per-class 0 --epochs 100
```

`--refresh-cache` re-extracts image landmarks. A limit of `0` means **no limit**.

---

## Accounts, Approval, and History

Signity stores login data in `signity.db` by default. Set `SIGNITY_DATABASE` to use a different SQLite database file. Passwords are stored as hashes.

1. Create the first administrator:
   ```bash
   python manage.py create-admin --name "Admin Name" --email admin@example.com
   ```

2. Sign in at `/login/admin` and open `/admin` to approve or reject registered users.

3. Pending and rejected users cannot open the live interpreter, call the prediction API, or view history.

4. Each user's stabilized interpretation is saved to their `/history` page.

---

## Use The Live Interpreter

1. Click **Start camera**.
2. Allow webcam access.
3. Keep one hand clearly in the frame.
4. Hold the sign naturally while the app collects a 20-frame sequence.
5. A candidate appears immediately when it reaches 50% confidence. Hold it steady for three matching windows to confirm and speak it.
6. Toggle **Auto-Speak: ON** to enable automatic speech output, or click the button once to speak the current prediction manually.

`gTTS` requires internet access when generating speech audio. Some browsers block automatic audio; use the manual speak button if that happens.

### Live inference diagnostics

```bash
SIGNITY_INFERENCE_DEBUG=1 python app.py
```

---

## Emergency Admin Password Reset

If the admin account is locked out, run:

```bash
python reset_admin.py
```

This resets the primary admin password to `admin123`. Delete or secure this script after recovery.
