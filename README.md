# Signity ASL Interpreter

A Flask-based American Sign Language (ASL) interpreter built with:

- **OpenCV** for image handling
- **MediaPipe** for CPU multi-threaded hand landmark tracking
- **TensorFlow** for an NVIDIA GPU-accelerated sequential neural network (Conv1D + Bidirectional LSTMs)
- **gTTS** and **Web Speech API** for text-to-speech output

Designed to run seamlessly across OS platforms:

- **Windows 11:** CPU execution (lightweight live inference & standard dataset training).
- **Pop!\_OS / Ubuntu Linux:** CPU MediaPipe extraction + NVIDIA GPU acceleration for TensorFlow training and inference.

---

## System Requirements & Environment

### Pop!\_OS / Ubuntu Linux (Recommended for GPU Training)

- **OS:** Pop!\_OS 22.04+ or Ubuntu 20.04/22.04
- **GPU:** NVIDIA dedicated GPU (GeForce RTX / GTX series)
- **Python:** 3.10 – 3.11
- **Virtual Environment:** `tf_env` (created with `python -m venv tf_env`)

#### 1. Confirm NVIDIA Driver

If you installed the NVIDIA edition of Pop!\_OS, its driver is normally already installed. Confirm it can see your card:

```bash
nvidia-smi
```

If that command fails after a driver update or GPU error, reboot first:

```bash
sudo reboot
```

After rebooting, run `nvidia-smi` again. If it still fails, repair the Pop!\_OS driver, then reboot:

```bash
sudo apt update
sudo apt install system76-driver-nvidia
sudo reboot
```

On a laptop with switchable graphics, select **NVIDIA** or **Compute** graphics mode in the Pop!\_OS system menu and reboot. From a terminal: `sudo system76-power graphics nvidia` or `sudo system76-power graphics compute`.

#### 2. Install Python Dependencies

From the project root, with your environment activated:

```bash
source tf_env/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The `requirements.txt` includes TensorFlow with CUDA/cuDNN runtime libraries (`tensorflow[and-cuda]`), MediaPipe, OpenCV, scikit-learn, matplotlib, seaborn, tqdm, and other dependencies.

#### 3. Verify TensorFlow GPU Setup

```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

Successful output includes a `PhysicalDevice` entry for your NVIDIA GPU. The project automatically configures CUDA library paths and enables GPU memory growth on startup.

---

### Windows 11 (CPU Execution)

Windows 11 runs the project using TensorFlow on CPU. Live inference and lightweight training run smoothly on CPU.

#### 1. Create and Activate Virtual Environment

Open **PowerShell** or **Command Prompt** in the project directory:

```powershell
python -m venv tf_env
.\tf_env\Scripts\activate
```

#### 2. Install Dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Every new PowerShell session requires activating the environment first:

```powershell
.\tf_env\Scripts\activate
```

---

## Architecture Overview

Signity uses a hybrid execution model optimized for both throughput and latency:

### CPU Multi-Threaded MediaPipe Extraction

- Image datasets are processed using **MediaPipe Hands** on CPU with multi-core parallel extraction (`concurrent.futures.ProcessPoolExecutor`).
- Each worker process initializes its own MediaPipe instance to avoid thread lock contention.
- Progress is displayed in real-time using `tqdm` progress bars with images/sec speed and ETA.
- Extracted 63-dimensional hand landmark vectors are cached to `datasets/_cache/processed_asl_image_landmarks.npz` to avoid redundant processing on subsequent runs.

### NVIDIA GPU-Accelerated TensorFlow Pipeline

- Cached landmark features are fed into a sequential neural network architecture:
  - **Conv1D** layer for local temporal feature extraction
  - **Bidirectional LSTM** layers for sequence modeling
  - **Dense** output layer with softmax for multi-class classification
- Training uses `tf.data.Dataset` pipelines with prefetching and shuffling for optimal GPU throughput.
- Batch size is tuned to **80** to maximize NVIDIA GPU VRAM utilization without out-of-memory risks.
- TensorFlow automatically configures CUDA/cuDNN paths and enables GPU memory growth on startup.

---

## Dataset Layout

The project reads datasets present in these locations:

- `datasets/processed_combine_asl_dataset` – Alphabet and digit image dataset classes.
- `datasets/asl_word_dataset` – Pickle word-sequence classes.

The image dataset is converted to 63 MediaPipe hand-landmark features and cached at `datasets/_cache/processed_asl_image_landmarks.npz`. Use `python train.py --refresh-cache` after modifying the image dataset.

The word pickle samples contain 75 holistic keypoints per frame. The loader selects the hand blocks, normalizes them, and resamples each sign to a fixed 20-frame LSTM sequence. Static alphabet/digit samples use their landmark vector repeated across those 20 frames so every class shares the same numerical input shape. Each hand is wrist-relative and scale-normalized independently, with a stable left-hand-then-right-hand feature order.

## Download the Datasets

The datasets are not included directly in this repository due to size limits.

1. Download `datasets.rar` from Google Drive: [Google Drive Link](https://drive.google.com/file/d/1SSoq6TQatHlbmkETgLmsi4eCp_hFMp2D/view?usp=sharing)
2. Extract the downloaded `datasets.rar` file.
3. Place the extracted `datasets` folder into the project root directory:

```text
Signity/
├── app.py
├── train.py
├── requirements.txt
├── datasets/
│   ├── processed_combine_asl_dataset/
│   ├── asl_word_dataset/
│   └── ...
├── model/
└── ...
```

Once the `datasets` folder is in place, train the model using:

```bash
python train.py
```

Or launch the live app interpreter:

```bash
python app.py
```

---

## Dataset & Caching Workflow

### Image Dataset (`datasets/processed_combine_asl_dataset`)

- Contains alphabet (A–Z) and digit (0–9) images organized in class folders.
- Each image is processed by MediaPipe to extract 21 hand landmarks × 3 coordinates = 63 features.
- Features are wrist-relative and scale-normalized.
- Results are cached to `datasets/_cache/processed_asl_image_landmarks.npz`.

### Word Dataset (`datasets/asl_word_dataset`)

- Contains dynamic word sequences as pickle files with holistic keypoints.
- Each sample contains 75 keypoints per frame; the pipeline selects hand blocks and normalizes them.
- Sequences are resampled to a fixed 20-frame LSTM input.
- Results are cached to `datasets/_cache/asl_word_sequences.npz`.

### Cache Invalidation

Caches are automatically invalidated when source images or pickle files are newer than the cache. Use `--refresh-cache` to force re-extraction:

```bash
python train.py --refresh-cache
```

---

## Usage Commands

### Training

Standard training with default settings (100 images/class, 35 word samples/class, batch size 80, up to 80 epochs with early stopping):

```bash
python train.py
```

Full-dataset training with GPU acceleration and parallel CPU landmark extraction:

```bash
python train.py --refresh-cache --max-images-per-class 0 --max-word-samples-per-class 0 --epochs 100 --batch-size 80
```

Useful training options:

```bash
python train.py --refresh-cache
python train.py --no-words
python train.py --max-images-per-class 250
python train.py --epochs 20 --batch-size 80
python train.py --epochs 100 --patience 15
python train.py --no-early-stopping
```

Training exports:

- `model.keras` — Trained Keras LSTM model
- `labels.npy` — Label mapping dictionary
- `reports/training_evaluation/` — Accuracy/loss curves and metrics summary

### Evaluation

Evaluate the trained model on a held-out test split and generate report figures:

```bash
python evaluate.py
```

Options:

```bash
python evaluate.py --no-words
python evaluate.py --output-dir reports/evaluation
python evaluate.py --refresh-cache
```

Evaluation outputs:

- `reports/evaluation/cm_alphabet.png`
- `reports/evaluation/cm_digits.png`
- `reports/evaluation/cm_words.png`
- `reports/evaluation/top_confused_pairs.png`

### Live Web App Stream

Launch the Flask webcam UI for real-time ASL interpretation:

```bash
python app.py
```

Open `http://127.0.0.1:5500` in your browser. The web app uses GPU-accelerated MediaPipe for webcam landmark tracking and TensorFlow GPU for real-time prediction.

### Database & Admin Setup

Initialize the SQLite database and create an admin account:

```bash
python manage.py init-db
python manage.py create-admin --name "Admin Name" --email admin@example.com
```

---

## Performance Tuning Notes

### Why Batch Size 80?

- The default batch size is **80** to maximize NVIDIA GPU VRAM utilization without triggering out-of-memory errors.
- The `tf.data.Dataset` pipeline uses `.prefetch(buffer_size=tf.data.AUTOTUNE)` to overlap data preprocessing and GPU execution.
- Shuffle buffer size is set to the full training set size for optimal randomization.

### Parallel CPU Landmark Extraction

- MediaPipe landmark extraction runs on CPU using `concurrent.futures.ProcessPoolExecutor` with `max_workers=os.cpu_count()`.
- Each worker process creates its own MediaPipe Hands instance, eliminating thread lock contention.
- Images are split into batches of max 16 for efficient parallel processing.
- Progress is displayed via `tqdm` with real-time throughput (images/sec) and ETA.

### GPU Memory Growth

- TensorFlow is configured with `tf.config.experimental.set_memory_growth(gpu, True)` for all detected GPUs.
- This prevents TensorFlow from allocating all VRAM upfront and allows dynamic memory allocation during training.

---

## Project Structure

```text
signity/
├── app.py                      # Application factory, blueprint registration & server startup
├── config.py                   # Environment configuration & system thresholds
├── database.py                 # SQLite connection helpers & schema management
├── shared.py                   # Authentication & security decorators (@login_required, @csrf_required)
├── manage.py                   # Management CLI tool (creating/resetting admin accounts, database init)
├── train.py                    # Script to train the Keras LSTM model
├── evaluate.py                 # Evaluation pipeline with metrics and report generation
├── model.keras                 # Trained LSTM model file
├── labels.npy                  # Label mapping dictionary
├── signity.db                  # SQLite database file
├── cuda_config.py              # CUDA library path configuration for TensorFlow GPU
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
├── datasets/                   # ASL Datasets (Image and Word Sequence data)
│   └── _cache/                 # Cached landmark features (.npz files)
├── reports/                    # Generated training curves and evaluation figures
│   ├── training_evaluation/    # Accuracy/loss curves from train.py
│   └── evaluation/             # Confusion matrices and error analysis from evaluate.py
└── requirements.txt            # Python dependencies
```

---

## Features & Live Interpreter Workflow

### Accounts, Approval, and History

Signity stores login data in `signity.db` by default (configurable via `SIGNITY_DATABASE`). Passwords are standard hashed entries.

1. Administrators sign in at `/login/admin` to open `/admin` for user approval management.
2. Pending and rejected users cannot access the live interpreter, prediction APIs, or history pages.
3. User interaction logs and stabilized predictions are persisted to each user's `/history` page. History can be cleared anytime using **Clear history**.

### Live Interpreter Mechanics

1. Click **Start camera** and grant webcam access.
2. Keep one hand clearly in the frame.
3. Hold the sign naturally while the app collects a 20-frame sequence.
4. **Top-3 Candidate Display:** The UI continuously presents the top candidate sign along with the 2 closest runner-up options and their confidence percentages.
5. **Auto-Speak Toggle:** Toggle between `Auto-Speak: OFF` and `Auto-Speak: ON`. When set to ON, confirmed predictions speak automatically via gTTS/Web Speech API. When set to OFF, click **Speak result** to read manually.
6. A candidate appears immediately when reaching 50% confidence. Hold steady for 3 consecutive matching prediction windows to stabilize, confirm, and record to history.

### Live Inference Diagnostics

To output valid-frame counts, landmark shapes, raw probabilities, predicted labels, and execution timings to your terminal console:

**Linux / macOS:**

```bash
SIGNITY_INFERENCE_DEBUG=1 python app.py
```

**Windows (PowerShell):**

```powershell
$env:SIGNITY_INFERENCE_DEBUG="1"; python app.py
```
