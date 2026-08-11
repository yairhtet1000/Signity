# Signity ASL Interpreter

A Flask-based American Sign Language interpreter built with:

- OpenCV for image handling
- MediaPipe for hand landmark tracking
- TensorFlow for a stacked LSTM classifier
- gTTS for text-to-speech output

## Dataset Layout

The project now reads the datasets that are present in this repo:

- `datasets/processed_combine_asl_dataset` for alphabet and digit image classes
- `datasets/asl_word_dataset` for pickle word-sequence classes

The image dataset is converted to 63 MediaPipe hand-landmark features and cached
at `datasets/_cache/processed_asl_image_landmarks.npz`. Use
`python train.py --refresh-cache` after changing the image dataset.

The word pickle samples contain 75 holistic keypoints per frame. The loader
selects the hand blocks, normalizes them, and resamples each sign to a fixed
20-frame LSTM sequence. Static alphabet/digit samples use their landmark vector
repeated across those 20 frames so every class has the same numerical input
shape. Each hand is wrist-relative and scale-normalized independently, with a
stable left-hand-then-right-hand feature order.

## Download the Datasets

The datasets are not included in this repository because of their size.

1. Download **`datasets.rar`** from Google Drive:
   https://drive.google.com/file/d/1SSoq6TQatHlbmkETgLmsi4eCp_hFMp2D/view?usp=sharing

2. Extract the downloaded `datasets.rar` file.

3. Copy the extracted **`datasets`** folder into the project root so your directory looks like this:

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

The `datasets` folder contains the two required datasets:

- `processed_combine_asl_dataset` – Alphabet and digit image dataset
- `asl_word_dataset` – Word sequence dataset

Once the `datasets` folder is placed in the project directory, you can train the model using:

```bash
python train.py
```

or run the interpreter with an existing trained model:

```bash
python app.py
```

## Setup: Pop!_OS with an NVIDIA GPU

These instructions replace the previous Nobara-specific setup. The project
uses the NVIDIA driver supplied by Pop!_OS and installs TensorFlow's matching
CUDA and cuDNN runtime libraries inside the Python environment. You do **not**
need to install the full CUDA toolkit just to run or train this project.

### 1. Confirm the Pop!_OS NVIDIA driver works

If you installed the NVIDIA edition of Pop!_OS, its driver is normally already
installed. Confirm it can see your card:

```bash
nvidia-smi
```

If that command fails after a driver update or after a GPU error, reboot first.
This resets a loaded NVIDIA kernel driver that has stopped responding:

```bash
sudo reboot
```

After the reboot, run `nvidia-smi` again. If it still fails, repair the
Pop!_OS driver, then reboot:

```bash
sudo apt update
sudo apt install system76-driver-nvidia
sudo reboot
```

On a laptop with switchable graphics, select **NVIDIA** or **Compute** graphics
mode in the Pop!_OS system menu and reboot. From a terminal, the equivalent is
`sudo system76-power graphics nvidia` or `sudo system76-power graphics compute`.

### 2. Create the Python 3.11 TensorFlow environment

From the project root:

```bash
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.11 python3.11-venv
python3.11 -m venv tf_env
source tf_env/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install "tensorflow[and-cuda]"
```

Every new terminal needs the activation command before running the project:

```bash
source tf_env/bin/activate
```

If you previously ran `pip install tensorflow`, keep the same `tf_env` and run
only the final command above. The `[and-cuda]` extra installs CUDA and cuDNN
runtime libraries required for NVIDIA GPU use; it does not replace the Pop!_OS
NVIDIA driver.

### 3. Verify TensorFlow can use the GPU

```bash
python scripts/check_gpu.py
```

Successful output ends with at least one `PhysicalDevice` under **TensorFlow
GPU devices**. If it reports `none` and its NVIDIA driver diagnostic says it
cannot communicate with the driver, reboot before changing anything in
`tf_env`. TensorFlow cannot use the GPU until `nvidia-smi` works. If
`nvidia-smi` works but TensorFlow still reports no GPU, rerun the
`tensorflow[and-cuda]` installation command from step 2. The app and trainer
can still use the CPU, but training will be much slower.

The project automatically restarts its Python process once, when necessary, so
the CUDA libraries installed by pip are available before TensorFlow starts.
This avoids the common Linux error: `Cannot dlopen some GPU libraries`.
`scripts/use_dl_env_gpu.sh` remains available if you need to run a separate
TensorFlow command that does not use this project's Python files.

### Optional: Conda

If you prefer Conda, create and activate an environment first, then use the
same pip commands from step 2. The helper scripts no longer assume a username,
environment name, or Python 3.10 path.

## Train

```bash
python train.py
```

Training runs for up to 80 epochs and stops early by default when validation
loss no longer improves. The optimizer uses cosine learning-rate decay across
the configured training steps.

### Full-dataset training

To rebuild landmark caches and train with every available image and word
sample, activate the TensorFlow environment and run:

```bash
source tf_env/bin/activate
python train.py --refresh-cache --max-images-per-class 0 --max-word-samples-per-class 0 --epochs 100
```

`--refresh-cache` re-extracts image landmarks using the current wrist-relative
normalization. A limit of `0` means **no limit**, so both dataset arguments use
all available samples. `--epochs 100` is the maximum; early stopping remains
enabled and may complete sooner when validation loss stops improving.

Useful options:

```bash
python train.py --refresh-cache
python train.py --no-words
python train.py --max-images-per-class 250
python train.py --epochs 20 --batch-size 64
python train.py --epochs 100 --patience 15
python train.py --no-early-stopping
```

The default training caps are tuned for a practical local run:

- 100 alphabet/digit image samples per class
- 35 word sequences per class per split
- 20 landmark frames per sequence
- up to 80 epochs with early stopping enabled

Changing the sequence length and landmark normalization requires retraining.
Run `python train.py --refresh-cache` to rebuild the cached image landmarks;
the word-sequence cache also refreshes automatically. The app can load a legacy
`model.h5` and automatically supplies its 5-frame input, but its predictions
will not match the new wrist-relative preprocessing. Retrain to create a
compatible 20-frame `model.keras` before evaluating live accuracy.

Validation accuracy does not guarantee the same result from a webcam. The live
pipeline requires a 20-frame capture with at least 80% usable hand-landmark
frames. It displays a candidate at 50% confidence, then confirms it only after
three consecutive matching predictions. Confirmed signs are saved to history
and the app attempts to speak them automatically.

Training saves:

- `model.keras` (native Keras format)
- `labels.npy`

The live app will also load an existing legacy `model.h5`, but new training
runs save `model.keras` and no longer emit Keras's legacy-format warning.

## Run

```bash
python app.py
```

Open `http://127.0.0.1:5500`.

## Accounts, approval, and history

Signity stores login data in `signity.db` by default. Set `SIGNITY_DATABASE`
to use a different SQLite database file. Passwords are stored as hashes.

Create the first administrator before registering users:

```bash
python manage.py create-admin --name "Admin Name" --email admin@example.com
```

Then an administrator can sign in at `/login`, open `/admin`, and approve or
reject registered users. Pending and rejected users cannot open the live
interpreter, call the prediction API, or view history. Approval decisions and
admin actions are timestamped. Each user's stabilized interpretation is saved
to their `/history` page. For stable logins across server restarts, set a
secret:

```bash
export SIGNITY_SECRET_KEY="replace-with-a-long-random-secret"
```

## Use The Live Interpreter

1. Click **Start camera**.
2. Allow webcam access.
3. Keep one hand clearly in the frame.
4. Hold the sign naturally while the app collects a 20-frame sequence.
5. A candidate appears immediately when it reaches 50% confidence. Hold it
   steady for three matching windows to confirm and speak it.
6. Click **Speak result** to repeat the most recently confirmed sign.

`gTTS` requires internet access when generating speech audio. Some browsers
block automatic audio; use **Speak result** if that happens.

### Live inference diagnostics

To print valid-frame counts, landmark/sequence shapes, probabilities, the
predicted label, and confidence to the server console:

```bash
SIGNITY_INFERENCE_DEBUG=1 python app.py
```
