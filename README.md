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
selects the stronger hand block, normalizes it, and resamples each sign to the
fixed LSTM sequence length.

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

If that command fails, update and install Pop!_OS's NVIDIA driver, then reboot:

```bash
sudo apt update
sudo apt install system76-driver-nvidia
sudo reboot
```

On a laptop with switchable graphics, select **NVIDIA** or **Compute** graphics
mode in the Pop!_OS system menu and reboot. From a terminal, the equivalent is
`sudo system76-power graphics nvidia` or `sudo system76-power graphics compute`.

### 2. Create the project environment

From the project root:

```bash
sudo apt install python3-venv
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt "tensorflow[and-cuda]"
```

Every new terminal needs the activation command before running the project:

```bash
source .venv/bin/activate
```

### 3. Verify TensorFlow can use the GPU

```bash
python scripts/check_gpu.py
```

Successful output ends with at least one `PhysicalDevice` under **TensorFlow
GPU devices**. If it reports `none`, first make sure `nvidia-smi` succeeds, then
re-run the TensorFlow installation command from step 2. The app and trainer can
still use the CPU, but training will be much slower.

The project automatically finds the CUDA libraries installed by pip before
TensorFlow starts. `scripts/use_dl_env_gpu.sh` is only an optional manual helper
for an already-active virtual environment or Conda environment; it is not
needed for the normal Pop!_OS setup.

### Optional: Conda

If you prefer Conda, create and activate an environment first, then use the
same pip commands from step 2. The helper scripts no longer assume a username,
environment name, or Python 3.10 path.

## Train

```bash
python train.py
```

Training runs for the full 80 epochs by default. Add `--early-stopping` only
when you want validation loss to stop training early.

Useful options:

```bash
python train.py --refresh-cache
python train.py --no-words
python train.py --max-images-per-class 250
python train.py --epochs 20 --batch-size 64
python train.py --early-stopping --patience 12
```

The default training caps are tuned for a practical local run:

- 100 alphabet/digit image samples per class
- 35 word sequences per class per split
- 80 epochs unless `--early-stopping` is provided

The current checked model was fine-tuned after the main run and validates at
about 79.7% exact top-1 accuracy and 89.1% top-3 accuracy. The live app applies
a confidence gate so displayed predictions validate above 80% accuracy.

Training saves:

- `model.h5`
- `labels.npy`

## Run

```bash
python app.py
```

Open `http://127.0.0.1:5000`.

## Use The Live Interpreter

1. Click **Start camera**.
2. Allow webcam access.
3. Keep one hand clearly in the frame.
4. Hold the sign steady while the app collects a short sequence.
5. Click **Speak result** to hear the prediction.

`gTTS` requires internet access when generating speech audio.
