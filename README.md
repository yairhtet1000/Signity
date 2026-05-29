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

## Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Train

```bash
python train.py
```

Useful options:

```bash
python train.py --refresh-cache
python train.py --no-words
python train.py --max-images-per-class 250
python train.py --epochs 20 --batch-size 64
```

The default training caps are tuned for a practical local run:

- 100 alphabet/digit image samples per class
- 35 word sequences per class per split
- 80 epochs with early stopping

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
