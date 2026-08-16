const camera = document.getElementById("camera");
const snapshot = document.getElementById("snapshot");
const startButton = document.getElementById("startButton");
const stopButton = document.getElementById("stopButton");
const speakButton = document.getElementById("speakButton");
const predictionElement = document.getElementById("prediction");
const confidenceElement = document.getElementById("confidence");
const topPredictionsElement = document.getElementById("topPredictions");
const statusElement = document.getElementById("statusMessage");
const overlayElement = document.getElementById("labelOverlay");
const processingIndicator = document.getElementById("processingIndicator");
const toastElement = document.getElementById("toast");
const trackingDot = document.getElementById("trackingDot");
const trackingText = document.getElementById("trackingText");

const csrfToken = CSRF_TOKEN;
const SEQUENCE_LENGTH = window.EXPECTED_SEQUENCE_LENGTH || 20;
const CAPTURE_INTERVAL_MS = 100;
const PREDICTION_INTERVAL_MS = 1000;

const frameBuffer = [];

let stream = null;
let prediction = "...";
let lastRecordedWord = null;
let captureIntervalId = null;
let predictionIntervalId = null;
let predictionInFlight = false;
let autoSpeakEnabled = false;

function showToast(message) {
  if (!toastElement) return;
  toastElement.textContent = message;
  toastElement.hidden = false;
  toastElement.classList.add("is-visible");
  window.setTimeout(() => {
    toastElement.classList.remove("is-visible");
    toastElement.hidden = true;
  }, 4000);
}

function updatePredictionUI(result) {
  const label =
    result.label && result.label !== "Unsure"
      ? result.label
      : result.top && result.top[0]
        ? result.top[0].label
        : "Unsure";

  predictionElement.textContent = label;
  confidenceElement.textContent =
    Math.round(result.confidence * 100) + "% confidence";
  confidenceElement.classList.toggle("is-low", !result.is_confident);
  overlayElement.textContent = label;
  predictionElement.classList.toggle("prediction-muted", label === "Unsure");

  if (topPredictionsElement) {
    if (result.top && result.top.length > 0) {
      topPredictionsElement.innerHTML = result.top
        .slice(0, 3)
        .map((item, index) => {
          const pct = Math.round(item.confidence * 100);
          return `<div class="top-candidate" role="listitem">
          <div class="top-candidate-meta">
            <span class="top-rank">${index + 1}</span>
            <span class="top-label">${item.label}</span>
            <span class="top-pct">${pct}%</span>
          </div>
          <span class="top-bar" aria-hidden="true"><span class="top-bar-fill" style="width: ${pct}%"></span></span>
        </div>`;
        })
        .join("");
    } else {
      topPredictionsElement.innerHTML = "";
    }
  }
}

function setTrackingStatus(validFrames, expectedFrames) {
  const ratio = validFrames / expectedFrames;
  const valid = ratio >= 0.4;
  trackingDot?.classList.toggle("is-valid", valid);
  trackingDot?.classList.toggle("is-invalid", !valid);
  if (trackingText)
    trackingText.textContent = `${validFrames}/${expectedFrames} hand frames detected`;
}

async function confirmHistory(label) {
  try {
    await fetch("/history/confirm", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-CSRF-Token": csrfToken,
      },
      body: JSON.stringify({ label }),
    });
  } catch (error) {
    console.warn("Could not save interpretation history", error);
  }
}

function speakLabel(label) {
  const audio = new Audio(`/tts?text=${encodeURIComponent(label)}`);
  audio.play().catch(() => {
    statusElement.textContent =
      "Stable sign detected. Click Speak result if audio was blocked.";
  });
}

async function startCamera() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480 },
      audio: false,
    });
    camera.srcObject = stream;
    await camera.play();

    prediction = "...";
    predictionElement.textContent = "Scanning...";
    confidenceElement.textContent = "";
    statusElement.textContent =
      "Camera active. Position your hand in frame and wait for recognition.";
    overlayElement.textContent = "Scanning for a sign";

    startPredictionLoop();
  } catch (error) {
    statusElement.textContent = `Camera error: ${error.message}`;
    overlayElement.textContent = "Camera error";
  }
}

function stopCamera() {
  if (stream) {
    stream.getTracks().forEach((track) => track.stop());
    stream = null;
  }
  clearInterval(captureIntervalId);
  clearInterval(predictionIntervalId);
  captureIntervalId = null;
  predictionIntervalId = null;
  predictionInFlight = false;
  frameBuffer.length = 0;

  prediction = "...";
  lastRecordedWord = null;
  predictionElement.textContent = "Camera stopped";
  confidenceElement.textContent = "";
  overlayElement.textContent = "Camera stopped";
  statusElement.textContent = "Press Start camera to try again.";
}

function drawFrame() {
  const ctx = snapshot.getContext("2d");
  snapshot.width = camera.videoWidth;
  snapshot.height = camera.videoHeight;
  ctx.drawImage(camera, 0, 0, snapshot.width, snapshot.height);
}

function captureFrame() {
  if (!camera || camera.readyState !== 4) {
    return;
  }

  drawFrame();
  const dataUrl = snapshot.toDataURL("image/jpeg", 0.6);
  frameBuffer.push(dataUrl);

  while (frameBuffer.length > SEQUENCE_LENGTH) {
    frameBuffer.shift();
  }
}

async function requestPrediction() {
  if (predictionInFlight || frameBuffer.length < SEQUENCE_LENGTH) {
    return;
  }

  predictionInFlight = true;
  processingIndicator.hidden = false;

  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-CSRF-Token": csrfToken,
      },
      body: JSON.stringify({ images: [...frameBuffer] }),
    });

    const data = await response.json();

    if (!response.ok || !data.success) {
      const message = data.error || "Prediction request failed.";
      statusElement.textContent = message;
      overlayElement.textContent = message;
      if (response.status === 401 || response.status === 403) {
        showToast(message);
      }
      return;
    }

    const result = data.data;
    updatePredictionUI(result);
    setTrackingStatus(result.valid_frames, SEQUENCE_LENGTH);

    if (
      result.is_confident &&
      result.label &&
      result.label !== "Unsure" &&
      result.label !== lastRecordedWord
    ) {
      lastRecordedWord = result.label;
      prediction = result.label;
      confirmHistory(prediction);
      if (autoSpeakEnabled) {
        speakLabel(prediction);
      }
      statusElement.textContent =
        "Recognized a sign from the recent frame sequence.";
    } else if (result.label && result.label !== "Unsure") {
      prediction = result.label;
      statusElement.textContent = "Sign unclear. Hold the sign steady.";
    } else {
      statusElement.textContent = "Sign unclear. Hold the sign steady.";
    }
  } catch (error) {
    statusElement.textContent = `Prediction error: ${error.message}`;
    overlayElement.textContent = "Prediction failed";
  } finally {
    predictionInFlight = false;
    processingIndicator.hidden = true;
  }
}

function startPredictionLoop() {
  clearInterval(captureIntervalId);
  clearInterval(predictionIntervalId);

  captureFrame();
  captureIntervalId = setInterval(captureFrame, CAPTURE_INTERVAL_MS);
  predictionIntervalId = setInterval(requestPrediction, PREDICTION_INTERVAL_MS);
}

function toggleAutoSpeak() {
  autoSpeakEnabled = !autoSpeakEnabled;
  speakButton.classList.toggle("is-on", autoSpeakEnabled);
  speakButton.setAttribute("aria-pressed", String(autoSpeakEnabled));

  const stateElement = speakButton.querySelector(".toggle-state");
  if (stateElement) {
    stateElement.textContent = autoSpeakEnabled ? "On" : "Off";
  }

  if (autoSpeakEnabled) {
    const text = predictionElement.textContent;
    if (
      text &&
      text !== "Waiting for camera..." &&
      text !== "Scanning..." &&
      text !== "Camera stopped" &&
      text !== "..."
    ) {
      speakLabel(text);
    }
  }
}

function speakPrediction() {
  const text = predictionElement.textContent;
  if (
    !text ||
    text === "Waiting for camera..." ||
    text === "Scanning..." ||
    text === "Camera stopped" ||
    text === "..."
  ) {
    statusElement.textContent = "No prediction yet. Start the camera first.";
    return;
  }

  speakLabel(text);
}

startButton.addEventListener("click", startCamera);
stopButton.addEventListener("click", stopCamera);
speakButton.addEventListener("click", toggleAutoSpeak);
