const camera = document.getElementById("camera");
const snapshot = document.getElementById("snapshot");
const startButton = document.getElementById("startButton");
const stopButton = document.getElementById("stopButton");
const speakButton = document.getElementById("speakButton");
const predictionElement = document.getElementById("prediction");
const confidenceElement = document.getElementById("confidence");
const statusElement = document.getElementById("statusMessage");
const overlayElement = document.getElementById("labelOverlay");
const processingIndicator = document.getElementById("processingIndicator");
const toastElement = document.getElementById("toast");
const trackingDot = document.getElementById("trackingDot");
const trackingText = document.getElementById("trackingText");
const csrfToken = document.querySelector('meta[name="csrf-token"]')?.content;

const SEQUENCE_LENGTH = 20;
const CAPTURE_INTERVAL_MS = 125;
const PREDICTION_INTERVAL_MS = 1000;
const ACCEPT_CONFIDENCE = 0.5;
const REQUIRED_CONSECUTIVE_PREDICTIONS = 3;
const frameBuffer = [];

// The live video stream from the webcam.
let stream = null;
// The currently displayed predicted label.
let prediction = "...";
// keep an interval alive while polling the backend for predictions.
let captureIntervalId = null;
let predictionIntervalId = null;
let predictionInFlight = false;
let candidateLabel = null;
let candidateCount = 0;
let lastConfirmedLabel = null;

function showToast(message) {
  if (!toastElement) return;
  toastElement.textContent = message;
  toastElement.classList.add("is-visible");
  window.setTimeout(() => toastElement.classList.remove("is-visible"), 4000);
}

function acceptedPrediction(label, confidence) {
  if (confidence < ACCEPT_CONFIDENCE) {
    candidateLabel = null;
    candidateCount = 0;
    return false;
  }
  if (label === candidateLabel) {
    candidateCount += 1;
  } else {
    candidateLabel = label;
    candidateCount = 1;
  }
  return candidateCount >= REQUIRED_CONSECUTIVE_PREDICTIONS;
}

function setTrackingStatus(validFrames, expectedFrames) {
  const ratio = validFrames / expectedFrames;
  const valid = ratio >= 0.8;
  trackingDot?.classList.toggle("is-valid", valid);
  trackingDot?.classList.toggle("is-invalid", !valid);
  if (trackingText) trackingText.textContent = `${validFrames}/${expectedFrames} hand frames detected`;
}

async function confirmHistory(label) {
  try {
    await fetch("/history/confirm", {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-CSRF-Token": csrfToken },
      body: JSON.stringify({ label }),
    });
  } catch (error) {
    console.warn("Could not save interpretation history", error);
  }
}

function speakLabel(label) {
  const audio = new Audio(`/tts?text=${encodeURIComponent(`Detected sign ${label}`)}`);
  audio.play().catch(() => {
    statusElement.textContent = "Stable sign detected. Click Speak result if audio was blocked.";
  });
}

async function startCamera() {
  try {
    // Request camera access and display the stream in the video element.
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480 },
      audio: false,
    });
    camera.srcObject = stream;
    await camera.play();
    statusElement.textContent =
      "Camera active. Position your hand in frame and wait for recognition.";
    overlayElement.textContent = "Scanning for a digit";
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
  candidateLabel = null;
  candidateCount = 0;
  lastConfirmedLabel = null;
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
  if (frameBuffer.length > SEQUENCE_LENGTH) {
    frameBuffer.shift();
  }
}

async function requestPrediction() {
  if (predictionInFlight || frameBuffer.length < SEQUENCE_LENGTH) {
    if (frameBuffer.length < SEQUENCE_LENGTH) {
      statusElement.textContent = `Collecting sign frames... (${frameBuffer.length}/${SEQUENCE_LENGTH})`;
      overlayElement.textContent = "Hold your hand steady";
    }
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
      if (response.status === 422) setTrackingStatus(0, SEQUENCE_LENGTH);
      if (response.status === 401 || response.status === 403) {
        showToast(message);
      }
      return;
    }

    const result = data.data;
    setTrackingStatus(result.valid_frames, SEQUENCE_LENGTH);
    const fallback = result.top?.[0];
    const displayLabel = result.label && result.label !== "Unsure"
      ? result.label
      : fallback && fallback.confidence >= 0.45
        ? `${fallback.label} (${(fallback.confidence * 100).toFixed(0)}%)`
        : "Unsure";
    predictionElement.textContent = displayLabel;
    predictionElement.classList.toggle("prediction-muted", result.label === "Unsure");
    const stable = acceptedPrediction(result.raw_label, result.confidence);
    if (stable && lastConfirmedLabel !== result.raw_label) {
      prediction = result.raw_label;
      predictionElement.textContent = prediction;
      predictionElement.classList.remove("prediction-muted");
      lastConfirmedLabel = result.raw_label;
      confirmHistory(prediction);
      speakLabel(prediction);
    }
    const topText = (result.top || [])
      .map((item) => `${item.label} ${(item.confidence * 100).toFixed(0)}%`)
      .join(" | ");
    confidenceElement.textContent = stable
      ? `Confidence: ${(result.confidence * 100).toFixed(1)}%`
      : `Closest: ${topText}`;
    overlayElement.textContent = stable
      ? `Predicted: ${prediction}`
      : `Candidate: ${displayLabel} (${candidateCount}/${REQUIRED_CONSECUTIVE_PREDICTIONS})`;
    statusElement.textContent = stable
      ? "Recognized a stable sign from the recent frame sequence."
      : "Live candidate shown. Hold the sign steady to confirm it.";
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

async function speakPrediction() {
  if (!prediction || prediction === "...") {
    statusElement.textContent = "No prediction yet. Start the camera first.";
    return;
  }

  speakLabel(prediction);
}

startButton.addEventListener("click", startCamera);
stopButton.addEventListener("click", stopCamera);
speakButton.addEventListener("click", speakPrediction);
