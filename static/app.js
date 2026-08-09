const camera = document.getElementById("camera");
const snapshot = document.getElementById("snapshot");
const startButton = document.getElementById("startButton");
const stopButton = document.getElementById("stopButton");
const speakButton = document.getElementById("speakButton");
const predictionElement = document.getElementById("prediction");
const confidenceElement = document.getElementById("confidence");
const statusElement = document.getElementById("statusMessage");
const overlayElement = document.getElementById("labelOverlay");

const SEQUENCE_LENGTH = 20;
const CAPTURE_INTERVAL_MS = 125;
const PREDICTION_INTERVAL_MS = 500;
const frameBuffer = [];
const predictionHistory = [];

// The live video stream from the webcam.
let stream = null;
// The currently displayed predicted label.
let prediction = "...";
// keep an interval alive while polling the backend for predictions.
let captureIntervalId = null;
let predictionIntervalId = null;
let predictionInFlight = false;

function stableLabel(nextLabel) {
  predictionHistory.push(nextLabel);
  if (predictionHistory.length > 4) {
    predictionHistory.shift();
  }

  const counts = predictionHistory.reduce((acc, label) => {
    acc[label] = (acc[label] || 0) + 1;
    return acc;
  }, {});
  return Object.entries(counts).sort((a, b) => b[1] - a[1])[0][0];
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
  predictionHistory.length = 0;
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
  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ images: frameBuffer }),
    });
    const data = await response.json();

    if (data.error) {
      statusElement.textContent = data.error;
      overlayElement.textContent = data.error;
      return;
    }

    prediction = stableLabel(data.label);
    predictionElement.textContent = prediction;
    const topText = (data.top || [])
      .map((item) => `${item.label} ${(item.confidence * 100).toFixed(0)}%`)
      .join(" | ");
    confidenceElement.textContent = data.is_confident
      ? `Confidence: ${(data.confidence * 100).toFixed(1)}%`
      : `Closest: ${topText}`;
    overlayElement.textContent = data.is_confident
      ? `Predicted: ${prediction}`
      : "Hold sign steady";
    statusElement.textContent = data.is_confident
      ? "Recognized a stable sign from the recent frame sequence."
      : "Prediction is not confident yet. Keep your hand visible and steady.";
  } catch (error) {
    statusElement.textContent = `Prediction error: ${error.message}`;
    overlayElement.textContent = "Prediction failed";
  } finally {
    predictionInFlight = false;
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

  const audio = new Audio(
    `/tts?text=${encodeURIComponent(`Detected sign ${prediction}`)}`,
  );
  audio.play().catch((error) => {
    statusElement.textContent = `Audio playback failed: ${error.message}`;
  });
}

startButton.addEventListener("click", startCamera);
stopButton.addEventListener("click", stopCamera);
speakButton.addEventListener("click", speakPrediction);
