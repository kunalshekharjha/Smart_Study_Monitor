// Smart Study Monitor - browser port of app.py.
//
// Same idea as the Python version: watch the eye-to-face ratio for dozing off,
// watch for the face disappearing, and watch for a phone in frame. Everything
// runs locally through MediaPipe's WebAssembly build; no frame leaves the page.

import {
  FilesetResolver,
  FaceLandmarker,
  ObjectDetector,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/vision_bundle.mjs";

const WASM_BASE = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/wasm";

// Landmark indices, identical to the Python version.
const LEFT_EYE_TOP = 159;
const LEFT_EYE_BOTTOM = 145;
const FACE_LEFT = 130;
const FACE_RIGHT = 243;

// The Python version counted frames (15 and 20) at roughly 13 fps. A browser
// runs the loop far faster, so the same frame counts would fire in a quarter of
// the time. Thresholds are in milliseconds here to keep the real-world feel and
// stay independent of how fast the machine happens to be.
const SLEEP_MS = 1100;
const COVER_MS = 1500;

const PHONE_DETECT_EVERY = 3; // object detection is the expensive half
const PHONE_MIN_SCORE = 0.5;

const el = {
  video: document.getElementById("video"),
  canvas: document.getElementById("canvas"),
  overlay: document.getElementById("overlay"),
  status: document.getElementById("status"),
  startBtn: document.getElementById("startBtn"),
  stopBtn: document.getElementById("stopBtn"),
  controls: document.getElementById("controls"),
  thr: document.getElementById("thr"),
  thrOut: document.getElementById("thrOut"),
  mute: document.getElementById("mute"),
  fps: document.getElementById("fps"),
};

const ctx = el.canvas.getContext("2d");

const alarms = {
  cover: new Audio("audio/faudio.mp3"),
  sleep: new Audio("audio/alarm.mp3"),
  phone: new Audio("audio/paudio.mp3"),
};
for (const a of Object.values(alarms)) a.preload = "auto";

let faceLandmarker = null;
let objectDetector = null;
let stream = null;
let rafId = null;

let frameCount = 0;
let lastVideoTime = -1;
let phoneBoxes = []; // last known detections, redrawn on skipped frames
let closedSince = null;
let coveredSince = null;
let playing = null; // which alarm is currently sounding
let fpsAvg = 0;
let lastFrameAt = 0;

// ---------------------------------------------------------------- model load

async function loadModels() {
  const vision = await FilesetResolver.forVisionTasks(WASM_BASE);

  faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
    baseOptions: { modelAssetPath: "models/face_landmarker.task", delegate: "GPU" },
    runningMode: "VIDEO",
    numFaces: 1,
    outputFaceBlendshapes: false,
  });

  objectDetector = await ObjectDetector.createFromOptions(vision, {
    baseOptions: { modelAssetPath: "models/efficientdet_lite0.tflite", delegate: "GPU" },
    runningMode: "VIDEO",
    scoreThreshold: PHONE_MIN_SCORE,
    maxResults: 5,
  });
}

function setStatus(text, isError = false) {
  el.status.textContent = text;
  el.status.classList.toggle("error", isError);
}

// ---------------------------------------------------------------- camera

async function start() {
  if (!window.isSecureContext) {
    setStatus(
      "Cameras only work on an https:// address (or localhost). Open the secure link instead.",
      true,
    );
    return;
  }

  el.startBtn.disabled = true;
  setStatus("Waiting for camera permission…");

  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" },
      audio: false,
    });
  } catch (err) {
    el.startBtn.disabled = false;
    setStatus(cameraError(err), true);
    return;
  }

  el.video.srcObject = stream;
  await el.video.play();

  el.canvas.width = el.video.videoWidth;
  el.canvas.height = el.video.videoHeight;

  unlockAudio();

  el.overlay.hidden = true;
  el.controls.hidden = false;

  frameCount = 0;
  lastVideoTime = -1;
  phoneBoxes = [];
  closedSince = null;
  coveredSince = null;
  lastFrameAt = performance.now();

  rafId = requestAnimationFrame(loop);
}

function cameraError(err) {
  switch (err.name) {
    case "NotAllowedError":
      return "Camera permission was blocked. Allow it in the address bar, then press Start again.";
    case "NotFoundError":
      return "No camera found on this device.";
    case "NotReadableError":
      return "The camera is busy. Close Zoom, Teams or any other app using it and try again.";
    default:
      return "Could not open the camera: " + err.message;
  }
}

function stop() {
  if (rafId) cancelAnimationFrame(rafId);
  rafId = null;

  if (stream) for (const track of stream.getTracks()) track.stop();
  stream = null;

  for (const a of Object.values(alarms)) {
    a.pause();
    a.currentTime = 0;
  }
  playing = null;

  el.controls.hidden = true;
  el.overlay.hidden = false;
  el.startBtn.disabled = false;
  setStatus("");
}

// Browsers block audio that was not started by a user gesture. Priming each
// clip inside the Start click means the alarms can fire later on their own.
function unlockAudio() {
  for (const a of Object.values(alarms)) {
    a.play()
      .then(() => {
        a.pause();
        a.currentTime = 0;
      })
      .catch(() => {});
  }
}

// ---------------------------------------------------------------- main loop

function loop() {
  rafId = requestAnimationFrame(loop);

  const now = performance.now();
  const w = el.canvas.width;
  const h = el.canvas.height;

  // Selfie view: mirror the frame so moving right on screen matches real life.
  ctx.save();
  ctx.scale(-1, 1);
  ctx.drawImage(el.video, -w, 0, w, h);
  ctx.restore();

  // ------------------- 1. sleep & face cover -------------------
  let isSleepy = false;
  let isFaceCovered = false;

  if (el.video.currentTime !== lastVideoTime) {
    lastVideoTime = el.video.currentTime;
    const result = faceLandmarker.detectForVideo(el.video, now);
    const face = result.faceLandmarks[0];

    if (face) {
      coveredSince = null;

      const eyeDist = distance(face[LEFT_EYE_TOP], face[LEFT_EYE_BOTTOM], w, h);
      const faceDist = distance(face[FACE_LEFT], face[FACE_RIGHT], w, h);

      // faceDist can collapse to zero on a bad frame - the same division the
      // Python version used to crash on.
      if (faceDist > 0) {
        const ratio = (eyeDist / faceDist) * 100;
        const threshold = parseFloat(el.thr.value);

        if (ratio < threshold) {
          if (closedSince === null) closedSince = now;
        } else {
          closedSince = null;
        }

        label(`Eye Ratio: ${Math.round(ratio)}`, 16, 16, "#111", "#e6e9ee");
      }
    } else {
      closedSince = null;
      if (coveredSince === null) coveredSince = now;
    }
  }

  if (closedSince !== null && now - closedSince >= SLEEP_MS) isSleepy = true;
  if (coveredSince !== null && now - coveredSince >= COVER_MS) isFaceCovered = true;

  // ------------------- 2. phone detection -------------------
  if (frameCount % PHONE_DETECT_EVERY === 0) {
    const found = objectDetector.detectForVideo(el.video, now);
    phoneBoxes = [];

    for (const d of found.detections) {
      const top = d.categories[0];
      if (top && top.categoryName === "cell phone" && top.score > PHONE_MIN_SCORE) {
        phoneBoxes.push({ box: d.boundingBox, score: top.score });
      }
    }
  }
  frameCount++;

  const phoneDetected = phoneBoxes.length > 0;

  // Boxes come back in unmirrored video coordinates, so flip x to match the
  // mirrored frame on screen.
  for (const { box, score } of phoneBoxes) {
    const x = w - (box.originX + box.width);
    ctx.strokeStyle = "#ff2ed1";
    ctx.lineWidth = 3;
    ctx.strokeRect(x, box.originY, box.width, box.height);
    label(`Phone detected! ${Math.round(score * 100)}%`, x, Math.max(box.originY - 30, 4), "#ff2ed1", "#fff");
  }

  // ------------------- 3. alarms & banner -------------------
  if (isFaceCovered) {
    banner("DON'T COVER YOUR FACE!", "#d92121");
    fire("cover");
  } else if (isSleepy) {
    banner("WAKE UP & STUDY!", "#d92121");
    fire("sleep");
  } else if (phoneDetected) {
    banner("PUT THE PHONE AWAY!", "#e07b00");
    fire("phone");
  } else if (playing) {
    // Keep the message up until the clip finishes, as the Python version did.
    if (playing === "cover") banner("DON'T COVER YOUR FACE!", "#d92121");
    else if (playing === "sleep") banner("WAKE UP & STUDY!", "#d92121");
    else banner("PUT THE PHONE AWAY!", "#e07b00");
  }

  const dt = now - lastFrameAt;
  lastFrameAt = now;
  if (dt > 0) {
    fpsAvg = fpsAvg ? fpsAvg * 0.9 + (1000 / dt) * 0.1 : 1000 / dt;
    if (frameCount % 10 === 0) el.fps.textContent = `${Math.round(fpsAvg)} fps`;
  }
}

function distance(a, b, w, h) {
  // Landmarks are normalised 0-1, and x and y are divided by different numbers,
  // so scale back to pixels before measuring.
  const dx = (a.x - b.x) * w;
  const dy = (a.y - b.y) * h;
  return Math.hypot(dx, dy);
}

// ---------------------------------------------------------------- drawing

function label(text, x, y, bg, fg) {
  ctx.font = "600 20px system-ui, sans-serif";
  const pad = 8;
  const width = ctx.measureText(text).width + pad * 2;
  ctx.fillStyle = bg;
  ctx.fillRect(x, y, width, 30);
  ctx.fillStyle = fg;
  ctx.fillText(text, x + pad, y + 21);
}

function banner(text, colour) {
  ctx.font = "700 30px system-ui, sans-serif";
  const pad = 14;
  const width = ctx.measureText(text).width + pad * 2;
  const x = (el.canvas.width - width) / 2;
  const y = el.canvas.height - 80;

  ctx.fillStyle = colour;
  ctx.fillRect(x, y, width, 46);
  ctx.fillStyle = "#fff";
  ctx.fillText(text, x + pad, y + 33);
}

// ---------------------------------------------------------------- alarms

function fire(kind) {
  if (el.mute.checked) return;
  if (playing) return; // let the current clip finish, one alarm at a time

  const clip = alarms[kind];
  playing = kind;
  clip.currentTime = 0;
  clip.play().catch(() => {
    playing = null;
  });
  clip.onended = () => {
    playing = null;
  };
}

// ---------------------------------------------------------------- wiring

el.thr.addEventListener("input", () => {
  el.thrOut.textContent = el.thr.value;
});

el.startBtn.addEventListener("click", start);
el.stopBtn.addEventListener("click", stop);

loadModels()
  .then(() => {
    setStatus("");
    el.startBtn.disabled = false;
  })
  .catch((err) => {
    setStatus("Could not load the models: " + err.message, true);
  });
