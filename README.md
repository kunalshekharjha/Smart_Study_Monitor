# Smart Study Monitor

Watches you through a webcam while you study and calls you out when you doze
off, hide your face, or pick up your phone.

There are two versions in this repo:

| | Where it runs | What it needs |
| --- | --- | --- |
| [`docs/`](docs) | Any browser, any computer | Nothing. Open a link. |
| [`app.py`](app.py) | Windows/Mac/Linux desktop | Python plus ~1.2 GB of packages |

## The web version (recommended)

Everything runs inside the browser through MediaPipe's WebAssembly build. The
camera feed never leaves the machine - no server sees a single frame.

### Publishing it so anyone can open it

The `docs/` folder is laid out for GitHub Pages:

1. Push this repo to GitHub.
2. Go to **Settings -> Pages**.
3. Under **Source**, choose `Deploy from a branch`.
4. Pick branch `main` and folder `/docs`, then **Save**.

A minute later the site is live at:

```
https://<your-username>.github.io/Smart_Study_Monitor/
```

Send that link to anyone. It works on Windows, Mac, Linux, Chromebooks and
phones. Cameras require `https://`, which GitHub Pages provides automatically.

### Running it locally

```bash
python -m http.server 8123 --directory docs
```

Then open <http://localhost:8123>. Plain `file://` will not work - the browser
blocks module and model loading from the filesystem.

### Calibrating

The eye threshold is personal - it depends on your face and how far you sit
from the camera. Watch the on-screen number with your eyes open, then closed,
and set the slider between the two values.

## The desktop version

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt   # Linux/Mac: .venv/bin/pip
.venv/Scripts/python app.py
```

Press `q` in the video window to quit. Close any other app holding the camera
first, or it will refuse to start.

Note that `ultralytics` pulls in PyTorch, which is about 500 MB on its own.
The web version does the same job with 11 MB of models.

## How detection works

- **Dozing off** - the distance between the upper and lower eyelid, divided by
  the width of the face, times 100. Below the threshold for about a second
  raises the alarm. Dividing by face width keeps it stable as you lean toward
  or away from the camera.
- **Face covered** - no face found for about a second and a half.
- **Phone** - an object detector looks for the COCO `cell phone` class above
  50% confidence. It runs every third frame, since it is the expensive half of
  the loop, and the last result is drawn on the frames in between.

## Layout

```
app.py                     desktop version
requirements.txt           desktop dependencies
yolov8n.pt                 phone detector (desktop, needs PyTorch)
alarm.mp3 faudio.mp3 paudio.mp3    sleep / face-covered / phone alarms
docs/
  index.html  style.css  app.js
  models/face_landmarker.task        face mesh
  models/efficientdet_lite0.tflite   phone detector (web)
  audio/                             same three alarms
```
