# Emotion IoT – Mood Monitor

Reads your facial expression from a webcam, labels your mood (Happy, Sad, Angry, Surprised or Neutral), and shows it on a live web dashboard.

**Privacy:** no images or video are saved. Only the mood label, its score and a timestamp are logged.

## How it works

1. **Camera:** OpenCV reads webcam frames (it tries cameras `0` and `1`, or a phone camera URL set in `CAM_SOURCES`).
2. **Face tracking:** MediaPipe Face Landmarker (`face_landmarker.task`) finds 478 face points and 52 expression "blendshapes" (smile, brow down, jaw open, and so on).
3. **Emotion rules:** `emotion_scores()` turns the blendshapes into a 0–1 score for each emotion. Scores are smoothed so the label doesn't flicker. If no emotion reaches 25%, the mood is Neutral.
4. **IoT dashboard:** `dashboard.py` writes every mood change to `mood_log.csv` and serves a Flask web page that updates live.

## Setup

Needs Python 3.12 and a webcam (tested on Windows).

```bash
python -m venv .venv
.venv\Scripts\activate
pip install mediapipe==0.10.14 opencv-python pillow numpy flask
```

## Run

```bash
python emotion_detector.py
```

- A window opens with the camera feed, a face mesh and a mood panel.
- Open **http://localhost:5000** for the dashboard.

| Key | Action |
|-----|--------|
| `Q` | Quit |
| `M` | Show/hide the face mesh |
| `D` | Print the strongest face movements (for tuning the rules) |

## Dashboard

- **Current mood** and how long you've been in it
- **Last 10 minutes:** a timeline of moods
- **Share of this session:** time spent in each mood
- **Check-in banner:** appears after 30 seconds of Sad or Angry

The dashboard only listens on `127.0.0.1`, so only this computer can open it.

## Files

| File | What it does |
|------|--------------|
| `emotion_detector.py` | Camera, face tracking, emotion rules, on-screen UI |
| `dashboard.py` | Mood log (CSV) and Flask web server |
| `templates/dashboard.html` | Dashboard web page |
| `face_landmarker.task` | MediaPipe face model |
| `test_emotion.py`, `test_dashboard.py` | Self-checks |
| `mood_log.csv` | Created when you run it (not in git) |

## Tests

```bash
python test_emotion.py
python test_dashboard.py
```

## Settings

Values you can change at the top of the files:

- `emotion_detector.py`: `CAM_SOURCES`, `NEUTRAL_BELOW` (Neutral threshold), `SMOOTHING`
- `dashboard.py`: `CHECK_IN_AFTER`, `NO_FACE_AFTER`, `TIMELINE_SECONDS`
