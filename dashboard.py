"""The IoT part: a mood log (CSV) and a live web dashboard served by Flask.

Privacy: only mood labels, scores and timestamps are kept. Never images or video.
"""
import csv
import logging
import os
import threading
import time

from flask import Flask, jsonify, render_template

LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mood_log.csv")
CHECK_IN_AFTER = 30     # seconds of Sad/Angry before the banner (short for the demo)
NO_FACE_AFTER = 2       # seconds without a face before we say "No face" (ignores short blinks of detection)
TIMELINE_SECONDS = 600  # the timeline shows the last 10 minutes

app = Flask(__name__)
lock = threading.Lock()  # the camera thread writes, the web thread reads
segments = []            # one [mood, start_time, last_seen_time] per unbroken stretch of a mood
last_logged = None       # last mood written to the CSV


def update(label, score, now=None):
    """Called for every frame that has a face. Only touches the disk when the mood changes."""
    global last_logged
    now = now or time.time()
    with lock:
        current = segments[-1] if segments else None
        face_was_there = current is not None and now - current[2] < NO_FACE_AFTER
        if face_was_there:
            current[2] = now
        if not face_was_there or current[0] != label:
            segments.append([label, now, now])

    if label != last_logged:
        try:
            with open(LOG_PATH, "a", newline="") as f:
                writer = csv.writer(f)
                if f.tell() == 0:
                    writer.writerow(["timestamp", "emotion", "score"])
                writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)), label, f"{score:.2f}"])
            last_logged = label
        except OSError as error:  # e.g. the file is open in Excel; keep the camera running and retry next frame
            print(f"Could not write {LOG_PATH}: {error}")


def snapshot(now=None):
    """Everything the web page needs, as plain data."""
    now = now or time.time()
    with lock:
        copy = [list(s) for s in segments]

    label, seconds = None, 0
    if copy and now - copy[-1][2] < NO_FACE_AFTER:
        label, seconds = copy[-1][0], now - copy[-1][1]

    share = {}
    for mood, start, end in copy:
        share[mood] = share.get(mood, 0) + (end - start)

    oldest = now - TIMELINE_SECONDS
    timeline = [{"mood": mood, "from": max(start, oldest) - oldest, "to": end - oldest}
                for mood, start, end in copy if end > oldest]

    return {"mood": label, "seconds": seconds, "share": share, "timeline": timeline,
            "window": TIMELINE_SECONDS,
            "check_in": label in ("Sad", "Angry") and seconds > CHECK_IN_AFTER}


@app.route("/")
def page():
    return render_template("dashboard.html")


@app.route("/api/state")
def state():
    return jsonify(snapshot())


def start(port=5000):
    """Run the web server in a background thread. daemon=True: it stops when the camera app quits."""
    logging.getLogger("werkzeug").setLevel(logging.ERROR)  # don't print a line for every poll
    # 127.0.0.1 = only this laptop can open the dashboard
    threading.Thread(target=app.run, kwargs={"host": "127.0.0.1", "port": port}, daemon=True).start()
    print(f"Dashboard: http://localhost:{port}")
