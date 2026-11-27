import os
import time
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision
from PIL import Image, ImageDraw, ImageFont

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_landmarker.task")
CAM_SOURCES = [0, 1]  # tries each until one opens; phone camera: "http://<phone-ip>:8080/video"
NEUTRAL_BELOW = 0.25  # if no emotion score reaches this, call it neutral
SMOOTHING = 0.3       # 0..1, lower = steadier label but slower to react

# --- Design: dark slate UI, violet brand, one distinct hue per emotion ---
PANEL_W = 280
BG, SURFACE, TEXT, MUTED, BRAND = "#0F172A", "#1E293B", "#F1F5F9", "#94A3B8", "#A78BFA"
COLORS = {"Happy": "#FBBF24", "Sad": "#60A5FA", "Angry": "#F87171",
          "Surprised": "#F472B6", "Neutral": "#94A3B8"}
EMOTIONS = ["Happy", "Sad", "Angry", "Surprised"]

FONT_DIR = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts")


def load_font(size, bold=False):
    names = ["Inter-SemiBold.ttf", "segoeuib.ttf"] if bold else ["Inter-Regular.ttf", "segoeui.ttf"]
    for name in names:
        try:
            return ImageFont.truetype(os.path.join(FONT_DIR, name), size)
        except OSError:
            pass
    return ImageFont.load_default(size)


FONTS = {"eyebrow": load_font(12, True), "title": load_font(22, True), "mood": load_font(40, True),
         "body": load_font(15), "small": load_font(13), "pill": load_font(18, True)}


def bgr(hex_color):
    return tuple(int(hex_color[i:i + 2], 16) for i in (5, 3, 1))


def emotion_scores(b):
    """b = dict of MediaPipe blendshape name -> 0..1. Returns emotion -> 0..1."""
    avg = lambda *names: sum(b.get(n, 0) for n in names) / len(names)
    smile = avg("mouthSmileLeft", "mouthSmileRight")
    frown = avg("mouthFrownLeft", "mouthFrownRight")
    brow_down = avg("browDownLeft", "browDownRight")
    jaw_open = b.get("jawOpen", 0)
    brow_up = avg("browInnerUp", "browOuterUpLeft", "browOuterUpRight")
    above = lambda v, rest: max(0.0, v - rest)  # ignore the small amount a relaxed face already shows
    sad_signals = (above(frown, 0.02) * 3                            # mouth corners down (weak in MediaPipe, boosted)
                   + above(b.get("mouthShrugLower", 0), 0.15) * 1.2  # lower lip pushed up, a pout
                   + above(b.get("browInnerUp", 0), 0.1) * 0.4       # worried inner brows, a helper not required
                   + above(avg("eyeLookDownLeft", "eyeLookDownRight"), 0.2) * 0.4)  # looking down
    return {
        "Happy": smile,
        "Sad": min(1.0, sad_signals) * (1 - smile) * (1 - jaw_open),
        "Angry": min(1.0, brow_down * 1.5 + avg("mouthPressLeft", "mouthPressRight") * 0.5) * (1 - smile),
        "Surprised": (jaw_open + brow_up) / 2 if jaw_open > 0.2 else 0.0,
    }


def classify(scores):
    best = max(scores, key=scores.get)
    return best if scores[best] >= NEUTRAL_BELOW else "Neutral"


def draw_mesh_glow(frame, points, color):
    """Landmark dots plus a blurred copy of them, added on top so they glow."""
    layer = np.zeros_like(frame)
    for x, y in points:
        cv2.circle(layer, (x, y), 1, color, -1, cv2.LINE_AA)
    glow = cv2.GaussianBlur(layer, (0, 0), 6)
    return cv2.add(frame, cv2.addWeighted(glow, 1.2, layer, 0.35, 0))


def draw_corners(draw, box, color, width=3):
    x1, y1, x2, y2 = box
    n = int(min(x2 - x1, y2 - y1) * 0.18)
    for cx, cy, dx, dy in [(x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)]:
        draw.line([(cx, cy), (cx + dx * n, cy)], fill=color, width=width)
        draw.line([(cx, cy), (cx, cy + dy * n)], fill=color, width=width)


def draw_pill(draw, text, center_x, bottom_y, color, min_x, max_x):
    l, t, r, b = draw.textbbox((0, 0), text, font=FONTS["pill"])
    w, h = r - l + 32, b - t + 16
    x = min(max(center_x - w // 2, min_x + 8), max_x - w - 8)
    y = max(8, bottom_y - h - 10)
    draw.rounded_rectangle([x, y, x + w, y + h], radius=h // 2, fill=color)
    draw.text((x + 16 - l, y + 8 - t), text, font=FONTS["pill"], fill=BG)


def draw_panel(draw, height, label, scores):
    draw.rectangle([0, 0, PANEL_W, height], fill=BG)
    draw.line([(PANEL_W - 1, 0), (PANEL_W - 1, height)], fill=SURFACE, width=1)
    draw.text((24, 24), "EMOTION IOT", font=FONTS["eyebrow"], fill=BRAND)
    draw.text((24, 42), "Mood Monitor", font=FONTS["title"], fill=TEXT)
    draw.line([(24, 84), (PANEL_W - 24, 84)], fill=SURFACE, width=1)

    draw.text((24, 100), "CURRENT MOOD", font=FONTS["eyebrow"], fill=MUTED)
    draw.text((24, 116), label or "No face", font=FONTS["mood"], fill=COLORS.get(label, MUTED))

    bar_w, bar_h = PANEL_W - 48, 8
    for i, name in enumerate(EMOTIONS):
        y = 186 + i * 46
        score = scores.get(name, 0.0)
        draw.text((24, y), name, font=FONTS["body"], fill=TEXT)
        pct = f"{round(score * 100)}%"
        draw.text((PANEL_W - 24 - draw.textlength(pct, font=FONTS["body"]), y), pct, font=FONTS["body"], fill=MUTED)
        draw.rounded_rectangle([24, y + 24, 24 + bar_w, y + 24 + bar_h], radius=bar_h // 2, fill=SURFACE)
        fill_w = int(bar_w * score)
        if fill_w >= bar_h:
            draw.rounded_rectangle([24, y + 24, 24 + fill_w, y + 24 + bar_h], radius=bar_h // 2, fill=COLORS[name])

    if height - 32 > y + 24 + bar_h + 16:  # only when the frame is tall enough not to overlap the bars
        draw.text((24, height - 32), "Q quit    M mesh    D debug", font=FONTS["small"], fill=MUTED)


def render(frame, points, label, scores, show_mesh):
    """frame = mirrored BGR camera image; points = face landmark pixels or None. Returns the BGR window image."""
    h, w = frame.shape[:2]
    if points and show_mesh:
        frame = draw_mesh_glow(frame, points, bgr(COLORS[label]))

    canvas = Image.new("RGB", (PANEL_W + w, h), BG)
    canvas.paste(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), (PANEL_W, 0))
    draw = ImageDraw.Draw(canvas)
    draw_panel(draw, h, label if points else None, scores if points else {})

    if points:
        xs, ys = [p[0] + PANEL_W for p in points], [p[1] for p in points]
        box = (min(xs) - 12, min(ys) - 12, max(xs) + 12, max(ys) + 12)
        draw_corners(draw, box, COLORS[label])
        draw_pill(draw, label, (box[0] + box[2]) // 2, box[1], COLORS[label], PANEL_W, PANEL_W + w)

    return cv2.cvtColor(np.array(canvas), cv2.COLOR_RGB2BGR)


def open_camera():
    for src in CAM_SOURCES:
        cap = cv2.VideoCapture(src, cv2.CAP_DSHOW) if isinstance(src, int) else cv2.VideoCapture(src)
        if cap.isOpened() and cap.read()[0]:
            print(f"Using camera {src}")
            return cap
        cap.release()
    return None


def create_landmarker(running_mode=vision.RunningMode.VIDEO):
    return vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        # buffer, not path: mediapipe on Windows mangles absolute model paths
        base_options=BaseOptions(model_asset_buffer=open(MODEL_PATH, "rb").read()),
        running_mode=running_mode,
        output_face_blendshapes=True,
        num_faces=1,
    ))


def main():
    cap = open_camera()
    if cap is None:
        print("Error: no camera found. Check CAM_SOURCES at the top of the file.")
        return

    landmarker = create_landmarker()
    smoothed, label, show_mesh = {}, "Neutral", True
    start = time.time()
    print("Press Q to quit, M to toggle the face mesh.")
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Camera stopped sending frames.")
            break
        frame = cv2.flip(frame, 1)  # mirror, feels natural on a webcam
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = landmarker.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), int((time.time() - start) * 1000))

        points = None
        if result.face_blendshapes:
            shapes = {c.category_name: c.score for c in result.face_blendshapes[0]}
            raw = emotion_scores(shapes)
            smoothed = {k: SMOOTHING * v + (1 - SMOOTHING) * smoothed.get(k, v) for k, v in raw.items()}
            label = classify(smoothed)
            points = [(int(p.x * w), int(p.y * h)) for p in result.face_landmarks[0]]

        cv2.imshow("Emotion IoT - Mood Monitor", render(frame, points, label, smoothed, show_mesh))
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("m"):
            show_mesh = not show_mesh
        if key == ord("d") and points:  # tuning aid: print the strongest face movements right now
            top = sorted(shapes.items(), key=lambda kv: -kv[1])[:12]
            print(f"[{label}] " + ", ".join(f"{k} {v:.2f}" for k, v in top if v > 0.05))

    landmarker.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
