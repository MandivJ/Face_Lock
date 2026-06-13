import cv2
import numpy as np
from insightface.app import FaceAnalysis
import os
import time
import csv
import subprocess
from datetime import datetime

# ── 1. Load model ──────────────────────────────────────────────────────────────
app = FaceAnalysis(name='buffalo_l',
                   allowed_modules=['detection', 'recognition'],
                   providers=['CPUExecutionProvider'])
app.prepare(ctx_id=-1, det_size=(640, 640))

# ── 2. Load known embeddings ───────────────────────────────────────────────────
EMBEDDINGS_DIR = "embeddings"
THRESHOLD      = 0.5
MIN_DET_SCORE  = 0.7
COOLDOWN_SEC   = 3.0        # seconds between repeated grants for same face
LOG_FILE       = "access_log.csv"

known = {}
for file in os.listdir(EMBEDDINGS_DIR):
    if file.endswith(".npy"):
        name = os.path.splitext(file)[0]
        known[name] = np.load(os.path.join(EMBEDDINGS_DIR, file))
        print(f"Loaded: {name}")

if not known:
    print("No embeddings found — run register.py first.")
    exit()

# ── 3. Setup CSV log ───────────────────────────────────────────────────────────
log_exists = os.path.exists(LOG_FILE)
log_file   = open(LOG_FILE, "a", newline="")
log_writer = csv.writer(log_file)

if not log_exists:
    log_writer.writerow(["timestamp", "name", "score", "result"])

def log_attempt(name, score, result):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_writer.writerow([ts, name, f"{score:.4f}", result])
    log_file.flush()
    print(f"[{ts}]  {result:7s}  {name}  score={score:.4f}")

# ── 4. Cooldown tracker ────────────────────────────────────────────────────────
last_grant_time = {}   # { "mandiv": timestamp }

def is_on_cooldown(name):
    if name not in last_grant_time:
        return False
    return (time.time() - last_grant_time[name]) < COOLDOWN_SEC

def update_cooldown(name):
    last_grant_time[name] = time.time()

# ── 5. Liveness — eye aspect ratio via kps ────────────────────────────────────
# kps layout: [right_eye, left_eye, nose, right_mouth, left_mouth]
# We track eye midpoint movement across frames to detect live motion
# A spoofed photo has static kps; a live face has slight natural movement

KPS_HISTORY_LEN = 8          # frames to track
kps_history     = []         # list of nose-tip (x,y) positions

def update_liveness(kps):
    nose = kps[2]            # nose tip is most stable landmark for motion check
    kps_history.append(nose)
    if len(kps_history) > KPS_HISTORY_LEN:
        kps_history.pop(0)

def is_live():
    if len(kps_history) < KPS_HISTORY_LEN:
        return None          # not enough data yet — neutral, don't block
    pts   = np.array(kps_history)
    spread = np.max(pts, axis=0) - np.min(pts, axis=0)
    motion = np.linalg.norm(spread)
    return motion > 1.5      # at least 1.5px natural micro-movement across 8 frames

# ── 6. OS action on grant (optional) ──────────────────────────────────────────
def on_grant(name):
    # uncomment whichever you want:
    # subprocess.run(["rundll32", "user32.dll,LockWorkStation"])  # lock PC
    # subprocess.run(["start", "", "notepad.exe"], shell=True)    # open app
    pass

# ── 7. Open webcam ─────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

print("\nFace-lock running — press 'q' to exit\n")

# ── 8. Lock loop ───────────────────────────────────────────────────────────────
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces     = app.get(frame_rgb)

    for face in faces:

        # ── det_score gate ─────────────────────────────────────────────────────
        if face.det_score < MIN_DET_SCORE:
            continue

        # ── liveness check ─────────────────────────────────────────────────────
        if face.kps is not None:
            update_liveness(face.kps)

        live   = is_live()
        spoof  = (live == False)    # False = confirmed static; None = still warming up

        # ── cosine similarity ──────────────────────────────────────────────────
        emb        = face.embedding / np.linalg.norm(face.embedding)
        best_name  = None
        best_score = -1.0

        for name, known_emb in known.items():
            score = np.dot(emb, known_emb)
            if score > best_score:
                best_score = score
                best_name  = name

        # ── access decision ────────────────────────────────────────────────────
        granted = (best_score >= THRESHOLD) and (not spoof)

        if granted:
            if not is_on_cooldown(best_name):
                update_cooldown(best_name)
                log_attempt(best_name, best_score, "GRANT")
                on_grant(best_name)
            label = f"{best_name} ({best_score:.2f}) GRANTED"
            color = (0, 255, 0)

        else:
            reason = "SPOOF?" if spoof else "DENIED"
            label  = f"Unknown ({best_score:.2f}) {reason}"
            color  = (0, 0, 255)

            if not is_on_cooldown("__denied__"):
                update_cooldown("__denied__")
                log_attempt(best_name or "unknown", best_score, reason)

        # ── draw bbox + label ──────────────────────────────────────────────────
        x1, y1, x2, y2 = face.bbox.astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # ── draw kps dots ──────────────────────────────────────────────────────
        if face.kps is not None:
            for x, y in face.kps.astype(int):
                cv2.circle(frame, (x, y), 3, (255, 255, 0), -1)

        # ── liveness warmup indicator ──────────────────────────────────────────
        if live is None:
            cv2.putText(frame, "Liveness warming up...", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    cv2.imshow("Face Lock", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
log_file.close()
print("Face-lock closed.")