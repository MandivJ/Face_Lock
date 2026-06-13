import cv2
import numpy as np
from insightface.app import FaceAnalysis
import os

# Load model
app = FaceAnalysis(name='buffalo_l',
                   allowed_modules=['detection', 'recognition'],
                   providers=['CPUExecutionProvider'])
app.prepare(ctx_id=-1, det_size=(640, 640))

# Config

PERSON_NAME   = "mandiv"
PHOTOS_FOLDER = "register_photos/mandiv"   
MIN_CONF      = 0.7

# Process each photo

embeddings = []

for filename in os.listdir(PHOTOS_FOLDER):
    if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue

    path    = os.path.join(PHOTOS_FOLDER, filename)
    img_bgr = cv2.imread(path)

    if img_bgr is None:
        print(f"  [SKIP] Could not load: {filename}")
        continue

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    faces   = app.get(img_rgb)

    # filter confidence
    faces = [f for f in faces if f.det_score >= MIN_CONF]

    if not faces:
        print(f"  [SKIP] No valid face in: {filename}")
        continue

    if len(faces) > 1:
        print(f"  [SKIP] Multiple faces in: {filename} — use solo photos only")
        continue

    # normalize and collect
    emb = faces[0].embedding
    emb = emb / np.linalg.norm(emb)
    embeddings.append(emb)

    print(f"  [OK]   {filename}  conf={faces[0].det_score:.3f}")

# Average embeddings

if len(embeddings) < 3:
    print(f"\nNot enough valid photos ({len(embeddings)}) — need at least 3.")
    exit()

avg_emb = np.mean(embeddings, axis=0)
avg_emb = avg_emb / np.linalg.norm(avg_emb)   # re-normalize after averaging

print(f"\nProcessed : {len(embeddings)} photos")
print(f"Avg embed : shape={avg_emb.shape}, norm={np.linalg.norm(avg_emb):.6f} ✓")

# Save to disk

os.makedirs("embeddings", exist_ok=True)
save_path = f"embeddings/{PERSON_NAME}.npy"
np.save(save_path, avg_emb)

print(f"Saved     : {save_path}")