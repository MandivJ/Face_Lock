import cv2
import numpy as np
from insightface.app import FaceAnalysis

# Load modal
app = FaceAnalysis(name='buffalo_l', allowed_modules=['detection', 'recognition'], providers=['CPUExecutionProvider'])
app.prepare(ctx_id=-1, det_size=(640, 640))

# Load image
img_path = "known_faces/img2.jpg"
img = cv2.imread(img_path)

# Downscale large images before passing to model
max_dim = 1280
h, w = img.shape[:2]
if max(h, w) > max_dim:
    scale = max_dim / max(h, w)
    img = cv2.resize(img, (int(w * scale), int(h * scale)))

if img is None:
    raise FileNotFoundError(f"Image not found at path: {img_path}")

# Run inference
faces = app.get(img)

if not faces:
    print("No faces detected! Try a clearer/closer image.")
else:
    print(f"Detected {len(faces)} face(s).")
    for i, face in enumerate(faces):
        bbox = face.bbox.astype(int)
        emb  = face.embedding / np.linalg.norm(face.embedding)
        norm = np.linalg.norm(emb)
        conf = face.det_score 

        def normalize(emb):
            return emb / np.linalg.norm(emb)

        emb_normalized = normalize(face.embedding)
        print(np.linalg.norm(emb_normalized)) 

        print(f"Face {i+1}:")
        print(f"  BBox        : {bbox.tolist()}")
        print(f"  Confidence  : {conf:.4f}")
        print(f"  Embedding   : shape={emb.shape}, dtype={emb.dtype}")
        print(f"  Emb sample  : {emb[:6].round(4).tolist()}  ...")

        # Verify embedding is nomalized
        norm = np.linalg.norm(emb)
        print(f"  L2 norm     : {norm:.6f}  {'✓ normalized' if abs(norm - 1.0) < 0.01 else '⚠ NOT normalized'}")

        # Draw bbox on image
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        label = f"conf={conf:.2f}"
        cv2.putText(img, label, (bbox[0], bbox[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
# Show result
h, w = img.shape[:2]
scale = 700 / max(h, w)
display = cv2.resize(img, (int(w * scale), int(h * scale)))
cv2.imshow("InsightFace - Detection Test", display)
cv2.waitKey(0)
cv2.destroyAllWindows()