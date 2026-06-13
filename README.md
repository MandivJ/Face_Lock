Face-Lock is a real-time face recognition access control system built with InsightFace 
and OpenCV. It uses the buffalo_l ArcFace model to generate 512-dimensional face 
embeddings, which are compared against registered identities using cosine similarity 
to grant or deny access.

Features:
- Face registration from multiple photos per person (5–10 angles) with averaged embeddings
- Real-time webcam inference using InsightFace's buffalo_l model (CPU-only, no GPU required)
- Cosine similarity matching with configurable threshold
- Liveness detection via facial landmark motion tracking (anti-spoofing)
- Per-face cooldown timer to prevent log spam and UI flicker
- Access logging to CSV with timestamp, matched name, and similarity score
- Green/red bounding box overlay for granted/denied access

Tech Stack: Python · InsightFace · ONNX Runtime · OpenCV · NumPy
