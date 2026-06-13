import numpy as np

# Load saved embedding
emb = np.load("embeddings/mandiv.npy")

# Basic checks
print(f"Shape  : {emb.shape}")               # (512,)
print(f"Dtype  : {emb.dtype}")               # float32
print(f"Norm   : {np.linalg.norm(emb):.6f}") # 1.000000 ✓
print(f"Sample : {emb[:4].round(4).tolist()}")

# Confirm it's normalized 
assert emb.shape == (512,),             "❌ Wrong shape"
assert emb.dtype == np.float32,         "❌ Wrong dtype"
assert abs(np.linalg.norm(emb) - 1.0) < 1e-5, "❌ Not normalized"

print("\n✅ Embedding verified — ready for Phase 4")