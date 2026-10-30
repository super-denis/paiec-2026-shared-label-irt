# ===== Cellule 4 : embeddings des textes d'items (GPU) =====
import os, glob, time, subprocess, sys
import numpy as np, pandas as pd

MODEL = "intfloat/multilingual-e5-small"   # rapide, multilingue, 384 dimensions
PREFIX = "passage: "                       # e5 : "passage: " ; pour bge-m3 : ""
MAX_LEN = 256                              # longueur max en tokens
BATCH = 128
CHUNK = 20000                              # affichage de la progression par paquets
OUT = "/kaggle/working/emb"

def find(name):
    d = os.environ.get("PREP_DIR")
    hits = glob.glob(os.path.join(d, name)) if d else []
    hits += glob.glob(f"/kaggle/input/**/{name}", recursive=True)
    hits += glob.glob(f"/kaggle/working/**/{name}", recursive=True)
    if not hits:
        raise FileNotFoundError(name + " introuvable : ajoute la sortie du notebook paiec-prep-items comme Input")
    return hits[0]

os.makedirs(OUT, exist_ok=True)
import torch
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "sentence-transformers"], check=True)
    from sentence_transformers import SentenceTransformer

dev = "cuda" if torch.cuda.is_available() else "cpu"
print("Appareil :", dev, "|", torch.cuda.get_device_name(0) if dev == "cuda" else "ATTENTION : pas de GPU (active un accelerateur GPU)")
if dev == "cpu":
    raise SystemExit("Pas de GPU : Session options > Accelerator > GPU T4, puis relance.")

items = pd.read_parquet(find("items_sample.parquet"), columns=["folder", "item_id", "content", "item_features"])
feat = items["item_features"].fillna("").astype(str)
text = np.where(feat != "", feat + " || " + items["content"].fillna("").astype(str), items["content"].fillna("").astype(str))
texts = [PREFIX + t for t in text]
print(len(texts), "textes a encoder")

model = SentenceTransformer(MODEL, device=dev)
model.max_seq_length = MAX_LEN
model.half()
t0 = time.time()
parts = []
for a in range(0, len(texts), CHUNK):
    parts.append(model.encode(texts[a:a + CHUNK], batch_size=BATCH, normalize_embeddings=True,
                              convert_to_numpy=True, show_progress_bar=False))
    print(f"  {min(a + CHUNK, len(texts))}/{len(texts)} | {(time.time() - t0) / 60:.1f} min")
emb = np.vstack(parts).astype(np.float16)

np.save(os.path.join(OUT, "emb.npy"), emb)
items[["folder", "item_id"]].to_parquet(os.path.join(OUT, "emb_keys.parquet"), index=False)

# verification : les textes d'un meme benchmark doivent se ressembler plus que ceux de benchmarks differents
rng = np.random.default_rng(0)
f = items["folder"].to_numpy()
a, b = rng.integers(0, len(emb), 20000), rng.integers(0, len(emb), 20000)
cos = (emb[a].astype(np.float32) * emb[b].astype(np.float32)).sum(1)
same = f[a] == f[b]
print("\n=== Resume ===")
print("Embeddings :", emb.shape, emb.dtype, "| %.0f Mo" % (emb.nbytes / 1e6), "| duree %.1f min" % ((time.time() - t0) / 60))
print("Similarite cosinus moyenne : meme benchmark %.3f | benchmarks differents %.3f" % (cos[same].mean(), cos[~same].mean()))
print("Valeurs NaN :", int(np.isnan(emb.astype(np.float32)).sum()))
