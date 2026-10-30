# ===== Cellule 3 : echantillon de textes d'items des benchmarks binaires =====
import os, glob, time
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq

DOWNLOAD = True                               # False = donnees deja presentes dans ROOT
REPO = "aims-foundations/measurement-db"
ROOT = "/kaggle/temp/measurement-db-items"    # donnees brutes (temporaires)
OUT = "/kaggle/working/items"                 # sortie compacte
MAX_ITEMS = 3000                              # items gardes par benchmark (tires au hasard)
HEAD, TAIL = 500, 1000                        # texte long : debut + fin (la question est souvent a la fin)
SEED = 0

def find(name):
    d = os.environ.get("PREP_DIR")
    hits = glob.glob(os.path.join(d, name)) if d else []
    hits += glob.glob(f"/kaggle/input/**/{name}", recursive=True)
    hits += glob.glob(f"/kaggle/working/**/{name}", recursive=True)
    if not hits:
        raise FileNotFoundError(name + " introuvable : ajoute la sortie du notebook paiec-prep-data comme Input")
    return hits[0]

os.makedirs(ROOT, exist_ok=True)
os.makedirs(OUT, exist_ok=True)
bm = pd.read_parquet(find("benchmarks_meta.parquet"), columns=["folder", "kept_binary"])
folders = sorted(bm.loc[bm["kept_binary"], "folder"].unique())
print(len(folders), "benchmarks binaires")

# ---------- 1. Telechargement (tables items uniquement) ----------
if DOWNLOAD:
    from huggingface_hub import snapshot_download
    from kaggle_secrets import UserSecretsClient
    token = UserSecretsClient().get_secret("HF_TOKEN")
    t0 = time.time()
    snapshot_download(repo_id=REPO, repo_type="dataset", local_dir=ROOT, token=token, max_workers=16,
                      allow_patterns=[f"{f}/formatted_tables/items.parquet" for f in folders])
    print(f"Telechargement termine en {(time.time() - t0) / 60:.1f} min")

# ---------- 2. Echantillonnage ----------
COLS = ["item_id", "benchmark_id", "content", "item_features"]
rng = np.random.default_rng(SEED)
parts, report = [], []
t0 = time.time()
for i, folder in enumerate(folders, 1):
    path = os.path.join(ROOT, folder, "formatted_tables", "items.parquet")
    try:
        pf = pq.ParquetFile(path)
        n = pf.metadata.num_rows
        selected = np.ones(n, bool)
        if n > MAX_ITEMS:
            selected = np.zeros(n, bool)
            selected[rng.choice(n, MAX_ITEMS, replace=False)] = True
        got, off = [], 0
        for rg in range(pf.num_row_groups):
            t = pf.read_row_group(rg, columns=COLS)
            m = selected[off:off + len(t)]
            off += len(t)
            if m.any():
                got.append(t.filter(pa.array(m)))
        d = pa.concat_tables(got).to_pandas()
        full = d["content"].fillna("").astype(str)
        d["content_len"] = full.str.len().astype("int32")
        d["content"] = np.where(d["content_len"] > HEAD + TAIL, full.str[:HEAD] + " [...] " + full.str[-TAIL:], full)
        d["item_features"] = d["item_features"].fillna("").astype(str)
        d.insert(0, "folder", folder)
        parts.append(d)
        report.append((folder, n, len(d)))
    except Exception as e:
        report.append((folder, -1, repr(e)[:120]))
    if i % 25 == 0 or i == len(folders):
        print(f"  {i}/{len(folders)} | {sum(len(p) for p in parts) / 1e3:.0f} k items | {(time.time() - t0) / 60:.1f} min")

items = pd.concat(parts, ignore_index=True)
items.to_parquet(os.path.join(OUT, "items_sample.parquet"), index=False)

print("\n=== Resume ===")
print("Items gardes :", len(items), "| benchmarks :", items["folder"].nunique())
print("Longueur moyenne du texte gardé : %.0f caracteres | textes vides : %.1f%%"
      % (items["content"].str.len().mean(), 100 * (items["content"].str.len() == 0).mean()))
print("Fichier : %.1f Mo" % (os.path.getsize(os.path.join(OUT, "items_sample.parquet")) / 1e6))
bad = [r for r in report if r[1] == -1]
print("Echecs :", len(bad), bad[:5])
