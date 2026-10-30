# ===== Cellule 1 : telechargement + compactage des benchmarks binaires =====
import os, sys, time, subprocess

DOWNLOAD = True                          # False = ne telecharge pas (donnees deja presentes)
REPO = "aims-foundations/measurement-db"
ROOT = "/kaggle/temp/measurement-db"     # donnees brutes (temporaires)
OUT = "/kaggle/working/compact"          # sortie compacte (gardee avec Save Version)
LIMIT = None                             # None = tout. Mettre 3 pour un test rapide.
TABLES = ("benchmarks", "subjects", "responses")

if DOWNLOAD:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "huggingface_hub"], check=False)

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

os.makedirs(ROOT, exist_ok=True)
os.makedirs(OUT, exist_ok=True)

# ---------- 1. Telechargement ----------
if DOWNLOAD:
    from huggingface_hub import HfApi, snapshot_download
    from kaggle_secrets import UserSecretsClient
    token = UserSecretsClient().get_secret("HF_TOKEN")
    api = HfApi()
    try:
        print("Connecte a Hugging Face :", api.whoami(token=token)["name"])
    except Exception as e:
        raise SystemExit("Connexion impossible. Active Internet (Session options > Internet On) "
                         "et verifie le secret HF_TOKEN.\n" + repr(e))
    if LIMIT:
        names = sorted(x.path for x in api.list_repo_tree(REPO, repo_type="dataset", token=token)
                       if "." not in x.path)[:LIMIT]
        patterns = [f"{n}/formatted_tables/{t}.parquet" for n in names for t in TABLES]
    else:
        patterns = [f"*/formatted_tables/{t}.parquet" for t in TABLES]
    t0 = time.time()
    snapshot_download(repo_id=REPO, repo_type="dataset", local_dir=ROOT,
                      allow_patterns=patterns, token=token, max_workers=16)
    print(f"Telechargement termine en {(time.time() - t0) / 60:.1f} min")

# ---------- 2. Compactage ----------
def tpath(folder, table):
    return os.path.join(ROOT, folder, "formatted_tables", table + ".parquet")

folders = sorted(d for d in os.listdir(ROOT) if os.path.exists(tpath(d, "benchmarks")))
if LIMIT:
    folders = folders[:LIMIT]
print(len(folders), "benchmarks a traiter")

schema = pa.schema([("folder", pa.string()), ("subject_id", pa.string()), ("item_id", pa.string()),
                    ("benchmark_id", pa.string()), ("trial", pa.int16()), ("response", pa.int8())])
writer = pq.ParquetWriter(os.path.join(OUT, "responses_binary.parquet"), schema, compression="zstd")

bench_parts, subj_parts, kept, skipped = [], [], [], []
rows_kept = 0
t0 = time.time()
for i, folder in enumerate(folders, 1):
    try:
        b = pq.read_table(tpath(folder, "benchmarks")).to_pandas()
        b.insert(0, "folder", folder)
        bench_parts.append(b)
        s = pq.read_table(tpath(folder, "subjects")).to_pandas()
        s.insert(0, "folder", folder)
        subj_parts.append(s)
        if not os.path.exists(tpath(folder, "responses")):
            skipped.append((folder, "pas de table responses"))
            continue
        t = pq.read_table(tpath(folder, "responses"),
                          columns=["subject_id", "item_id", "benchmark_id", "trial", "response"])
        r = pc.cast(t.column("response"), pa.float64())
        valid = pc.is_valid(r)
        is01 = pc.is_in(r, value_set=pa.array([0.0, 1.0], pa.float64()))
        n_bad = pc.sum(pc.and_(valid, pc.invert(is01))).as_py() or 0
        if n_bad > 0:
            skipped.append((folder, f"non binaire ({n_bad} valeurs hors 0/1)"))
            continue
        t = t.filter(valid)
        r = pc.filter(r, valid)
        n = t.num_rows
        if n == 0:
            skipped.append((folder, "aucune reponse valide"))
            continue
        out = pa.table({
            "folder": pa.repeat(folder, n),
            "subject_id": pc.cast(t.column("subject_id"), pa.string()),
            "item_id": pc.cast(t.column("item_id"), pa.string()),
            "benchmark_id": pc.cast(t.column("benchmark_id"), pa.string()),
            "trial": pc.cast(pc.fill_null(t.column("trial"), 0), pa.int16(), safe=False),
            "response": pc.cast(r, pa.int8()),
        }, schema=schema)
        writer.write_table(out)
        kept.append((folder, n, pc.mean(out.column("response")).as_py()))
        rows_kept += n
    except Exception as e:
        skipped.append((folder, "erreur: " + repr(e)[:150]))
    if i % 25 == 0 or i == len(folders):
        print(f"  {i}/{len(folders)} | {len(kept)} gardes | {rows_kept / 1e6:.1f} M lignes | {(time.time() - t0) / 60:.1f} min")
writer.close()

# ---------- 3. Metadonnees ----------
def _fix(v):
    if v is None or isinstance(v, str):
        return v
    if hasattr(v, "tolist"):
        return str(v.tolist())
    if isinstance(v, (list, tuple, dict)):
        return str(v)
    return v

def safe_parquet(df, path):
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].map(_fix)
    df.to_parquet(path, index=False)

kept_set = {k[0] for k in kept}
bm = pd.concat(bench_parts, ignore_index=True)
bm["kept_binary"] = bm["folder"].isin(kept_set)
safe_parquet(bm, os.path.join(OUT, "benchmarks_meta.parquet"))
sub = pd.concat(subj_parts, ignore_index=True)
safe_parquet(sub, os.path.join(OUT, "subjects_meta.parquet"))

print("\n=== Resume ===")
print("Benchmarks gardes :", len(kept), "| ecartes :", len(skipped))
print("Lignes gardees    : %.1f M | taux de reussite moyen : %.3f"
      % (rows_kept / 1e6, sum(k[1] * k[2] for k in kept) / max(rows_kept, 1)))
print("Modeles distincts :", sub[sub["folder"].isin(kept_set)]["normalized_name"].nunique())
for f in sorted(os.listdir(OUT)):
    print("  %-26s %8.1f Mo" % (f, os.path.getsize(os.path.join(OUT, f)) / 1e6))
for name, why in skipped[:15]:
    print("  ecarte:", name, "-", why)
