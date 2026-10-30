# ===== Cellule 6 : a priori par modele, reglage par budget, export des parametres =====
import os, glob, time, json
import numpy as np, pandas as pd
import pyarrow.parquet as pq
from scipy import sparse
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

SEED, MIN_ITEMS, MIN_EVAL, N_FOLDS = 0, 80, 10, 5
BUDGETS = np.array([0, 1, 3, 7, 15, 31])
W = np.array([0.1, 0.2, 0.2, 0.2, 0.2, 0.1])     # poids exacts du Brier ALC

def find(name):
    d = os.environ.get("PREP_DIR")
    hits = glob.glob(os.path.join(d, name)) if d else []
    hits += glob.glob(f"/kaggle/input/**/{name}", recursive=True)
    hits += glob.glob(f"/kaggle/working/**/{name}", recursive=True)
    if not hits:
        raise FileNotFoundError(name + " introuvable : ajoute la sortie du notebook paiec-prep-data comme Input")
    return hits[0]

# ---------- 1. Chargement (codes entiers pour limiter la memoire) ----------
t0 = time.time()
tab = pq.read_table(find("responses_binary.parquet"),
                    read_dictionary=["folder", "subject_id", "item_id"]).unify_dictionaries()

def codes(col, keep_dict=True):
    c = tab[col]
    idx = np.concatenate([ch.indices.to_numpy(zero_copy_only=False) for ch in c.chunks]).astype(np.int64)
    return idx, (c.chunk(0).dictionary.to_pylist() if keep_dict else None)

f_idx, f_dict = codes("folder")
s_idx, s_dict = codes("subject_id")
i_idx, _ = codes("item_id", keep_dict=False)
y = tab["response"].to_numpy().astype(np.int8)
del tab
print(f"{len(y) / 1e6:.1f} M lignes chargees en {time.time() - t0:.0f}s")

# ---------- 2. Paires (modele, benchmark) et decoupage 50/50 des items ----------
S1 = len(s_dict) + 1
key = f_idx * S1 + s_idx
pair_u, p_inv = np.unique(key, return_inverse=True)
p_inv = p_inv.reshape(-1)
del key, f_idx, s_idx
P = len(pair_u)
pair_folder, pair_subj = pair_u // S1, pair_u % S1

I1 = int(i_idx.max()) + 1
iu, i_inv = np.unique(p_inv.astype(np.int64) * I1 + i_idx, return_inverse=True)
i_inv = i_inv.reshape(-1)
del i_idx
item_pair = (iu // I1).astype(np.int64)
n_items_pair = np.bincount(item_pair, minlength=P)

rng = np.random.default_rng(SEED)
is_eval = rng.random(len(iu)) < 0.5
r_ord = rng.random(len(iu))
lab = np.empty(len(iu), np.int8)
lab[i_inv[::-1]] = y[::-1]                      # premiere reponse de chaque item

row_eval = is_eval[i_inv]
n_eval = np.bincount(p_inv[row_eval], minlength=P).astype(float)
s_eval = np.bincount(p_inv[row_eval], weights=y[row_eval], minlength=P)
n_all = np.bincount(p_inv, minlength=P).astype(float)
s_all = np.bincount(p_inv, weights=y, minlength=P)
del row_eval, p_inv, i_inv, y

# 31 premiers labels d'acquisition de chaque paire, dans un ordre aleatoire fixe
acq = np.flatnonzero(~is_eval)
o = acq[np.lexsort((r_ord[acq], item_pair[acq]))]
ps = item_pair[o]
starts = np.r_[0, np.flatnonzero(np.diff(ps)) + 1]
counts = np.diff(np.r_[starts, len(ps)])
pos = np.arange(len(ps)) - np.repeat(starts, counts)
n_acq = np.zeros(P, int)
n_acq[ps[starts]] = counts
sel = pos < 31
L = np.zeros((P, 31), np.int8)
L[ps[sel], pos[sel]] = lab[o][sel]

ok = (n_items_pair >= MIN_ITEMS) & (n_acq >= 31) & (n_eval >= MIN_EVAL)
elig = np.flatnonzero(ok)
K = np.stack([L[:, :b].sum(1) for b in BUDGETS], 1).astype(float)    # succes parmi les n premiers labels
N = np.tile(BUDGETS.astype(float), (P, 1))
ybar = s_eval / np.maximum(n_eval, 1)
print(f"{P} paires au total, {len(elig)} eligibles (>= {MIN_ITEMS} items), "
      f"{len(np.unique(pair_folder[elig]))} benchmarks")

def score(Pm, idx):
    """Pm : (len(idx), 6) probabilites. Retourne le Brier par budget et le Brier ALC."""
    yb = ybar[idx][:, None]
    B = Pm ** 2 - 2 * Pm * yb + yb               # Brier moyen sur les cibles d'evaluation
    return B.mean(0), float((B * W).sum(1).mean())

# ---------- 3. Metadonnees des modeles ----------
meta = pd.read_parquet(find("subjects_meta.parquet"),
                       columns=["folder", "subject_id", "normalized_name"]).drop_duplicates(["folder", "subject_id"])
pairs = pd.DataFrame({"folder": [f_dict[i] for i in pair_folder],
                      "subject_id": [s_dict[i] for i in pair_subj]}).merge(
    meta, on=["folder", "subject_id"], how="left")
assert len(pairs) == P
skey = pairs["normalized_name"].fillna("").astype(str).str.strip().to_numpy()
print(f"paires avec un modele nomme : {(skey != '').mean():.0%}")

# ---------- 4. Modeles ----------
def fit_rasch(train, C):
    """Modele additif en logit : capacite du modele + difficulte du benchmark (un seul fit par pli)."""
    tr = train[n_all[train] >= 20]
    rate = s_all[tr] / n_all[tr]
    w = np.minimum(n_all[tr], 100.0)
    bcodes = {b: j for j, b in enumerate(np.unique(pair_folder[tr]))}
    scodes = {s: j for j, s in enumerate(sorted({skey[i] for i in tr if skey[i]}))}
    rows, cols = [], []
    for r, i in enumerate(tr):
        rows.append(r); cols.append(bcodes[pair_folder[i]])
        if skey[i]:
            rows.append(r); cols.append(len(bcodes) + scodes[skey[i]])
    X = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(tr), len(bcodes) + len(scodes)))
    X2 = sparse.vstack([X, X]).tocsr()
    y2 = np.r_[np.ones(len(tr)), np.zeros(len(tr))]
    w2 = np.r_[w * rate, w * (1 - rate)]
    clf = LogisticRegression(C=C, max_iter=300).fit(X2, y2, sample_weight=w2)
    coef = clf.coef_[0]
    ability = {s: coef[len(bcodes) + j] for s, j in scodes.items()}
    return float(clf.intercept_[0]), ability, float(rate.mean())


C_GRID, M_GRID, LAM_GRID = [0.02, 0.2], [0.5, 1, 2, 3, 4, 6, 8, 12], [0.0, 0.25, 0.5, 0.75, 1.0]
folders_el = np.unique(pair_folder[elig])
fold_of = dict(zip(folders_el, np.random.default_rng(1).permutation(len(folders_el)) % N_FOLDS))
pf_fold = np.array([fold_of.get(f, -1) for f in pair_folder])

# a priori p0 = 0,5 + lam * (sigmoid(z) - 0,5) ; prediction = (K + m * p0) / (n + m), reglages (m, lam) par budget
acc = {}
t0 = time.time()
for fold in range(N_FOLDS):
    te = elig[pf_fold[elig] == fold]
    if len(te) == 0:
        continue
    held = [f for f in folders_el if fold_of[f] == fold]
    tr = np.flatnonzero(~np.isin(pair_folder, held))
    yb = ybar[te]
    for C in C_GRID:
        icpt, ability, mu = fit_rasch(tr, C)
        pz = expit(np.array([icpt + ability.get(skey[i], 0.0) for i in te]))
        for bi, b in enumerate(BUDGETS):
            for lam in LAM_GRID:
                prior = 0.5 + lam * (pz - 0.5)
                for m in ([0.0] if b == 0 else M_GRID):
                    pm = prior if b == 0 else (K[te, bi] + m * prior) / (b + m)
                    acc.setdefault((C, bi, m, lam), []).append(pm ** 2 - 2 * pm * yb + yb)
    print(f"  pli {fold + 1}/{N_FOLDS} termine ({time.time() - t0:.0f}s)")

summary = {}
for C in C_GRID:
    pick = []
    for bi in range(6):
        cand = [(np.concatenate(v).mean(), k[2], k[3]) for k, v in acc.items() if k[0] == C and k[1] == bi]
        pick.append(min(cand))
    summary[C] = pick
    alc = sum(w * p[0] for w, p in zip(W, pick))
    print(f"\nC={C} : Brier ALC = {alc:.4f}")
    for b, (sc, m, lam) in zip(BUDGETS, pick):
        print(f"   B{b:<2d} {sc:.4f}  m={m}  lam={lam}")
best_C = min(C_GRID, key=lambda C: sum(w * p[0] for w, p in zip(W, summary[C])))
print("\nC retenu :", best_C)

# reference : memes pairs, moyenne empirique et 0,5
ref_emp = np.concatenate([np.where(N[te] > 0, K[te] / np.maximum(N[te], 1), 0.5) for te in
                          [elig[pf_fold[elig] == f] for f in range(N_FOLDS)] if len(te)])
ref_idx = np.concatenate([elig[pf_fold[elig] == f] for f in range(N_FOLDS)])
print("Reference moyenne empirique : ALC = %.4f | 0,5 constant : ALC = 0.2500" % score(ref_emp, ref_idx)[1])

# ajustement final sur toutes les donnees et export
icpt, ability, mu = fit_rasch(np.arange(P), best_C)
params = {"icpt": icpt, "ability": {k: float(v) for k, v in ability.items()},
          "budgets": [int(b) for b in BUDGETS],
          "m": [float(p[1]) for p in summary[best_C]], "lam": [float(p[2]) for p in summary[best_C]],
          "cv_brier_alc": float(sum(w * p[0] for w, p in zip(W, summary[best_C]))), "C": best_C}
out_dir = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
json.dump(params, open(os.path.join(out_dir, "model_params.json"), "w"))
print("Export : model_params.json |", len(params["ability"]), "modeles nommes | intercept %.3f" % icpt)
