# ===== Cellule 7 : labels partages entre modeles d'un meme benchmark (simulation) =====
import os, glob, time
import numpy as np, pandas as pd
import pyarrow.parquet as pq
from scipy.special import expit

SEED, N_FOLDS, MIN_EVAL = 0, 5, 10
MIN_ITEMS = int(os.environ.get("MIN_ITEMS", 80))
EV_CAP = int(os.environ.get("EV_CAP", 200))
BUDGETS = np.array([0, 1, 3, 7, 15, 31])
W = np.array([0.1, 0.2, 0.2, 0.2, 0.2, 0.1])
S_GRID = [3, 10, 30, 10 ** 6]                  # nombre de paires (modeles) dont les labels sont visibles par benchmark
CFG = [(1.0, 0.7), (1.0, 1.5), (2.0, 0.7), (2.0, 1.5)]     # (tau_modele, tau_item)

def find(name):
    d = os.environ.get("PREP_DIR")
    hits = glob.glob(os.path.join(d, name)) if d else []
    hits += glob.glob(f"/kaggle/input/**/{name}", recursive=True)
    hits += glob.glob(f"/kaggle/working/**/{name}", recursive=True)
    if not hits:
        raise FileNotFoundError(name + " introuvable : ajoute la sortie du notebook paiec-prep-data comme Input")
    return hits[0]

# ---------- 1. Chargement ----------
t0 = time.time()
tab = pq.read_table(find("responses_binary.parquet"),
                    read_dictionary=["folder", "subject_id", "item_id"]).unify_dictionaries()
def codes(col):
    c = tab[col]
    return np.concatenate([ch.indices.to_numpy(zero_copy_only=False) for ch in c.chunks]).astype(np.int64)
f_idx, s_idx, i_idx = codes("folder"), codes("subject_id"), codes("item_id")
S1, I1 = int(s_idx.max()) + 1, int(i_idx.max()) + 1
y = tab["response"].to_numpy().astype(np.int8)
del tab

pair_u, p_inv = np.unique(f_idx * S1 + s_idx, return_inverse=True)
p_inv = p_inv.reshape(-1)
P = len(pair_u)
pair_folder = pair_u // S1
del f_idx, s_idx
iu, i_inv = np.unique(p_inv * I1 + i_idx, return_inverse=True)
i_inv = i_inv.reshape(-1)
item_pair, item_code = iu // I1, iu % I1
n_items_pair = np.bincount(item_pair, minlength=P)
ymean_i = np.bincount(i_inv, weights=y) / np.bincount(i_inv)
lab = np.empty(len(iu), np.int8)
lab[i_inv[::-1]] = y[::-1]
del y, i_idx, p_inv, i_inv, iu
print(f"{P} paires, {len(item_pair) / 1e6:.1f} M (paire, item) | charge en {time.time() - t0:.0f}s")

# decoupage 50/50 INDEPENDANT par paire (comme le reglement : le decoupage est defini par paire)
rng = np.random.default_rng(SEED)
is_eval = rng.random(len(item_pair)) < 0.5
r_ord = rng.random(len(item_pair))

def first_k(mask, k):
    idx = np.flatnonzero(mask)
    idx = idx[np.lexsort((r_ord[idx], item_pair[idx]))]
    pp = item_pair[idx]
    st = np.r_[0, np.flatnonzero(np.diff(pp)) + 1]
    cnt = np.diff(np.r_[st, len(pp)])
    rank = np.arange(len(pp)) - np.repeat(st, cnt)
    keep = rank < k
    return idx[keep], rank[keep], pp[keep]

a_idx, a_rank, a_pair = first_k(~is_eval, 31)
L = np.zeros((P, 31), np.int8); L_it = np.full((P, 31), -1, np.int64)
L[a_pair, a_rank] = lab[a_idx]; L_it[a_pair, a_rank] = item_code[a_idx]
n_acq = np.bincount(a_pair, minlength=P)
e_idx, _, _ = first_k(is_eval, EV_CAP)
ev_pair, ev_item, ev_y = item_pair[e_idx], item_code[e_idx], ymean_i[e_idx]
n_eval = np.bincount(ev_pair, minlength=P).astype(float)
del item_pair, item_code, ymean_i, lab, r_ord, is_eval, a_idx, a_rank, a_pair, e_idx

elig = np.flatnonzero((n_items_pair >= MIN_ITEMS) & (n_acq >= 31) & (n_eval >= MIN_EVAL))
ev_order = np.argsort(ev_pair, kind="stable")
ev_start = np.searchsorted(ev_pair[ev_order], np.arange(P + 1))
print(f"{len(elig)} paires eligibles, {len(np.unique(pair_folder[elig]))} benchmarks")

# ---------- 2. Modele : logit = c + capacite(modele) + difficulte(item), estime sur les labels visibles ----------
def fit_irt(sidx, jidx, yv, nS, nJ, tau_a, tau_d, tau_c=1.5, iters=20):
    c, a, d = 0.0, np.zeros(nS), np.zeros(nJ)
    for _ in range(iters):
        p = expit(c + a[sidx] + d[jidx])
        c += ((yv - p).sum() - c / tau_c ** 2) / ((p * (1 - p)).sum() + 1 / tau_c ** 2)
        p = expit(c + a[sidx] + d[jidx])
        a += (np.bincount(sidx, yv - p, nS) - a / tau_a ** 2) / (np.bincount(sidx, p * (1 - p), nS) + 1 / tau_a ** 2)
        p = expit(c + a[sidx] + d[jidx])
        d += (np.bincount(jidx, yv - p, nJ) - d / tau_d ** 2) / (np.bincount(jidx, p * (1 - p), nJ) + 1 / tau_d ** 2)
    return c, a, d

# ---------- 3. Simulation ----------
folders_el = np.unique(pair_folder[elig])
res = {}   # (S, nom) -> liste de matrices (pairs, 6)
t1 = time.time()
rng2 = np.random.default_rng(7)
for k, f in enumerate(folders_el):
    allp = elig[pair_folder[elig] == f]
    for S in S_GRID:
        vis = allp if S >= len(allp) else rng2.choice(allp, S, replace=False)
        T = len(vis)
        ybar = np.array([ev_y[ev_order[ev_start[p]:ev_start[p + 1]]].mean() for p in vis])
        out = {"base": np.zeros((T, 6))}
        for cfg in CFG:
            out[cfg] = np.zeros((T, 6))
        # lignes d'evaluation de ces paires
        rows = np.concatenate([ev_order[ev_start[p]:ev_start[p + 1]] for p in vis])
        evs = np.repeat(np.arange(T), [ev_start[p + 1] - ev_start[p] for p in vis])
        evj, evy = ev_item[rows], ev_y[rows]
        nT = np.bincount(evs, minlength=T).astype(float)
        for bi, b in enumerate(BUDGETS):
            kk = L[vis, :b].sum(1)
            pb = (kk + 1) / (b + 2)                                 # reference : lissage vers 0,5
            out["base"][:, bi] = pb ** 2 - 2 * pb * ybar + ybar
            for cfg in CFG:
                if b == 0:
                    out[cfg][:, bi] = 0.25                      # sans label, p = 0,5
                    continue
                sidx = np.repeat(np.arange(T), b)
                jj = L_it[vis, :b].ravel()
                yv = L[vis, :b].ravel().astype(float)
                ju, jidx = np.unique(jj, return_inverse=True)
                c, a, d = fit_irt(sidx, jidx.reshape(-1), yv, T, len(ju), cfg[0], cfg[1])
                pos = np.searchsorted(ju, evj); pos[pos >= len(ju)] = len(ju) - 1
                dj = np.where(ju[pos] == evj, d[pos], 0.0)
                pe = expit(c + a[evs] + dj)
                out[cfg][:, bi] = np.bincount(evs, pe ** 2 - 2 * pe * evy + evy, minlength=T) / nT
        for name, B in out.items():
            res.setdefault((S, name), []).append(B)
    if (k + 1) % 20 == 0 or k + 1 == len(folders_el):
        print(f"  {k + 1}/{len(folders_el)} benchmarks ({time.time() - t1:.0f}s)")

# ---------- 4. Resultats ----------
rows = []
for (S, name), bl in res.items():
    B = np.vstack(bl)
    rows.append([("tous" if S >= 10 ** 6 else S), "lissage vers 0,5 (soumission v1)" if name == "base"
                 else f"labels partages tau_modele={name[0]} tau_item={name[1]}",
                 float((B * W).sum(1).mean()), *B.mean(0)])
tb = pd.DataFrame(rows, columns=["modeles_visibles", "modele", "Brier_ALC"] + [f"B{b}" for b in BUDGETS])
pd.set_option("display.width", 220, "display.max_colwidth", 60)
for S in tb["modeles_visibles"].unique():
    print(f"\n=== {S} modeles visibles par benchmark ===")
    print(tb[tb.modeles_visibles == S].drop(columns="modeles_visibles").sort_values("Brier_ALC").round(4).to_string(index=False))
tb.to_csv("/kaggle/working/results_cell7.csv" if os.path.isdir("/kaggle/working") else "results_cell7.csv", index=False)
