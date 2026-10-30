# ===== Cellule 9 : a priori de capacite par nom de modele + facteurs latents + reglages =====
import os, glob, time
import numpy as np, pandas as pd
import pyarrow.parquet as pq
from scipy import sparse
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

SEED, N_FOLDS, MIN_EVAL = 0, 5, 10
MIN_ITEMS = int(os.environ.get("MIN_ITEMS", 80))
EV_CAP = int(os.environ.get("EV_CAP", 200))
BUDGETS = np.array([0, 1, 3, 7, 15, 31])
W = np.array([0.1, 0.2, 0.2, 0.2, 0.2, 0.1])
S_GRID = [30, 10 ** 6]

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
def codes(col, keep=False):
    c = tab[col]
    idx = np.concatenate([ch.indices.to_numpy(zero_copy_only=False) for ch in c.chunks]).astype(np.int64)
    return (idx, c.chunk(0).dictionary.to_pylist()) if keep else idx
f_idx, f_dict = codes("folder", True)
s_idx, s_dict = codes("subject_id", True)
i_idx = codes("item_id")
S1, I1 = len(s_dict) + 1, int(i_idx.max()) + 1
y = tab["response"].to_numpy().astype(np.int8)
del tab

pair_u, p_inv = np.unique(f_idx * S1 + s_idx, return_inverse=True)
p_inv = p_inv.reshape(-1)
P = len(pair_u)
pair_folder, pair_subj = pair_u // S1, pair_u % S1
del f_idx, s_idx
n_all = np.bincount(p_inv, minlength=P).astype(float)
s_all = np.bincount(p_inv, weights=y, minlength=P)
iu, i_inv = np.unique(p_inv * I1 + i_idx, return_inverse=True)
i_inv = i_inv.reshape(-1)
item_pair, item_code = iu // I1, iu % I1
n_items_pair = np.bincount(item_pair, minlength=P)
ymean_i = np.bincount(i_inv, weights=y) / np.bincount(i_inv)
lab = np.empty(len(iu), np.int8)
lab[i_inv[::-1]] = y[::-1]
del y, i_idx, p_inv, i_inv, iu
print(f"{P} paires, {len(item_pair) / 1e6:.1f} M (paire, item) | charge en {time.time() - t0:.0f}s")

rng = np.random.default_rng(SEED)
is_eval = rng.random(len(item_pair)) < 0.5            # decoupage independant par paire
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

# ---------- 2. Noms de modeles et a priori de capacite (appris hors du benchmark teste) ----------
meta = pd.read_parquet(find("subjects_meta.parquet"),
                       columns=["folder", "subject_id", "normalized_name"]).drop_duplicates(["folder", "subject_id"])
pairs = pd.DataFrame({"folder": np.array(f_dict, dtype=object)[pair_folder],
                      "subject_id": np.array(s_dict, dtype=object)[pair_subj]}).merge(
    meta, on=["folder", "subject_id"], how="left")
assert len(pairs) == P
skey = pairs["normalized_name"].fillna("").astype(str).str.strip().to_numpy()
print(f"{len(elig)} paires eligibles, {len(np.unique(pair_folder[elig]))} benchmarks | "
      f"modele nomme : {(skey[elig] != '').mean():.0%} des paires eligibles")

def fit_rasch(train, C=0.02):
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
    return {s: float(coef[len(bcodes) + j]) for s, j in scodes.items()}

folders_el = np.unique(pair_folder[elig])
fold_of = dict(zip(folders_el, np.random.default_rng(1).permutation(len(folders_el)) % N_FOLDS))
abil = {}
for fold in range(N_FOLDS):
    held = [f for f in folders_el if fold_of[f] == fold]
    abil[fold] = fit_rasch(np.flatnonzero(~np.isin(pair_folder, held)))
print(f"a priori de capacite appris en {time.time() - t0:.0f}s")

# ---------- 3. Modele : logit = c + capacite + difficulte (+ facteurs latents) ----------
def fit_model(sidx, jidx, yv, nS, nJ, tau_a, tau_d, am, rank=0, tau_f=1.0, tau_c=1.5, iters=20):
    c, a, d = 0.0, am.copy(), np.zeros(nJ)
    U = V = None
    if rank:
        r2 = np.random.default_rng(0)
        U, V = 0.1 * r2.standard_normal((nS, rank)), 0.1 * r2.standard_normal((nJ, rank))
    def prob():
        eta = c + a[sidx] + d[jidx]
        if rank:
            eta = eta + (U[sidx] * V[jidx]).sum(1)
        return expit(eta)
    def newton_vec(own, other, oidx, tgt, n, p, w):
        g = np.stack([np.bincount(oidx, (yv - p) * other[tgt, k], n) for k in range(rank)], 1) - own / tau_f ** 2
        H = np.zeros((n, rank, rank)) + np.eye(rank) / tau_f ** 2
        for k in range(rank):
            for l in range(rank):
                H[:, k, l] += np.bincount(oidx, w * other[tgt, k] * other[tgt, l], n)
        return own + np.linalg.solve(H, g[..., None])[..., 0]
    for _ in range(iters):
        p = prob()
        c += ((yv - p).sum() - c / tau_c ** 2) / ((p * (1 - p)).sum() + 1 / tau_c ** 2)
        p = prob()
        a += (np.bincount(sidx, yv - p, nS) - (a - am) / tau_a ** 2) / (np.bincount(sidx, p * (1 - p), nS) + 1 / tau_a ** 2)
        p = prob()
        d += (np.bincount(jidx, yv - p, nJ) - d / tau_d ** 2) / (np.bincount(jidx, p * (1 - p), nJ) + 1 / tau_d ** 2)
        if rank:
            p = prob()
            U = newton_vec(U, V, sidx, jidx, nS, p, p * (1 - p))
            p = prob()
            V = newton_vec(V, U, jidx, sidx, nJ, p, p * (1 - p))
    return c, a, d, U, V

# nom -> (tau_modele, tau_item, kappa_prior_nom, rang, tau_facteurs)
CFG = {
    "IRT 1.0/1.5 (v2)": (1.0, 1.5, 0.0, 0, 1.0),
    "IRT 1.5/2.5": (1.5, 2.5, 0.0, 0, 1.0),
    "IRT 1.25/2.0": (1.25, 2.0, 0.0, 0, 1.0),
    "IRT 1.25/2.0 + prior nom x0.5": (1.25, 2.0, 0.5, 0, 1.0),
    "IRT 1.25/2.0 + prior nom x1.0": (1.25, 2.0, 1.0, 0, 1.0),
    "IRT 0.8/2.0 + prior nom x1.0": (0.8, 2.0, 1.0, 0, 1.0),
    "IRT 1.25/2.0 + facteurs r=2 tau=1.0": (1.25, 2.0, 0.0, 2, 1.0),
    "IRT 1.25/2.0 + facteurs r=2 tau=1.5": (1.25, 2.0, 0.0, 2, 1.5),
    "IRT 1.25/2.0 + facteurs r=2 tau=0.75": (1.25, 2.0, 0.0, 2, 0.75),
}

# ---------- 4. Simulation ----------
res, cover = {}, []
t1 = time.time()
rng2 = np.random.default_rng(7)
for k, f in enumerate(folders_el):
    allp = elig[pair_folder[elig] == f]
    ability = abil[fold_of[f]]
    for S in S_GRID:
        vis = allp if S >= len(allp) else rng2.choice(allp, S, replace=False)
        T = len(vis)
        cnt = [ev_start[p + 1] - ev_start[p] for p in vis]
        rows = np.concatenate([ev_order[ev_start[p]:ev_start[p + 1]] for p in vis])
        evs, evj, evy = np.repeat(np.arange(T), cnt), ev_item[rows], ev_y[rows]
        nT = np.bincount(evs, minlength=T).astype(float)
        ybar = np.bincount(evs, evy, minlength=T) / nT
        ab_prior = np.array([ability.get(skey[p], 0.0) for p in vis])
        out = {name: np.zeros((T, 6)) for name in list(CFG) + ["lissage vers 0,5 (v1)"]}
        for bi, b in enumerate(BUDGETS):
            pb = (L[vis, :b].sum(1) + 1) / (b + 2)
            out["lissage vers 0,5 (v1)"][:, bi] = pb ** 2 - 2 * pb * ybar + ybar
            if b == 0:
                for name in CFG:
                    out[name][:, bi] = 0.25
                continue
            sidx = np.repeat(np.arange(T), b)
            ju, jidx = np.unique(L_it[vis, :b].ravel(), return_inverse=True)
            jidx = jidx.reshape(-1)
            yv = L[vis, :b].ravel().astype(float)
            pos = np.searchsorted(ju, evj); pos[pos >= len(ju)] = len(ju) - 1
            cov = ju[pos] == evj
            if b == 31 and S >= 10 ** 6:
                cover.append((cov.mean(), len(evj)))
            for name, (ta, td, kap, rk_, tf) in CFG.items():
                c, a, d, U, V = fit_model(sidx, jidx, yv, T, len(ju), ta, td, kap * ab_prior, rk_, tf)
                eta = c + a[evs] + np.where(cov, d[pos], 0.0)
                if rk_:
                    eta = eta + np.where(cov, (U[evs] * V[pos]).sum(1), 0.0)
                pe = expit(eta)
                out[name][:, bi] = np.bincount(evs, pe ** 2 - 2 * pe * evy + evy, minlength=T) / nT
        for name, B in out.items():
            res.setdefault((S, name), []).append(B)
    if (k + 1) % 20 == 0 or k + 1 == len(folders_el):
        print(f"  {k + 1}/{len(folders_el)} benchmarks ({time.time() - t1:.0f}s)")

# ---------- 5. Resultats ----------
rows = []
for (S, name), bl in res.items():
    B = np.vstack(bl)
    rows.append([("tous" if S >= 10 ** 6 else S), name, float((B * W).sum(1).mean()), *B.mean(0)])
tb = pd.DataFrame(rows, columns=["modeles_visibles", "modele", "Brier_ALC"] + [f"B{b}" for b in BUDGETS])
pd.set_option("display.width", 220, "display.max_colwidth", 60)
for S in tb["modeles_visibles"].unique():
    print(f"\n=== {S} modeles visibles par benchmark ===")
    print(tb[tb.modeles_visibles == S].drop(columns="modeles_visibles").sort_values("Brier_ALC").round(4).to_string(index=False))
if cover:
    cv = np.array(cover)
    print("\nPart des items d'evaluation deja etiquetes par d'autres modeles (tous visibles, B31) : %.1f%%" % (100 * (cv[:, 0] * cv[:, 1]).sum() / cv[:, 1].sum()))
tb.to_csv("/kaggle/working/results_cell9.csv" if os.path.isdir("/kaggle/working") else "results_cell9.csv", index=False)
