# ===== Cellule 8 : labels partages + reglages + texte a l'interieur du benchmark =====
import os, glob, time
import numpy as np, pandas as pd
import pyarrow as pa, pyarrow.compute as pc, pyarrow.parquet as pq
from scipy import sparse
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression

SEED, N_FOLDS, MIN_EVAL = 0, 5, 10
MIN_ITEMS = int(os.environ.get("MIN_ITEMS", 80))
EV_CAP = int(os.environ.get("EV_CAP", 200))          # items d'evaluation max par paire (sous-echantillon aleatoire)
BUDGETS = np.array([0, 1, 3, 7, 15, 31])
W = np.array([0.1, 0.2, 0.2, 0.2, 0.2, 0.1])
C_GRID, TAU_GRID, A_GRID, LAM_GRID = [0.02, 0.2], [1.0, 2.0, 3.0], [0.35, 0.7, 1.0], [0.003, 0.03, 0.3]

def find(name):
    d = os.environ.get("PREP_DIR")
    hits = glob.glob(os.path.join(d, name)) if d else []
    hits += glob.glob(f"/kaggle/input/**/{name}", recursive=True)
    hits += glob.glob(f"/kaggle/working/**/{name}", recursive=True)
    if not hits:
        raise FileNotFoundError(name + " introuvable : ajoute les sorties de paiec-prep-data et paiec-embed comme Inputs")
    return hits[0]

# ---------- 1. Chargement, restreint aux items qui ont un embedding ----------
t0 = time.time()
tab = pq.read_table(find("responses_binary.parquet"),
                    read_dictionary=["folder", "subject_id", "item_id"]).unify_dictionaries()
def codes(col):
    c = tab[col]
    idx = np.concatenate([ch.indices.to_numpy(zero_copy_only=False) for ch in c.chunks]).astype(np.int64)
    return idx, c.chunk(0).dictionary
f_idx, f_dic = codes("folder"); s_idx, s_dic = codes("subject_id"); i_idx, i_dic = codes("item_id")
y = tab["response"].to_numpy().astype(np.int8)
del tab
I1 = len(i_dic) + 1

emb = np.load(find("emb.npy"))
keys = pd.read_parquet(find("emb_keys.parquet"))
fc = pc.index_in(pa.array(keys["folder"].astype(str)), value_set=f_dic.cast(pa.string())).fill_null(-1).to_numpy()
ic = pc.index_in(pa.array(keys["item_id"]).cast(i_dic.type), value_set=i_dic).fill_null(-1).to_numpy()
valid = (fc >= 0) & (ic >= 0)
ekey = fc[valid].astype(np.int64) * I1 + ic[valid]
order = np.argsort(ekey)
ekey_s = ekey[order]
assert len(np.unique(ekey_s)) == len(ekey_s), "cles d'embedding en double"
Emb = emb[np.flatnonzero(valid)[order]].astype(np.float32)
E, D = Emb.shape
item_fold = ekey_s // I1
del emb, keys, fc, ic, ekey, order

rk = f_idx * I1 + i_idx
it = np.searchsorted(ekey_s, rk)
it[it >= E] = E - 1
hit = ekey_s[it] == rk
f_idx, s_idx, y, it = f_idx[hit], s_idx[hit], y[hit], it[hit]
del rk, i_idx, hit
print(f"{len(y) / 1e6:.1f} M lignes gardees (items avec embedding), {E} items, charge en {time.time() - t0:.0f}s")

# ---------- 2. Paires, items, decoupage 50/50 ----------
S1 = len(s_dic) + 1
pair_u, p_inv = np.unique(f_idx * S1 + s_idx, return_inverse=True)
p_inv = p_inv.reshape(-1)
P = len(pair_u)
pair_folder, pair_subj = pair_u // S1, pair_u % S1
del f_idx, s_idx

iu, i_inv = np.unique(p_inv * E + it, return_inverse=True)
i_inv = i_inv.reshape(-1)
item_pair, item_it = iu // E, iu % E
n_items_pair = np.bincount(item_pair, minlength=P)
ymean_i = np.bincount(i_inv, weights=y) / np.bincount(i_inv)
lab = np.empty(len(iu), np.int8)
lab[i_inv[::-1]] = y[::-1]                         # premiere reponse de chaque (paire, item)
it_n = np.bincount(it, minlength=E).astype(float)  # reponses par item, tous modeles confondus
it_s = np.bincount(it, weights=y, minlength=E)
n_all = np.bincount(p_inv, minlength=P).astype(float)
s_all = np.bincount(p_inv, weights=y, minlength=P)
del y, it, p_inv, i_inv, iu

rng = np.random.default_rng(SEED)
is_eval = rng.random(len(item_pair)) < 0.5
r_ord = rng.random(len(item_pair))

def first_k(mask, k):
    """Indices des k premiers elements (ordre aleatoire fixe) de chaque paire parmi mask."""
    idx = np.flatnonzero(mask)
    idx = idx[np.lexsort((r_ord[idx], item_pair[idx]))]
    pp = item_pair[idx]
    st = np.r_[0, np.flatnonzero(np.diff(pp)) + 1]
    cnt = np.diff(np.r_[st, len(pp)])
    rank = np.arange(len(pp)) - np.repeat(st, cnt)
    keep = rank < k
    return idx[keep], rank[keep], pp[keep]

acq_idx, acq_rank, acq_pair = first_k(~is_eval, 31)
L = np.zeros((P, 31), np.int8); L_it = np.full((P, 31), -1, np.int64)
L[acq_pair, acq_rank] = lab[acq_idx]; L_it[acq_pair, acq_rank] = item_it[acq_idx]
n_acq = np.bincount(acq_pair, minlength=P)

ev_idx, _, _ = first_k(is_eval, EV_CAP)
ev_pair, ev_it, ev_y = item_pair[ev_idx], item_it[ev_idx], ymean_i[ev_idx]
n_eval = np.bincount(ev_pair, minlength=P).astype(float)
ybar = np.bincount(ev_pair, weights=ev_y, minlength=P) / np.maximum(n_eval, 1)
del item_pair, item_it, ymean_i, lab, r_ord, is_eval, acq_idx, acq_rank, acq_pair, ev_idx

elig = np.flatnonzero((n_items_pair >= MIN_ITEMS) & (n_acq >= 31) & (n_eval >= MIN_EVAL))
K = np.stack([L[:, :b].sum(1) for b in BUDGETS], 1).astype(float)
N = np.tile(BUDGETS.astype(float), (P, 1))
print(f"{P} paires, {len(elig)} eligibles, {len(np.unique(pair_folder[elig]))} benchmarks | pret en {time.time() - t0:.0f}s")


S_GRID = [30, 10 ** 6]                                     # nombre de modeles visibles par benchmark (30, puis tous)
CFG0 = [(1.0, 1.5), (1.0, 2.5), (0.7, 2.5), (1.5, 2.5), (1.0, 4.0)]     # (tau_modele, tau_item)
TXT_BASE = (1.0, 2.5)
TXT = [(lam, kappa) for lam in (0.3, 1.0, 3.0) for kappa in (0.5, 1.0)]  # (ridge, facteur de confiance)

ev_order = np.argsort(ev_pair, kind="stable")
ev_start = np.searchsorted(ev_pair[ev_order], np.arange(P + 1))

def fit_irt(sidx, jidx, yv, nS, nJ, tau_a, tau_d, dm=None, tau_c=1.5, iters=20):
    c, a = 0.0, np.zeros(nS)
    dm = np.zeros(nJ) if dm is None else dm
    d = dm.copy()
    for _ in range(iters):
        p = expit(c + a[sidx] + d[jidx])
        c += ((yv - p).sum() - c / tau_c ** 2) / ((p * (1 - p)).sum() + 1 / tau_c ** 2)
        p = expit(c + a[sidx] + d[jidx])
        a += (np.bincount(sidx, yv - p, nS) - a / tau_a ** 2) / (np.bincount(sidx, p * (1 - p), nS) + 1 / tau_a ** 2)
        p = expit(c + a[sidx] + d[jidx])
        d += (np.bincount(jidx, yv - p, nJ) - (d - dm) / tau_d ** 2) / (np.bincount(jidx, p * (1 - p), nJ) + 1 / tau_d ** 2)
    p = expit(c + a[sidx] + d[jidx])
    return c, a, d, np.bincount(jidx, p * (1 - p), nJ)

def text_prior(items, d, info, lam):
    """Ridge (dans le benchmark) de la difficulte estimee sur l'embedding ; renvoie une fonction items -> prior de difficulte."""
    X, w = Emb[items], info + 1e-6
    ws = w.sum()
    mx = (X * w[:, None]).sum(0) / ws
    sd = np.sqrt((((X - mx) ** 2) * w[:, None]).sum(0) / ws) + 1e-3
    Xc = (X - mx) / sd
    beta = np.linalg.solve((Xc * w[:, None]).T @ Xc + lam * ws * np.eye(D), Xc.T @ (w * d))
    return lambda it: ((Emb[it] - mx) / sd) @ beta

folders_el = np.unique(pair_folder[elig])
res, cover = {}, []
t1 = time.time()
rng2 = np.random.default_rng(7)
for k, f in enumerate(folders_el):
    allp = elig[pair_folder[elig] == f]
    for S in S_GRID:
        vis = allp if S >= len(allp) else rng2.choice(allp, S, replace=False)
        T = len(vis)
        cnt = [ev_start[p + 1] - ev_start[p] for p in vis]
        rows = np.concatenate([ev_order[ev_start[p]:ev_start[p + 1]] for p in vis])
        evs, evj, evy = np.repeat(np.arange(T), cnt), ev_it[rows], ev_y[rows]
        nT = np.bincount(evs, minlength=T).astype(float)
        ybar = np.bincount(evs, evy, minlength=T) / nT
        out = {"base": np.zeros((T, 6))}
        for cfg in CFG0:
            out[("irt",) + cfg] = np.zeros((T, 6))
        for t in TXT:
            out[("txt",) + t] = np.zeros((T, 6))
        def brier(pe):
            return np.bincount(evs, pe ** 2 - 2 * pe * evy + evy, minlength=T) / nT
        for bi, b in enumerate(BUDGETS):
            pb = (L[vis, :b].sum(1) + 1) / (b + 2)
            out["base"][:, bi] = pb ** 2 - 2 * pb * ybar + ybar
            if b == 0:
                for key in out:
                    if key != "base":
                        out[key][:, bi] = 0.25
                continue
            sidx = np.repeat(np.arange(T), b)
            ju, jidx = np.unique(L_it[vis, :b].ravel(), return_inverse=True)
            jidx = jidx.reshape(-1)
            yv = L[vis, :b].ravel().astype(float)
            pos = np.searchsorted(ju, evj); pos[pos >= len(ju)] = len(ju) - 1
            cov = ju[pos] == evj
            if b == 31 and S >= 10 ** 6:
                cover.append((cov.mean(), len(evj)))
            for cfg in CFG0:
                c, a, d, _ = fit_irt(sidx, jidx, yv, T, len(ju), cfg[0], cfg[1])
                out[("irt",) + cfg][:, bi] = brier(expit(c + a[evs] + np.where(cov, d[pos], 0.0)))
            c, a, d, info = fit_irt(sidx, jidx, yv, T, len(ju), *TXT_BASE)
            for lam in sorted({t[0] for t in TXT}):
                prior = text_prior(ju, d, info, lam)
                m_cov, m_ev = prior(ju), prior(evj)
                for kappa in sorted({t[1] for t in TXT}):
                    c2, a2, d2, _ = fit_irt(sidx, jidx, yv, T, len(ju), *TXT_BASE, dm=kappa * m_cov)
                    dj = np.where(cov, d2[pos], kappa * m_ev)
                    out[("txt", lam, kappa)][:, bi] = brier(expit(c2 + a2[evs] + dj))
        for name, B in out.items():
            res.setdefault((S, name), []).append(B)
    if (k + 1) % 20 == 0 or k + 1 == len(folders_el):
        print(f"  {k + 1}/{len(folders_el)} benchmarks ({time.time() - t1:.0f}s)")

rows = []
for (S, name), bl in res.items():
    B = np.vstack(bl)
    label = ("lissage vers 0,5 (v1)" if name == "base" else
             f"IRT partage tau_modele={name[1]} tau_item={name[2]}" if name[0] == "irt" else
             f"IRT partage + texte ridge={name[1]} confiance={name[2]}")
    rows.append([("tous" if S >= 10 ** 6 else S), label, float((B * W).sum(1).mean()), *B.mean(0)])
tb = pd.DataFrame(rows, columns=["modeles_visibles", "modele", "Brier_ALC"] + [f"B{b}" for b in BUDGETS])
pd.set_option("display.width", 220, "display.max_colwidth", 70)
for S in tb["modeles_visibles"].unique():
    print(f"\n=== {S} modeles visibles par benchmark ===")
    print(tb[tb.modeles_visibles == S].drop(columns="modeles_visibles").sort_values("Brier_ALC").round(4).to_string(index=False))
if cover:
    cv = np.array(cover)
    print("\nPart des items d'evaluation deja etiquetes par d'autres modeles (tous visibles, B31) : %.1f%%" % (100 * (cv[:, 0] * cv[:, 1]).sum() / cv[:, 1].sum()))
tb.to_csv("/kaggle/working/results_cell8.csv" if os.path.isdir("/kaggle/working") else "results_cell8.csv", index=False)
