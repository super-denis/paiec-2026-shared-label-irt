# ===== Cellule 12 : prediction bayesienne complete (variance a posteriori) au lieu du mode MAP =====
import os, glob, time
import numpy as np, pandas as pd
import pyarrow.parquet as pq
from scipy.special import expit

SEED, N_FOLDS, MIN_EVAL = 0, 5, 10
MIN_ITEMS = int(os.environ.get("MIN_ITEMS", 80))
EV_CAP = int(os.environ.get("EV_CAP", 200))
BUDGETS = np.array([0, 1, 3, 7, 15, 31])
W = np.array([0.1, 0.2, 0.2, 0.2, 0.2, 0.1])

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


print(f"{len(elig)} paires eligibles, {len(np.unique(pair_folder[elig]))} benchmarks")
folders_el = np.unique(pair_folder[elig])

# ---------- 2. Modele : logit = c + capacite + difficulte ; tau fixes ou estimes (Bayes empirique) ----------
def fit_model(sidx, jidx, yv, nS, nJ, tau_a, tau_d, eb=False, lo=0.6, hi=3.0, tau_c=1.5, iters=24):
    c, a, d = 0.0, np.zeros(nS), np.zeros(nJ)
    for it_ in range(iters):
        p = expit(c + a[sidx] + d[jidx])
        hc = (p * (1 - p)).sum() + 1 / tau_c ** 2
        c += ((yv - p).sum() - c / tau_c ** 2) / hc
        p = expit(c + a[sidx] + d[jidx])
        ha = np.bincount(sidx, p * (1 - p), nS) + 1 / tau_a ** 2
        a += (np.bincount(sidx, yv - p, nS) - a / tau_a ** 2) / ha
        p = expit(c + a[sidx] + d[jidx])
        hd = np.bincount(jidx, p * (1 - p), nJ) + 1 / tau_d ** 2
        d += (np.bincount(jidx, yv - p, nJ) - d / tau_d ** 2) / hd
        if eb and it_ >= 3 and it_ % 2 == 1:
            tau_a = float(np.clip(np.sqrt(np.mean(a ** 2 + 1 / ha)), lo, hi))
            tau_d = float(np.clip(np.sqrt(np.mean(d ** 2 + 1 / hd)), lo, hi))
    return c, a, d, ha, hd, hc, tau_a, tau_d

# nom du modele ajuste -> (tau_modele, tau_item, bayes empirique ?)
FITS = {"EB 1.25/2.0": (1.25, 2.0, True), "fixe 1.25/2.0": (1.25, 2.0, False)}
# facteur sur la variance a posteriori : 0 = mode MAP simple (soumission v3), 1 = prediction bayesienne approchee
LAMS = [0.0, 0.5, 1.0, 1.5, 2.0]
S_GRID = [10, 30, 100, 10 ** 6]

# ---------- 3. Simulation ----------
res, taus = {}, {}
t1 = time.time()
rng2 = np.random.default_rng(7)
for k, f in enumerate(folders_el):
    allp = elig[pair_folder[elig] == f]
    for S in S_GRID:
        if S < 10 ** 6 and S >= len(allp):
            continue                                   # meme situation que "tous"
        vis = allp if S >= len(allp) else rng2.choice(allp, S, replace=False)
        T = len(vis)
        cnt = [ev_start[p + 1] - ev_start[p] for p in vis]
        rows = np.concatenate([ev_order[ev_start[p]:ev_start[p + 1]] for p in vis])
        evs, evj, evy = np.repeat(np.arange(T), cnt), ev_item[rows], ev_y[rows]
        nT = np.bincount(evs, minlength=T).astype(float)
        ybar = np.bincount(evs, evy, minlength=T) / nT
        names = [f"{fn} x{lam}" for fn in FITS for lam in LAMS]
        out = {name: np.zeros((T, 6)) for name in names + ["lissage vers 0,5 (v1)"]}
        for bi, b in enumerate(BUDGETS):
            pb = (L[vis, :b].sum(1) + 1) / (b + 2)
            out["lissage vers 0,5 (v1)"][:, bi] = pb ** 2 - 2 * pb * ybar + ybar
            if b == 0:
                for name in names:
                    out[name][:, bi] = 0.25
                continue
            sidx = np.repeat(np.arange(T), b)
            ju, jidx = np.unique(L_it[vis, :b].ravel(), return_inverse=True)
            jidx = jidx.reshape(-1)
            yv = L[vis, :b].ravel().astype(float)
            pos = np.searchsorted(ju, evj); pos[pos >= len(ju)] = len(ju) - 1
            cov = ju[pos] == evj
            for fn, (ta, td, eb) in FITS.items():
                c, a, d, ha, hd, hc, ta2, td2 = fit_model(sidx, jidx, yv, T, len(ju), ta, td, eb)
                if fn == "EB 1.25/2.0":
                    taus.setdefault((S, b), []).append((ta2, td2))
                eta = c + a[evs] + np.where(cov, d[pos], 0.0)
                var = 1 / hc + 1 / ha[evs] + np.where(cov, 1 / hd[pos], td2 ** 2)   # variance a posteriori (diagonale)
                for lam in LAMS:
                    pe = expit(eta / np.sqrt(1 + lam * np.pi / 8 * var))
                    out[f"{fn} x{lam}"][:, bi] = np.bincount(evs, pe ** 2 - 2 * pe * evy + evy, minlength=T) / nT
        for name, B in out.items():
            res.setdefault((S, name), []).append(B)
    if (k + 1) % 20 == 0 or k + 1 == len(folders_el):
        print(f"  {k + 1}/{len(folders_el)} benchmarks ({time.time() - t1:.0f}s)")

# ---------- 4. Resultats ----------
rows = []
for (S, name), bl in res.items():
    B = np.vstack(bl)
    rows.append([("tous" if S >= 10 ** 6 else S), name, float((B * W).sum(1).mean()), *B.mean(0)])
tb = pd.DataFrame(rows, columns=["modeles_visibles", "modele", "Brier_ALC"] + [f"B{b}" for b in BUDGETS])
pd.set_option("display.width", 220, "display.max_colwidth", 50)
for S in tb["modeles_visibles"].unique():
    print(f"\n=== {S} modeles visibles par benchmark ===")
    print(tb[tb.modeles_visibles.astype(str) == str(S)].drop(columns="modeles_visibles").sort_values("Brier_ALC").round(4).to_string(index=False))
print("\nTau estimes (moyenne par benchmark) : (modeles visibles, budget) -> (tau_modele, tau_item)")
for key in sorted(taus):
    v = np.array(taus[key])
    print(f"   {('tous' if key[0] >= 10 ** 6 else key[0])!s:>5} B{key[1]:<2d} -> {v[:, 0].mean():.2f} / {v[:, 1].mean():.2f}")
tb.to_csv("/kaggle/working/results_cell12.csv" if os.path.isdir("/kaggle/working") else "results_cell12.csv", index=False)
