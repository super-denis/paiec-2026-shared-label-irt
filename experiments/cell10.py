# ===== Cellule 10 : politique d'acquisition (quels items demander) avec labels partages =====
import os, glob, time
import numpy as np, pandas as pd
import pyarrow.parquet as pq
from scipy.special import expit

SEED, MIN_EVAL = 0, 10
MIN_ITEMS = int(os.environ.get("MIN_ITEMS", 80))
EV_CAP = int(os.environ.get("EV_CAP", 200))
BUDGETS = np.array([0, 1, 3, 7, 15, 31])
W = np.array([0.1, 0.2, 0.2, 0.2, 0.2, 0.1])
S_GRID = [30, 10 ** 6]
TAU_A, TAU_D = 1.25, 2.0

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

rng = np.random.default_rng(SEED)
is_eval = rng.random(len(item_pair)) < 0.5
r_ord = rng.random(len(item_pair))

# pool d'acquisition complet de chaque paire, dans l'ordre aleatoire fixe
acq = np.flatnonzero(~is_eval)
o = acq[np.lexsort((r_ord[acq], item_pair[acq]))]
acq_item, acq_y = item_code[o], lab[o]
acq_start = np.searchsorted(item_pair[o], np.arange(P + 1))
n_acq = np.diff(acq_start)
del acq, o

ev = np.flatnonzero(is_eval)
ev = ev[np.lexsort((r_ord[ev], item_pair[ev]))]
pe_ = item_pair[ev]
st = np.r_[0, np.flatnonzero(np.diff(pe_)) + 1]
rank = np.arange(len(pe_)) - np.repeat(st, np.diff(np.r_[st, len(pe_)]))
ev = ev[rank < EV_CAP]
ev_pair, ev_item, ev_y = item_pair[ev], item_code[ev], ymean_i[ev]
n_eval = np.bincount(ev_pair, minlength=P).astype(float)
del item_pair, item_code, ymean_i, lab, r_ord, is_eval, ev, pe_, st, rank
elig = np.flatnonzero((n_items_pair >= MIN_ITEMS) & (n_acq >= 31) & (n_eval >= MIN_EVAL))
ev_order = np.argsort(ev_pair, kind="stable")
ev_start = np.searchsorted(ev_pair[ev_order], np.arange(P + 1))
print(f"{len(elig)} paires eligibles, {len(np.unique(pair_folder[elig]))} benchmarks")

# ---- budget 0 : meilleure constante (sans aucun label, 0,5 n'est pas forcement le meilleur choix) ----
yb_pair = np.bincount(ev_pair, weights=ev_y, minlength=P) / np.maximum(n_eval, 1)
fb = pair_folder[elig]
m_pairs = yb_pair[elig].mean()
m_bench = np.mean([yb_pair[elig][fb == f].mean() for f in np.unique(fb)])
print(f"Taux de reussite moyen des paires eligibles : {m_pairs:.3f} | moyenne par benchmark : {m_bench:.3f}")
print("Brier du budget 0 selon la constante c (0,5 donne 0,2500) :")
for c in np.arange(0.30, 0.601, 0.05):
    print(f"   c = {c:.2f} -> {np.mean(c ** 2 - 2 * c * yb_pair[elig] + yb_pair[elig]):.4f}")

# ---------- 2. Modele partage ----------
def fit_irt(sidx, jidx, yv, nS, nJ, tau_a=TAU_A, tau_d=TAU_D, tau_c=1.5, iters=20):
    c, a, d = 0.0, np.zeros(nS), np.zeros(nJ)
    for _ in range(iters):
        p = expit(c + a[sidx] + d[jidx])
        c += ((yv - p).sum() - c / tau_c ** 2) / ((p * (1 - p)).sum() + 1 / tau_c ** 2)
        p = expit(c + a[sidx] + d[jidx])
        a += (np.bincount(sidx, yv - p, nS) - a / tau_a ** 2) / (np.bincount(sidx, p * (1 - p), nS) + 1 / tau_a ** 2)
        p = expit(c + a[sidx] + d[jidx])
        d += (np.bincount(jidx, yv - p, nJ) - d / tau_d ** 2) / (np.bincount(jidx, p * (1 - p), nJ) + 1 / tau_d ** 2)
    return c, a, d

# ---------- 3. Politiques d'acquisition ----------
# politique : (mode, m).  "nouveaux" : on demande l'item si au plus m labels existent deja pour lui (sauf remplissage force)
#             "deja_vus"  : on demande l'item si au moins m labels existent deja (sauf remplissage force)
POLICIES = {"aleatoire (actuel)": None,
            "nouveaux d'abord (0 label)": ("nouveaux", 0),
            "peu vus (<=1 label)": ("nouveaux", 1),
            "deja vus d'abord (>=2 labels)": ("deja_vus", 2)}

def run_policy(vis, pol, rngp):
    T = len(vis)
    Lp = np.zeros((T, 31), np.int8); Li = np.full((T, 31), -1, np.int64)
    if pol is None:
        for t, p in enumerate(vis):
            Lp[t], Li[t] = acq_y[acq_start[p]:acq_start[p] + 31], acq_item[acq_start[p]:acq_start[p] + 31]
        return Lp, Li
    mode, m = pol
    cnt = {}
    for t in rngp.permutation(T):                       # les paires sont traitees l'une apres l'autre, ordre aleatoire
        p = vis[t]
        items = acq_item[acq_start[p]:acq_start[p + 1]].tolist()
        ys = acq_y[acq_start[p]:acq_start[p + 1]].tolist()
        n, got = len(items), 0
        for pos in range(n):
            if got >= 31:
                break
            c_ = cnt.get(items[pos], 0)
            ok = (n - pos <= 31 - got) or (c_ <= m if mode == "nouveaux" else c_ >= m)
            if ok:
                Lp[t, got], Li[t, got] = ys[pos], items[pos]
                cnt[items[pos]] = c_ + 1
                got += 1
    return Lp, Li

# ---------- 4. Simulation ----------
folders_el = np.unique(pair_folder[elig])
res, stats = {}, {}
t1 = time.time()
rng2 = np.random.default_rng(7)
for k, f in enumerate(folders_el):
    allp = elig[pair_folder[elig] == f]
    for S in S_GRID:
        vis = allp if S >= len(allp) else rng2.choice(allp, S, replace=False)
        T = len(vis)
        cnt_ev = [ev_start[p + 1] - ev_start[p] for p in vis]
        rows = np.concatenate([ev_order[ev_start[p]:ev_start[p + 1]] for p in vis])
        evs, evj, evy = np.repeat(np.arange(T), cnt_ev), ev_item[rows], ev_y[rows]
        nT = np.bincount(evs, minlength=T).astype(float)
        for pname, pol in POLICIES.items():
            Lp, Li = run_policy(vis, pol, np.random.default_rng(100 + k))
            B = np.zeros((T, 6))
            for bi, b in enumerate(BUDGETS):
                if b == 0:
                    B[:, bi] = 0.25
                    continue
                sidx = np.repeat(np.arange(T), b)
                ju, jidx = np.unique(Li[:, :b].ravel(), return_inverse=True)
                jidx = jidx.reshape(-1)
                c, a, d = fit_irt(sidx, jidx, Lp[:, :b].ravel().astype(float), T, len(ju))
                pos = np.searchsorted(ju, evj); pos[pos >= len(ju)] = len(ju) - 1
                cov = ju[pos] == evj
                pe = expit(c + a[evs] + np.where(cov, d[pos], 0.0))
                B[:, bi] = np.bincount(evs, pe ** 2 - 2 * pe * evy + evy, minlength=T) / nT
                if b == 31:
                    stats.setdefault((S, pname), []).append((cov.sum(), len(cov), len(ju)))
            res.setdefault((S, pname), []).append(B)
    if (k + 1) % 20 == 0 or k + 1 == len(folders_el):
        print(f"  {k + 1}/{len(folders_el)} benchmarks ({time.time() - t1:.0f}s)")

# ---------- 5. Resultats ----------
rows = []
for (S, name), bl in res.items():
    B = np.vstack(bl)
    sv = np.array(stats[(S, name)], float)
    rows.append([("tous" if S >= 10 ** 6 else S), name, float((B * W).sum(1).mean()), *B.mean(0),
                 100 * sv[:, 0].sum() / sv[:, 1].sum()])
tb = pd.DataFrame(rows, columns=["modeles_visibles", "politique", "Brier_ALC"] + [f"B{b}" for b in BUDGETS] + ["items_eval_couverts_%"])
pd.set_option("display.width", 220, "display.max_colwidth", 50)
for S in tb["modeles_visibles"].unique():
    print(f"\n=== {S} modeles visibles par benchmark ===")
    print(tb[tb.modeles_visibles == S].drop(columns="modeles_visibles").sort_values("Brier_ALC").round(4).to_string(index=False))
tb.to_csv("/kaggle/working/results_cell10.csv" if os.path.isdir("/kaggle/working") else "results_cell10.csv", index=False)
