# ===== Cellule 14 : second etage appris avec signaux de discrimination (2PL au premier ordre) =====
import os, glob, time, json
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


from sklearn.ensemble import HistGradientBoostingRegressor
from scipy.special import logit

EV_ROWS = int(os.environ.get("EV_ROWS", 40))      # items d'evaluation par paire utilises pour entrainer/evaluer
S_GRID = [10, 30, 100, 10 ** 6]
FEATS = ["eta", "eta_s", "dj", "cov", "nj", "var_a", "var_d", "b_own", "own_logit", "S", "base_logit",
         "tau_a", "tau_d", "item_rate_logit", "mean_labels",
         "step_item", "step_subj", "corr_item", "corr_subj", "res_item"]

# ---------- 2. Modele IRT (identique a la soumission v3) ----------
def fit_model(sidx, jidx, yv, nS, nJ, tau_a=1.25, tau_d=2.0, lo=0.6, hi=3.0, tau_c=1.5, iters=24):
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
        if it_ >= 3 and it_ % 2 == 1:
            tau_a = float(np.clip(np.sqrt(np.mean(a ** 2 + 1 / ha)), lo, hi))
            tau_d = float(np.clip(np.sqrt(np.mean(d ** 2 + 1 / hd)), lo, hi))
    return c, a, d, ha, hd, hc, tau_a, tau_d

fold_of = dict(zip(folders_el, np.random.default_rng(1).permutation(len(folders_el)) % N_FOLDS))

# ---------- 3. Simulation : une ligne par (paire visible, item d'evaluation, budget, nb de modeles visibles) ----------
X_parts, Y_parts, meta_parts = [], [], []
pid_counter = 0
t1 = time.time()
rng2 = np.random.default_rng(7)
for k, f in enumerate(folders_el):
    allp = elig[pair_folder[elig] == f]
    for sc, S in enumerate(S_GRID):
        if S < 10 ** 6 and S >= len(allp):
            continue
        vis = allp if S >= len(allp) else rng2.choice(allp, S, replace=False)
        T = len(vis)
        rows_l, evs_l = [], []
        for t, p in enumerate(vis):
            r = ev_order[ev_start[p]:ev_start[p] + EV_ROWS]
            rows_l.append(r); evs_l.append(np.full(len(r), t))
        rows, evs = np.concatenate(rows_l), np.concatenate(evs_l)
        evj, evy = ev_item[rows], ev_y[rows]
        for bi, b in enumerate(BUDGETS):
            if b == 0:
                continue
            sidx = np.repeat(np.arange(T), b)
            ju, jidx = np.unique(L_it[vis, :b].ravel(), return_inverse=True)
            jidx = jidx.reshape(-1)
            yv = L[vis, :b].ravel().astype(float)
            c, a, d, ha, hd, hc, ta, td = fit_model(sidx, jidx, yv, T, len(ju))
            pos = np.searchsorted(ju, evj); pos[pos >= len(ju)] = len(ju) - 1
            cov = ju[pos] == evj
            nj_all = np.bincount(jidx, minlength=len(ju)).astype(float)
            ys_all = np.bincount(jidx, yv, len(ju))
            base = float(np.clip(yv.mean(), 0.02, 0.98))
            nj = np.where(cov, nj_all[pos], 0.0)
            dj = np.where(cov, d[pos], 0.0)
            vard = np.where(cov, 1 / hd[pos], td ** 2)
            irate = np.where(cov, (ys_all[pos] + 2 * base) / (nj_all[pos] + 2), base)
            kk = L[vis, :b].sum(1)
            own = logit((kk + 1) / (b + 2))
            eta_s = c + a[evs]
            pf = expit(c + a[sidx] + d[jidx])                  # signaux de discrimination : un pas de Newton vers un 2PL
            rs, wt = yv - pf, pf * (1 - pf)
            step_j = np.bincount(jidx, rs * a[sidx], len(ju)) / (np.bincount(jidx, wt * a[sidx] ** 2, len(ju)) + 1.0)
            step_s = np.bincount(sidx, rs * d[jidx], T) / (np.bincount(sidx, wt * d[jidx] ** 2, T) + 1.0)
            res_j = np.bincount(jidx, rs, len(ju)) / (nj_all + 1.0)
            sj = np.where(cov, step_j[pos], 0.0)
            ss = step_s[evs]
            rj = np.where(cov, res_j[pos], 0.0)
            X = np.column_stack([
                eta_s + dj, eta_s, dj, cov.astype(float), nj, 1 / ha[evs], vard,
                np.full(len(evs), float(b)), own[evs], np.full(len(evs), float(T)),
                np.full(len(evs), logit(base)), np.full(len(evs), ta), np.full(len(evs), td),
                logit(np.clip(irate, 0.01, 0.99)), np.full(len(evs), float(b)),
                sj, ss, a[evs] * sj, dj * ss, rj])
            X_parts.append(X)
            Y_parts.append(evy)
            meta_parts.append(np.column_stack([pid_counter + evs, np.full(len(evs), sc), np.full(len(evs), bi),
                                               np.full(len(evs), fold_of[f])]))
        pid_counter += T
    if (k + 1) % 20 == 0 or k + 1 == len(folders_el):
        print(f"  {k + 1}/{len(folders_el)} benchmarks ({time.time() - t1:.0f}s)")
X = np.vstack(X_parts); Y = np.concatenate(Y_parts); M = np.vstack(meta_parts).astype(np.int64)
del X_parts, Y_parts, meta_parts
print(f"{len(Y) / 1e6:.2f} M lignes, {X.shape[1]} variables | simulation en {time.time() - t1:.0f}s")

# ---------- 4. Second etage : arbres, validation croisee par benchmark ----------
def new_gbm():
    return HistGradientBoostingRegressor(loss="squared_error", max_iter=int(os.environ.get("GBM_ITER", 120)),
                                         learning_rate=0.1, max_depth=4, min_samples_leaf=200,
                                         l2_regularization=1.0, early_stopping=False, random_state=0)

pred_stack = np.zeros(len(Y))
for fold in range(N_FOLDS):
    te = M[:, 3] == fold
    gbm = new_gbm().fit(X[~te], Y[~te])
    pred_stack[te] = gbm.predict(X[te])
    print(f"  pli {fold + 1}/{N_FOLDS} termine ({time.time() - t1:.0f}s)")
p0 = expit(X[:, 0])                                   # soumission v3 (mode IRT)
p1 = np.clip(pred_stack, 0.005, 0.995)                # second etage
VARIANTS = {"v3 (IRT seul)": p0, "second etage (arbres)": p1, "moyenne 50/50": 0.5 * (p0 + p1)}

def alc_table(pred):
    loss = pred ** 2 - 2 * pred * Y + Y
    out = {}
    for sc in np.unique(M[:, 1]):
        B = np.full(6, 0.25)
        for bi in range(1, 6):
            m = (M[:, 1] == sc) & (M[:, 2] == bi)
            pid = M[m, 0]
            u, inv = np.unique(pid, return_inverse=True)
            B[bi] = (np.bincount(inv, loss[m]) / np.bincount(inv)).mean()
        out[sc] = B
    return out

pd.set_option("display.width", 200)
tabs = {name: alc_table(pr) for name, pr in VARIANTS.items()}
for sc, S in enumerate(S_GRID):
    if sc not in tabs["v3 (IRT seul)"]:
        continue
    rows = [[name, float((tabs[name][sc] * W).sum()), *tabs[name][sc]] for name in VARIANTS]
    print(f"\n=== {'tous' if S >= 10 ** 6 else S} modeles visibles par benchmark (lignes d'evaluation communes) ===")
    print(pd.DataFrame(rows, columns=["modele", "Brier_ALC"] + [f"B{b}" for b in BUDGETS]).round(4).to_string(index=False))
res = [[("tous" if S >= 10 ** 6 else S), name, float((tabs[name][sc] * W).sum()), *tabs[name][sc]]
       for sc, S in enumerate(S_GRID) if sc in tabs["v3 (IRT seul)"] for name in VARIANTS]
pd.DataFrame(res, columns=["modeles_visibles", "modele", "Brier_ALC"] + [f"B{b}" for b in BUDGETS]).to_csv(
    "/kaggle/working/results_cell14.csv" if os.path.isdir("/kaggle/working") else "results_cell14.csv", index=False)

# ---------- 5. Modele final (tous les benchmarks) et export des arbres en JSON ----------
final = new_gbm().fit(X, Y)
trees = []
for it in final._predictors:
    nd = it[0].nodes
    trees.append({"f": [int(v) if not l else -1 for v, l in zip(nd["feature_idx"], nd["is_leaf"])],
                  "t": [float(v) for v in nd["num_threshold"]],
                  "l": [int(v) for v in nd["left"]], "r": [int(v) for v in nd["right"]],
                  "v": [float(v) for v in nd["value"]]})
export = {"features": FEATS, "baseline": float(np.ravel(final._baseline_prediction)[0]), "trees": trees}

def predict_export(x):
    s = export["baseline"]
    for tr in export["trees"]:
        n = 0
        while tr["f"][n] >= 0:
            n = tr["l"][n] if x[tr["f"][n]] <= tr["t"][n] else tr["r"][n]
        s += tr["v"][n]
    return s

idx = np.random.default_rng(0).choice(len(Y), 3000, replace=False)
err = max(abs(predict_export(X[i]) - final.predict(X[i:i + 1])[0]) for i in idx)
print(f"\nExport JSON : ecart max avec sklearn = {err:.2e} (doit etre < 1e-6)")
assert err < 1e-6, "export des arbres incorrect pour cette version de scikit-learn"
out_json = "/kaggle/working/stack_model14.json" if os.path.isdir("/kaggle/working") else "stack_model14.json"
with open(out_json, "w") as fh:
    json.dump(export, fh)
print("ecrit :", out_json, f"({os.path.getsize(out_json) / 1e3:.0f} ko)")
