# ===== Cellule 5 : difficulte d'item par embeddings, evaluation au niveau item =====
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

# ---------- 3. Modeles (metadonnees) ----------
meta = pd.read_parquet(find("subjects_meta.parquet"),
                       columns=["folder", "subject_id", "normalized_name"]).drop_duplicates(["folder", "subject_id"])
pairs = pd.DataFrame({"folder": f_dic.to_pylist(), }).iloc[pair_folder].reset_index(drop=True)
pairs["subject_id"] = np.array(s_dic.to_pylist(), dtype=object)[pair_subj]
pairs = pairs.merge(meta, on=["folder", "subject_id"], how="left")
assert len(pairs) == P
skey = pairs["normalized_name"].fillna("").astype(str).str.strip().to_numpy()

def fit_rasch(train, C):
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
    return float(clf.intercept_[0]), {s: coef[len(bcodes) + j] for s, j in scodes.items()}, float(rate.mean())

# ---------- 4. Cibles de difficulte d'item ----------
rb = np.bincount(item_fold, weights=it_s, minlength=len(f_dic)) / np.maximum(np.bincount(item_fold, weights=it_n, minlength=len(f_dic)), 1)
t_abs = logit(np.clip((it_s + 2 * rb[item_fold]) / (it_n + 2), 0.02, 0.98))      # logit du taux de reussite de l'item
wt = np.minimum(it_n, 30.0)
wt = wt / np.maximum(np.bincount(item_fold, weights=wt, minlength=len(f_dic))[item_fold], 1e-9)   # chaque benchmark pese pareil
tm = np.bincount(item_fold, weights=wt * t_abs, minlength=len(f_dic)) / np.maximum(np.bincount(item_fold, weights=wt, minlength=len(f_dic)), 1e-9)
t_win = t_abs - tm[item_fold]                                                    # ecart a la moyenne du benchmark

def ridge_fold(tr_items):
    X, w = Emb[tr_items], wt[tr_items]
    w = (w / w.mean()).astype(np.float32)
    mx = (X * w[:, None]).sum(0) / w.sum()
    sd = np.sqrt((((X - mx) ** 2) * w[:, None]).sum(0) / w.sum()) + 1e-6
    Xc = ((X - mx) / sd).astype(np.float32)
    Xw = Xc * w[:, None]
    G = (Xw.T @ Xc).astype(np.float64)
    betas, names = [], []
    for v, tgt in (("A", t_abs[tr_items]), ("B", t_win[tr_items])):
        b = (Xw.T @ (tgt - (tgt * w).sum() / w.sum())).astype(np.float64)
        for lam in LAM_GRID:
            betas.append(np.linalg.solve(G + lam * len(tr_items) * np.eye(D), b)); names.append((v, lam))
    Z = ((Emb - mx) / sd).astype(np.float32)
    return Z @ np.stack(betas, 1).astype(np.float32), names                      # (E, nconfigs)

# ---------- 5. Evaluation leave-benchmark-out ----------
folders_el = np.unique(pair_folder[elig])
fold_of = dict(zip(folders_el, np.random.default_rng(1).permutation(len(folders_el)) % N_FOLDS))
pf_fold = np.array([fold_of.get(f, -1) for f in pair_folder])

results, diag = {}, []
def add(name, B):
    results.setdefault(name, []).append(B)

def run_fold(fold, C_list, full_item_grid):
    held = np.array([f for f in folders_el if fold_of[f] == fold])
    te = elig[pf_fold[elig] == fold]
    tr = np.flatnonzero(~np.isin(pair_folder, held))
    T = len(te)
    loc = np.full(P, -1); loc[te] = np.arange(T)
    rows = np.flatnonzero(loc[ev_pair] >= 0)
    evp, evi, evy = loc[ev_pair[rows]], ev_it[rows], ev_y[rows]
    nT, yb = n_eval[te], ybar[te][:, None]
    B_pair = lambda Pm: Pm ** 2 - 2 * Pm * yb + yb
    add("0,5 constant", B_pair(np.full((T, 6), 0.5)))
    add("moyenne empirique (baseline officielle)", B_pair(np.where(N[te] > 0, K[te] / np.maximum(N[te], 1), 0.5)))
    H = names = None
    if full_item_grid:
        H, names = ridge_fold(np.flatnonzero((~np.isin(item_fold, held)) & (it_n >= 3)))
        # diagnostic : correlation intra-benchmark entre prediction et difficulte reelle sur les benchmarks laisses de cote
        for j, (v, lam) in enumerate(names):
            if v != "B":
                continue
            cs = []
            for f in held:
                m = np.flatnonzero((item_fold == f) & (it_n >= 10))
                if len(m) > 30 and H[m, j].std() > 0 and t_win[m].std() > 0:
                    cs.append(np.corrcoef(H[m, j], t_win[m])[0, 1])
            diag.append((fold, lam, np.mean(cs) if cs else np.nan, len(cs)))
    for C in C_list:
        icpt, ability, mu = fit_rasch(tr, C)
        z = np.array([icpt + ability.get(skey[i], 0.0) for i in te])
        for m in (2, 4, 8):
            add(f"retrecissement vers a priori modele m={m} C={C}", B_pair((K[te] + m * expit(z)[:, None]) / (N[te] + m)))
        cfgs = [("none", None, 0.0)]
        if full_item_grid:
            cfgs += [((v, lam), j, a) for j, (v, lam) in enumerate(names) for a in A_GRID]
        for tag, j, a in cfgs:
            g = H[:, j] if j is not None else None
            base = z[evp] + (a * g[evi] if j is not None else 0.0)
            gL = a * g[np.maximum(L_it[te], 0)] if j is not None else np.zeros((T, 31))
            eta0 = z[:, None] + gL
            for tau in TAU_GRID:
                out = np.empty((T, 6))
                for bi, b in enumerate(BUDGETS):
                    d = np.zeros(T)
                    if b > 0:
                        Lb, e0 = L[te, :b], eta0[:, :b]
                        for _ in range(10):
                            p = expit(e0 + d[:, None])
                            d = d + ((Lb - p).sum(1) - d / tau ** 2) / ((p * (1 - p)).sum(1) + 1 / tau ** 2)
                    pe = expit(base + d[evp])
                    out[:, bi] = np.bincount(evp, pe ** 2 - 2 * pe * evy + evy, minlength=T) / nT
                nm = "logit + decalage benchmark (sans texte)" if j is None else f"logit + texte {tag[0]} lam={tag[1]} a={a}"
                add(f"{nm} tau={tau} C={C}", out)

# choix de C sur le pli 0 (modele sans texte), puis grille complete sur tous les plis
run_fold(0, C_GRID, False)
sub = {k: np.vstack(v) for k, v in results.items() if k.startswith("logit + decalage")}
best_C = min(C_GRID, key=lambda C: min((sub[k] * W).sum(1).mean() for k in sub if k.endswith(f"C={C}")))
print("C retenu :", best_C)
results.clear()
t1 = time.time()
for fold in range(N_FOLDS):
    run_fold(fold, [best_C], True)
    print(f"  pli {fold + 1}/{N_FOLDS} termine ({time.time() - t1:.0f}s)")

# ---------- 6. Resultats ----------
rows = []
for name, bl in results.items():
    B = np.vstack(bl)
    rows.append([name, float((B * W).sum(1).mean()), *B.mean(0)])
res = pd.DataFrame(rows, columns=["modele", "Brier_ALC"] + [f"B{b}" for b in BUDGETS]).sort_values("Brier_ALC")
res["famille"] = res["modele"].str.replace(r"\s+(m|tau|lam)=.*$", "", regex=True).str.replace(r" a=.*$", "", regex=True)
res.to_csv("/kaggle/working/results_cell5.csv" if os.path.isdir("/kaggle/working") else "results_cell5.csv", index=False)
pd.set_option("display.width", 220, "display.max_colwidth", 70)
print("\n=== 15 meilleures configurations (plus bas = mieux) ===")
print(res.drop(columns="famille").head(15).round(4).to_string(index=False))
print("\n=== Meilleure configuration par famille ===")
print(res.groupby("famille").head(1).drop(columns="famille").round(4).to_string(index=False))
bb = [res.sort_values(f"B{b}").iloc[0] for b in BUDGETS]
print("\n=== Meilleure config par budget (optimiste, choisie a posteriori) ===")
for b, r in zip(BUDGETS, bb):
    print(f"B{b}: {r[f'B{b}']:.4f}  <- {r['modele']}")
print("ALC avec meilleur choix par budget : %.4f" % sum(w * r[f"B{b}"] for w, b, r in zip(W, BUDGETS, bb)))
dg = pd.DataFrame(diag, columns=["pli", "lam", "corr_intra_benchmark", "n_benchmarks"])
print("\n=== Correlation intra-benchmark (texte -> difficulte reelle), benchmarks jamais vus ===")
print(dg.groupby("lam")[["corr_intra_benchmark"]].mean().round(3).to_string())
