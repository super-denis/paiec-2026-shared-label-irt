"""Shared-label IRT with a learned second stage (pure Python).

Stage 1 (as in v3): for the benchmark of the query, fit logit p = c + ability(subject) + difficulty(item)
on every revealed label of that benchmark (any subject), with empirical Bayes prior scales.
Stage 2: a small gradient-boosted tree ensemble (stack_model.json, trained offline on simulated episodes
from the public training data, validated by leave-benchmark-out) corrects the stage 1 probability using
its uncertainty, the item coverage, the benchmark base rate and the number of labels.
Falls back to stage 1 when the target subject has no label on the benchmark, and to 0.5 with no label.
"""

import json
import math
import os

TAU_A0, TAU_D0, TAU_C, ITERS = 1.25, 2.0, 1.5, 24
TAU_LO, TAU_HI = 0.6, 3.0
_CACHE = {}
_STACK = None


def _stack():
    global _STACK
    if _STACK is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stack_model.json")
        try:
            with open(path, encoding="utf-8") as handle:
                _STACK = json.load(handle)
        except Exception:
            _STACK = {}
    return _STACK


def _sigmoid(x):
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def _logit(p):
    return math.log(p / (1.0 - p))


def _key(obj):
    return json.dumps(obj, sort_keys=True, default=str)


def _fit(entries):
    """entries: list of (subject_key, item_key, y). Returns a dict with the fitted quantities."""
    s_ids, j_ids = {}, {}
    s_idx = [s_ids.setdefault(s, len(s_ids)) for s, _, _ in entries]
    j_idx = [j_ids.setdefault(j, len(j_ids)) for _, j, _ in entries]
    y = [v for _, _, v in entries]
    nS, nJ, n = len(s_ids), len(j_ids), len(y)
    c, a, d = 0.0, [0.0] * nS, [0.0] * nJ
    ta, td = TAU_A0, TAU_D0
    ha, hd = [1.0] * nS, [1.0] * nJ
    for it in range(ITERS):
        p = [_sigmoid(c + a[s_idx[i]] + d[j_idx[i]]) for i in range(n)]
        g = sum(y[i] - p[i] for i in range(n)) - c / TAU_C ** 2
        h = sum(p[i] * (1 - p[i]) for i in range(n)) + 1 / TAU_C ** 2
        c += g / h
        p = [_sigmoid(c + a[s_idx[i]] + d[j_idx[i]]) for i in range(n)]
        g, ha = [-v / ta ** 2 for v in a], [1 / ta ** 2] * nS
        for i in range(n):
            g[s_idx[i]] += y[i] - p[i]
            ha[s_idx[i]] += p[i] * (1 - p[i])
        a = [a[k] + g[k] / ha[k] for k in range(nS)]
        p = [_sigmoid(c + a[s_idx[i]] + d[j_idx[i]]) for i in range(n)]
        g, hd = [-v / td ** 2 for v in d], [1 / td ** 2] * nJ
        for i in range(n):
            g[j_idx[i]] += y[i] - p[i]
            hd[j_idx[i]] += p[i] * (1 - p[i])
        d = [d[k] + g[k] / hd[k] for k in range(nJ)]
        if it >= 3 and it % 2 == 1:
            ta = min(TAU_HI, max(TAU_LO, math.sqrt(sum(a[k] ** 2 + 1 / ha[k] for k in range(nS)) / nS)))
            td = min(TAU_HI, max(TAU_LO, math.sqrt(sum(d[k] ** 2 + 1 / hd[k] for k in range(nJ)) / nJ)))
    nj, ys, own_n, own_k = [0] * nJ, [0.0] * nJ, [0] * nS, [0.0] * nS
    gj, ij, gs, is_, rj = [0.0] * nJ, [0.0] * nJ, [0.0] * nS, [0.0] * nS, [0.0] * nJ
    for i in range(n):
        si_, ji_ = s_idx[i], j_idx[i]
        nj[ji_] += 1
        ys[ji_] += y[i]
        own_n[si_] += 1
        own_k[si_] += y[i]
        pf = _sigmoid(c + a[si_] + d[ji_])
        r_, w_ = y[i] - pf, pf * (1 - pf)
        gj[ji_] += r_ * a[si_]
        ij[ji_] += w_ * a[si_] ** 2
        gs[si_] += r_ * d[ji_]
        is_[si_] += w_ * d[ji_] ** 2
        rj[ji_] += r_
    step_j = [gj[k] / (ij[k] + 1.0) for k in range(nJ)]
    step_s = [gs[k] / (is_[k] + 1.0) for k in range(nS)]
    res_j = [rj[k] / (nj[k] + 1.0) for k in range(nJ)]
    base = min(0.98, max(0.02, sum(y) / n))
    return {"c": c, "a": a, "d": d, "ha": ha, "hd": hd, "ta": ta, "td": td, "s": s_ids, "j": j_ids,
            "nj": nj, "ys": ys, "own_n": own_n, "own_k": own_k, "base": base, "T": nS, "N": n,
            "step_j": step_j, "step_s": step_s, "res_j": res_j}


def _tree_sum(stack, x):
    total = stack["baseline"]
    for tr in stack["trees"]:
        node = 0
        feat = tr["f"]
        while feat[node] >= 0:
            node = tr["l"][node] if x[feat[node]] <= tr["t"][node] else tr["r"][node]
        total += tr["v"][node]
    return total


def predict(input: list[dict], labeled: list | None = None) -> float:
    subject, item = input
    benchmark = item.get("benchmark_id")
    if not labeled or not isinstance(benchmark, str):
        return 0.5
    ck = (benchmark, len(labeled))
    if ck not in _CACHE:
        entries = []
        for entry in labeled:
            (obs_subject, obs_item), label = entry
            if obs_item.get("benchmark_id") != benchmark or label not in (0, 1):
                continue
            entries.append((_key(obs_subject), _key(obs_item), float(label)))
        if len(_CACHE) > 64:
            _CACHE.clear()
        _CACHE[ck] = _fit(entries) if entries else None
    fit = _CACHE[ck]
    if fit is None:
        return 0.5
    sk, jk = _key(subject), _key(item)
    si, ji = fit["s"].get(sk), fit["j"].get(jk)
    a_s = fit["a"][si] if si is not None else 0.0
    d_j = fit["d"][ji] if ji is not None else 0.0
    eta = fit["c"] + a_s + d_j
    p0 = min(0.98, max(0.02, _sigmoid(eta)))
    stack = _stack()
    if si is None or not stack or fit["own_n"][si] == 0:
        return float(p0)
    try:
        b_own = fit["own_n"][si]
        cov = 1.0 if ji is not None else 0.0
        n_j = float(fit["nj"][ji]) if ji is not None else 0.0
        var_d = 1.0 / fit["hd"][ji] if ji is not None else fit["td"] ** 2
        item_rate = ((fit["ys"][ji] + 2 * fit["base"]) / (fit["nj"][ji] + 2)) if ji is not None else fit["base"]
        own = _logit((fit["own_k"][si] + 1.0) / (b_own + 2.0))
        sj = fit["step_j"][ji] if ji is not None else 0.0
        ss = fit["step_s"][si]
        rj = fit["res_j"][ji] if ji is not None else 0.0
        feats = {"eta": eta, "eta_s": fit["c"] + a_s, "dj": d_j, "cov": cov, "nj": n_j,
                 "var_a": 1.0 / fit["ha"][si], "var_d": var_d, "b_own": float(b_own), "own_logit": own,
                 "S": float(fit["T"]), "base_logit": _logit(fit["base"]), "tau_a": fit["ta"], "tau_d": fit["td"],
                 "item_rate_logit": _logit(min(0.99, max(0.01, item_rate))), "mean_labels": fit["N"] / float(fit["T"]),
                 "step_item": sj, "step_subj": ss, "corr_item": a_s * sj, "corr_subj": d_j * ss, "res_item": rj}
        x = [feats[name] for name in stack["features"]]
        p1 = min(0.995, max(0.005, _tree_sum(stack, x)))
        w = float(stack.get("w", 1.0))
        return float(w * p1 + (1.0 - w) * p0)
    except Exception:
        return float(p0)
