"""Shared-label IRT predictor (pure Python).

Fits, for the target anonymous benchmark, logit p = c + ability(subject) + difficulty(item)
by penalised Newton steps on every revealed label for that benchmark (any subject).
The two prior scales (subject, item) are re-estimated from the fit itself (empirical Bayes),
starting from 1.25 / 2.0 and clipped to [0.6, 3.0].
With no labels the prediction is 0.5. If only the target subject's own labels are present
it reduces to a smoothed success rate. The fit is cached per (benchmark, number of labels).
"""

import json
import math

TAU_A0, TAU_D0, TAU_C, ITERS = 1.25, 2.0, 1.5, 24
TAU_LO, TAU_HI = 0.6, 3.0
_CACHE = {}


def _sigmoid(x):
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def _key(obj):
    return json.dumps(obj, sort_keys=True, default=str)


def _fit(entries):
    """entries: list of (subject_key, item_key, y). Returns (c, ability, difficulty)."""
    s_ids, j_ids = {}, {}
    s_idx = [s_ids.setdefault(s, len(s_ids)) for s, _, _ in entries]
    j_idx = [j_ids.setdefault(j, len(j_ids)) for _, j, _ in entries]
    y = [v for _, _, v in entries]
    nS, nJ, n = len(s_ids), len(j_ids), len(y)
    c, a, d = 0.0, [0.0] * nS, [0.0] * nJ
    ta, td = TAU_A0, TAU_D0
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
    return c, dict(zip(s_ids, a)), dict(zip(j_ids, d))


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
    c, ability, difficulty = fit
    value = _sigmoid(c + ability.get(_key(subject), 0.0) + difficulty.get(_key(item), 0.0))
    return float(min(0.98, max(0.02, value)))
