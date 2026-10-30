"""Model-prior shrinkage predictor.

Prior probability from a logit model (intercept + ability of the named model fitted
on the training data), then shrunk toward the subject's own acquired labels on the
same anonymous benchmark. The shrinkage strength and the prior scale are tuned per
label budget by leave-benchmark-out cross-validation. No external calls.
"""

import json
import math
import os

_PARAMS = None


def _params():
    global _PARAMS
    if _PARAMS is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_params.json")
        with open(path, encoding="utf-8") as handle:
            _PARAMS = json.load(handle)
    return _PARAMS


def _sigmoid(x):
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def predict(input: list[dict], labeled: list | None = None) -> float:
    subject, item = input
    params = _params()
    name = str(subject.get("normalized_name") or "").strip()
    prior_logit = params["icpt"] + params["ability"].get(name, 0.0)
    prob_model = _sigmoid(prior_logit)

    benchmark = item.get("benchmark_id")
    successes = 0
    count = 0
    for entry in labeled or []:
        (observed_subject, observed_item), label = entry
        if observed_subject != subject:
            continue
        if observed_item.get("benchmark_id") != benchmark:
            continue
        if label not in (0, 1):
            continue
        successes += int(label)
        count += 1

    budgets = params["budgets"]
    index = 0
    for i, budget in enumerate(budgets):
        if budget <= count:
            index = i
    m, lam = params["m"][index], params["lam"][index]
    prior = 0.5 + lam * (prob_model - 0.5)
    if count == 0:
        value = prior
    else:
        value = (successes + m * prior) / (count + m)
    return float(min(1.0, max(0.0, value)))
