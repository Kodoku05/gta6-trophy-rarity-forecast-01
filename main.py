"""Deterministic cohort forecasts for Exophase trophy earn rates."""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from statistics import median, pstdev

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]
LAUNCH = datetime(2026, 11, 19, tzinfo=timezone.utc)
LAM_GRID = [0.05, 0.1, 0.2, 0.35, 0.5, 0.8, 1.2, 2, 3, 5, 10, 25, 60]
K_GRID = [0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.8, 1.2, 2]


def clamp(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    return math.log(clamp(pct) / (100 - clamp(pct)))


def sigmoid_logit(x):
    if x < 0:
        e = math.exp(x)
        return 100 * e / (1 + e)
    return 100 / (1 + math.exp(-x))


def round4(x):
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def hours(value):
    return (datetime.fromisoformat(value.replace("Z", "+00:00")) - LAUNCH).total_seconds() / 3600


def line(xs, ys):
    xm, ym = sum(xs) / len(xs), sum(ys) / len(ys)
    den = sum((x - xm) ** 2 for x in xs)
    b = sum((x - xm) * (y - ym) for x, y in zip(xs, ys)) / den if den else 0
    return ym - b * xm, b


def curve(states, family):
    ts = [max(0.001, s["t"]) for s in states]
    ys = [math.log(s["N"]) for s in states]
    if family == "power":
        a, b = line([math.log1p(t) for t in ts], ys)
        return lambda t: a + b * math.log1p(max(0, t))
    if family == "sat":
        candidates = []
        for k in K_GRID:
            shapes = [-math.expm1(-k * (t + 0.05)) for t in ts]
            ninf = sum(shape * state["N"] for shape, state in zip(shapes, states)) / sum(shape * shape for shape in shapes)
            if ninf <= max(state["N"] for state in states):
                continue
            err = sum((math.log(ninf * shape) - y) ** 2 for shape, y in zip(shapes, ys))
            candidates.append((err, k, ninf))
        if not candidates:
            return curve(states, "power")
        _, k, ninf = min(candidates)
        return lambda t: math.log(ninf * -math.expm1(-k * (max(0, t) + 0.05)))
    candidates = []
    for factor in [1.05, 1.2, 1.5, 2, 3, 5]:
        cap = factor * max(s["N"] for s in states)
        a, b = line([math.log(t) for t in ts], [math.log(cap / s["N"] - 1) for s in states])
        if b >= 0:
            continue
        def predict(t, a=a, b=b, cap=cap):
            v = a + b * math.log(max(0.001, t))
            return math.log(cap) - (max(0, v) + math.log1p(math.exp(-abs(v))))
        candidates.append((sum((predict(t) - y) ** 2 for t, y in zip(ts, ys)), predict))
    return min(candidates, key=lambda item: item[0])[1] if candidates else curve(states, "power")


def predictions(states, target):
    result = [curve(states, f)(target) for f in ["power", "sat", "logistic"]]
    recent = [s for s in states if s["t"] >= states[-1]["t"] - 24]
    if len(recent) >= 3:
        result.extend(curve(recent, f)(target) for f in ["power", "sat"])
    return result


def n_forecast(states, target, days):
    last_log = math.log(states[-1]["N"])
    if len(states) < 2:
        return states[-1]["N"], 0.9
    preds = predictions(states, target)
    errors = [[] for _ in preds]
    origins = []
    cutoff = states[0]["t"] + 0.4 * (states[-1]["t"] - states[0]["t"])
    for j in range(4, len(states) - 1):
        if states[j]["t"] < cutoff:
            continue
        dest = next((s for s in states[j + 1:] if s["t"] >= states[j]["t"] + 24), states[-1])
        ps = predictions(states[:j + 1], dest["t"])
        # Recent families use their full-history counterpart until available.
        ps = ps + ps[:2] if len(ps) == 3 and len(preds) == 5 else ps
        es = [p - math.log(dest["N"]) for p in ps[:len(preds)]]
        for bucket, e in zip(errors, es):
            bucket.append(e)
        origins.append((es, (dest["t"] - states[j]["t"]) / 24))
    weights = [1 / len(preds)] * len(preds)
    if len(origins) >= 2:
        raw = [1 / ((sum(abs(e) for e in es) / len(es)) ** 2 + 0.05 ** 2) for es in errors]
        weights = [0.1 / len(preds) + 0.9 * w / sum(raw) for w in raw]
        forecast = sum(w * p for w, p in zip(weights, preds))
    else:
        forecast = median(preds)
    if len(origins) >= 3:
        sigma = pstdev(sum(w * e for w, e in zip(weights, es)) * math.sqrt(days / max(h, 1e-6)) for es, h in origins)
    else:
        residual = max(abs(curve(states, f)(s["t"]) - math.log(s["N"])) for f in ["power", "sat", "logistic"] for s in states)
        sigma = math.sqrt(residual ** 2 + (pstdev(preds) / 2) ** 2 + 0.15 ** 2 * days)
    return math.exp(min(last_log + math.log(10), max(last_log, forecast))), max(0.1, sigma)


def cohort_rate(cohorts, population, speed, time):
    return sum(size * -math.expm1(-speed * max(0, time - joined) / 24) for joined, size in cohorts if joined <= time) / population


def forecast_case(case, name):
    if case["levels"] != LEVELS:
        raise ValueError(f"{name}: unexpected levels")
    if not case["history"]:
        raise ValueError(f"{name} has no history")
    states = []
    for read in case["history"]:
        state = {"t": hours(read["at"]), "N": read["players_tracked"], "rates": tuple(read["rates"]), "n_reads": 1}
        if states and (state["N"], state["rates"]) == (states[-1]["N"], states[-1]["rates"]):
            state["n_reads"] += states[-1]["n_reads"]
            states[-1] = state
        else:
            states.append(state)
    target = hours(case["target"])
    days = case["horizon_minutes"] / 1440
    population, n_sd = n_forecast(states, target, days)
    cohorts = [(states[0]["t"], states[0]["N"])]
    jumps = []
    for a, b in zip(states, states[1:]):
        if b["N"] > a["N"]:
            cohorts.append(((a["t"] + b["t"]) / 2, b["N"] - a["N"]))
            jumps.append(math.log(b["N"] / a["N"]))
    fits = []
    for i in range(len(case["trophies"])):
        observed = [s for s in states if s["rates"][i] is not None]
        if len(observed) < 2:
            fits.append(None)
            continue
        best = None
        for speed in LAM_GRID:
            gs = [cohort_rate(cohorts, s["N"], speed, s["t"]) for s in observed]
            den = sum(g * g for g in gs)
            share = min(1, max(0, sum(s["rates"][i] / 100 * g for s, g in zip(observed, gs)) / max(den, 1e-20)))
            error = sum((logit(s["rates"][i]) - logit(100 * share * g)) ** 2 for s, g in zip(observed, gs))
            candidate = (error, speed, share)
            if best is None or error < best[0]:
                best = candidate
        err, speed, share = best
        fits.append((speed, share, math.sqrt(err / max(1, len(observed) - 2)) if len(observed) > 2 else 0.3))
    previous = states[-1]["N"]
    for j in range(1, 9):
        end = states[-1]["t"] + (target - states[-1]["t"]) * j / 8
        start = states[-1]["t"] + (target - states[-1]["t"]) * (j - 1) / 8
        current = states[-1]["N"] * math.exp(math.log(population / states[-1]["N"]) * j / 8)
        cohorts.append(((start + end) / 2, current - previous))
        previous = current
    jump_mean, jump_sd, refresh = 0.7, 0.9, 6.0
    if len(jumps) >= 2:
        jump_mean, jump_sd = sum(jumps) / len(jumps), pstdev(jumps)
        refresh = (states[-1]["t"] - states[0]["t"]) / (len(states) - 1)
    shift = 0
    if len(states) >= 2:
        stale = jump_sd * 0.5 * min(1, math.sqrt(days))
    else:
        probability = -math.expm1(-days * 24 / refresh)
        shift = -probability * jump_mean
        stale = max(0.3, math.sqrt(probability * jump_sd ** 2 + probability * (1 - probability) * jump_mean ** 2))
    rows = []
    all_fits = [f for f in fits if f is not None]
    for i, trophy in enumerate(case["trophies"]):
        rates = [s["rates"][i] for s in states if s["rates"][i] is not None]
        last = rates[-1] if rates else 0
        center, fit_sd = last, 0.3
        if fits[i] is not None:
            peers = [f for f, t in zip(fits, case["trophies"]) if f is not None and t["tier"] == trophy["tier"]] or all_fits
            speed, share, fit_sd = fits[i]
            speed = math.exp(0.85 * math.log(speed) + 0.15 * median([math.log(f[0]) for f in peers]))
            share = sigmoid_logit(0.85 * logit(100 * share) + 0.15 * median([logit(100 * f[1]) for f in peers])) / 100
            center = 100 * share * cohort_rate(cohorts, population, speed, target)
        spread = max(0.2, math.sqrt(n_sd ** 2 + (1.75 * max(0.1, fit_sd)) ** 2 + stale ** 2))
        if last >= 99.9 and all(r >= 99 for r in rates):
            center, spread = 100, min(0.15, spread)
        rows.append({"id": trophy["id"], "q": [round4(sigmoid_logit(logit(center) + shift + z * spread)) for z in Z]})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        result = forecast_case(json.loads(path.read_text(encoding="utf-8")), path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(json.dumps(result, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
