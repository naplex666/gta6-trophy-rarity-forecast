"""GTA VI trophy rarity forecaster (make.fun gta6-trophy-rarity-forecast-01).

Forecasts every PlayStation 5 trophy's Exophase earn rate at the case's
target, as five quantiles (0.05/0.25/0.5/0.75/0.95) per trophy. Scored by
mean pinball loss on the logit scale, so the median must be the honest
central forecast and the spread the honest uncertainty.

Model (standard library only, deterministic, no network):
  * Earners and players are modelled APART, each with a saturating
    (Michaelis-Menten) curve y(t) = y_inf * t / (t + k), fitted as the
    straight line 1/y against 1/t. The rate is their ratio. This matters
    because the two move on different clocks: early on the tracked player
    count explodes while earners grow slowly, so the displayed rate FALLS
    at refreshes; later the player count saturates and earners keep
    growing, so the rate RISES. Extrapolating the rate's own recent trend
    (as the house curve baseline does) carries the early denominator
    distortion 4.5 days forward and gets the direction wrong; the
    earners/players split does not. The earner clock k is pooled across
    trophies (a shared saturation timescale), each trophy keeping its own
    level. Flat history (the practice cases) fits nothing and falls back
    to persistence, the only honest median when no trend has been seen.
  * Spread has three parts, all in logit units: the fits' residual
    variance grown with the horizon, a refresh-shock term, and a
    model-misspecification term that grows with the horizon. Exophase
    refreshes its counts on its own schedule: a refresh jumps the tracked
    player count and usually DROPS every rate (the denominator grows faster
    than the earners), so refreshes are detected at runtime from
    players_tracked jumps and simultaneous rate drops; when no refresh has
    been seen yet, a prior shock of 0.7 logit applies (the documented first
    refresh dropped every rate but the one at 100%).
  * The spread is asymmetric: refresh shocks push rates down, not up, so
    the lower quantiles stretch further than the upper ones, in proportion
    to the shock's share of the total variance. At long horizons the trend
    dominates and the band is nearly symmetric again.
  * Any failure falls back to the house baseline (persistence with a
    0.5*sqrt(days) random-walk spread), so a case is always valid.

How the grader runs it: `python3 -P -s /bot/main.py` (PYTHONHASHSEED=0) in
python:3.13-slim, standard library only: no network, no pip packages.
"""

import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so `import model`
# finds a model.py shipped beside this file.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]
RELEASE = datetime.fromisoformat("2026-11-19T00:00:00+00:00")

# Spread calibration (logit units). Tuned to the contest's own facts: the
# house baseline scores ~1.51 on the practice because its 90-minute band is
# far too narrow for a refresh it never saw; a refresh that multiplies the
# tracked players ~6x moves small rates by roughly a logit.
SHOCK_PRIOR = 0.7       # refresh-shock sd when no refresh has been observed
SHOCK_FLOOR = 0.3       # observed refreshes never shrink the shock below this
SHOCK_CAP = 1.2
MISS_PER_SQRT_DAY = 0.35  # trend-shape (saturation, new cohorts) uncertainty
K_MIN, K_MAX = 1.0, 720.0  # saturation timescale bounds, hours
MOVE_MIN, MOVE_MAX = -2.5, 4.0  # clamp on the median's logit move over horizon


def clamp_pct(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    p = clamp_pct(pct)
    return math.log(p / (100.0 - p))


def sigmoid(lg: float) -> float:
    lg = max(-30.0, min(30.0, lg))
    return 100.0 / (1.0 + math.exp(-lg))


def stamp(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def round4(x: float):
    """x rounded to 4 decimals, half up, as the starter rounds it."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value == int(value) else value


def median(xs):
    xs = sorted(xs)
    n = len(xs)
    if n == 0:
        return 0.0
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def fit_line(xs, ys):
    """Least squares y = a + b*x. Returns (a, b, resid_sd, r2, se_b)."""
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return (my, 0.0, 0.0, 0.0, 0.0)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    ss_res = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - my) ** 2 for y in ys)
    resid_sd = math.sqrt(ss_res / max(1, n - 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    se_b = resid_sd / math.sqrt(sxx) if sxx > 0 else 0.0
    return (a, b, resid_sd, max(0.0, min(1.0, r2)), se_b)


def house_baseline(case: dict) -> dict:
    """The starter's forecast: persistence + 0.5*sqrt(days) spread."""
    last = case["history"][-1]
    spread = 0.5 * math.sqrt(case["horizon_minutes"] / 1440.0)
    rows = []
    for i, trophy in enumerate(case["trophies"]):
        rate = last["rates"][i]
        lg = logit(rate if rate is not None else 0.0)
        rows.append({"id": trophy["id"],
                     "q": [round4(sigmoid(lg + z * spread)) for z in Z]})
    return {"trophies": rows}


def forecast_case(case: dict, name: str = "") -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    history = case["history"]
    if not history:
        raise ValueError(f"{name} has no history")
    trophies = case["trophies"]
    horizon_days = case["horizon_minutes"] / 1440.0
    release = RELEASE
    try:
        release = datetime.fromisoformat(
            case["game"]["release_date"] + "T00:00:00+00:00")
    except Exception:
        pass
    target = stamp(case["target"])
    t_tgt = max(0.0, (target - release).total_seconds() / 3600.0)

    # ---- reads: (hours since release, players_tracked, raw %, logits) ----
    reads = []
    for read in history:
        t = (stamp(read["at"]) - release).total_seconds() / 3600.0
        raw = [r if r is not None else None for r in read["rates"]]
        lg = [logit(r) if r is not None else None for r in raw]
        reads.append((t, read.get("players_tracked"), raw, lg))
    t_last = reads[-1][0]
    last_rates = [logit(r) if r is not None else logit(0.0)
                  for r in history[-1]["rates"]]
    last_raw = [r if r is not None else 0.0 for r in history[-1]["rates"]]

    # ---- refresh detection: a players_tracked jump, or most trophies ----
    # ---- dropping together between two reads, is a refresh. Collect the
    # ---- per-trophy logit jumps at those reads as the shock sample. ----
    # ---- Only the last 24 h count: the launch-night explosions (the
    # ---- tracked count jumping 6x in hours) must not set the shock for
    # ---- a forecast days later, when refreshes are far smaller. ----
    jumps = []
    for (t0, p0, _r0, lg0), (t1, p1, _r1, lg1) in zip(reads, reads[1:]):
        players_jump = (p0 is not None and p1 is not None and p0 > 0
                        and p1 >= 1.10 * p0)
        drops = sum(1 for a, b in zip(lg0, lg1)
                    if a is not None and b is not None and b < a - 0.05)
        if players_jump or drops >= max(5, len(trophies) // 2):
            for a, b in zip(lg0, lg1):
                if a is not None and b is not None:
                    jumps.append((t1, b - a))
    recent_jumps = [j for (t, j) in jumps if t >= t_last - 24.0]
    if not recent_jumps:
        recent_jumps = [j for (_t, j) in jumps]
    if recent_jumps:
        # Robust sd of the refresh moves: median |jump| / 0.674.
        shock = median([abs(j) for j in recent_jumps]) / 0.674
        shock = max(SHOCK_FLOOR, min(SHOCK_CAP, shock))
    else:
        shock = SHOCK_PRIOR

    # ---- distinct refresh points: consecutive reads with identical ----
    # ---- (players, rates) are one refresh; collapse them so stale ----
    # ---- repeats do not dominate the fits. ----
    distinct = []
    for t, p, raw, lg in reads:
        key = (p, tuple(raw))
        if not distinct or distinct[-1][3] != key:
            distinct.append((t, p, raw, key))
    pts = [(t, p, raw) for (t, p, raw, _k) in distinct]
    recent = [pt for pt in pts if pt[0] >= max(1.0, t_last - 24.0)]

    def mm_k(points):
        """Saturation timescale k of y(t) = y_inf*t/(t+k), from (t, y)
        points via the straight line 1/y against 1/t; None if unidentified
        (flat or too few points)."""
        xs, ys = [], []
        for t, y in points:
            if t >= 1.0 and y is not None and y > 0:
                xs.append(1.0 / t)
                ys.append(1.0 / y)
        fit = fit_line(xs, ys)
        if fit is None:
            return None
        a, b, _sd, _r2, _se = fit
        if a <= 0 or b <= 0:
            return None
        return max(K_MIN, min(K_MAX, b / a))

    def mm_level(points, k):
        """y_inf from (t, y) points with k fixed: median of y*(t+k)/t."""
        vals = [y * (t + k) / t for (t, y) in points
                if t >= 1.0 and y is not None and y > 0]
        return median(vals) if vals else None

    # ---- players at the target: one saturating fit for the case. ----
    p_last = reads[-1][1] or 0
    k_p = mm_k([(t, p) for (t, p, _r) in pts])
    if k_p is None:
        p_inf, p_tgt = p_last, p_last
    else:
        p_inf = mm_level([(t, p) for (t, p, _r) in recent], k_p)
        p_inf = p_inf if p_inf else p_last
        p_tgt = p_inf * t_tgt / (t_tgt + k_p) if t_tgt > 0 else p_inf
    p_tgt = max(p_last, p_tgt) if p_last else p_tgt

    # ---- earners at the target: shared saturation clock k_E (median ----
    # ---- across trophies), each trophy its own level. ----
    ks = []
    for i in range(len(trophies)):
        e_pts = [(t, (raw[i] / 100.0 * p if raw[i] is not None and p else None))
                 for (t, p, raw) in pts]
        k_i = mm_k(e_pts)
        if k_i is not None:
            ks.append(k_i)
    k_e = median(ks) if ks else None

    medians = []
    resid_by_trophy = [None] * len(trophies)
    for i in range(len(trophies)):
        last_lg = last_rates[i]
        if k_e is None or p_tgt <= 0:
            medians.append(last_lg)
            continue
        e_recent = [(t, (raw[i] / 100.0 * p if raw[i] is not None and p else None))
                    for (t, p, raw) in recent]
        e_inf = mm_level(e_recent, k_e)
        if e_inf is None:
            # No usable recent earner count: persist the last rate.
            medians.append(last_lg)
            continue
        e_tgt = e_inf * t_tgt / (t_tgt + k_e) if t_tgt > 0 else e_inf
        rate_tgt = 100.0 * e_tgt / p_tgt
        med_lg = logit(rate_tgt)
        # Blend with persistence in logit space: the earners/players split
        # is the right structure, but any two-parameter curve fitted on ~2
        # days of data and pushed 4.5 days out carries shape error, and
        # the split's median runs hot (it understates how fast the player
        # count keeps growing). A 50/50 blend halves the worst-case median
        # error in either direction; persistence alone is the fallback
        # when the split says nothing.
        med_lg = 0.5 * med_lg + 0.5 * last_lg
        # Never let the median run away from the last seen rate.
        med_lg = max(last_lg + MOVE_MIN, min(last_lg + MOVE_MAX, med_lg))
        medians.append(med_lg)
        # Residuals of this trophy's observed rates against the E/P model,
        # for the spread: how well the split explains what was seen.
        res = []
        for (t, p, raw) in recent:
            if raw[i] is None or not p or t < 1.0:
                continue
            e_mod = e_inf * t / (t + k_e)
            p_mod = p_inf * t / (t + k_p) if k_p is not None else p
            if p_mod <= 0:
                continue
            res.append(logit(raw[i]) - logit(100.0 * e_mod / p_mod))
        if len(res) >= 3:
            m_res = sum(res) / len(res)
            resid_by_trophy[i] = math.sqrt(
                sum((r - m_res) ** 2 for r in res) / len(res))
    pool_sd = median([r for r in resid_by_trophy if r is not None]) \
        if any(r is not None for r in resid_by_trophy) else 0.15

    # ---- spread: residual variance grown with horizon + refresh shock ----
    # ---- + trend-shape misspecification; asymmetric (down wider). ----
    rows = []
    for i, trophy in enumerate(case["trophies"]):
        # A trophy whose own fit left no residual borrows the pooled scale.
        s_i = resid_by_trophy[i] if resid_by_trophy[i] is not None else pool_sd
        s_i = max(0.12, s_i)
        trend_var = (s_i * math.sqrt(1.0 + horizon_days)) ** 2
        miss_var = (MISS_PER_SQRT_DAY * math.sqrt(horizon_days)) ** 2
        sigma = math.sqrt(trend_var + shock * shock + miss_var)
        share = (shock * shock) / (sigma * sigma) if sigma > 0 else 0.0
        lower_mult = 1.0 + 0.35 * share
        upper_mult = 1.0 - 0.15 * share
        med_lg = medians[i]
        q = []
        for z in Z:
            mult = lower_mult if z < 0 else (upper_mult if z > 0 else 1.0)
            q.append(round4(sigmoid(med_lg + z * sigma * mult)))
        q = [max(0.0, min(100.0, v)) for v in q]
        q = sorted(q)  # rounding can never break the order, but stay safe
        rows.append({"id": trophy["id"], "q": q})
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = {}
        try:
            case = json.loads(path.read_text(encoding="utf-8"))
            answer = forecast_case(case, path.name)
            assert len(answer["trophies"]) == len(case["trophies"])
            for row in answer["trophies"]:
                assert len(row["q"]) == 5 and row["q"] == sorted(row["q"])
                assert all(math.isfinite(x) and 0 <= x <= 100 for x in row["q"])
        except Exception:
            try:
                answer = house_baseline(case)
            except Exception:
                answer = {"trophies": [
                    {"id": t["id"], "q": [0.01, 0.1, 1.0, 5.0, 20.0]}
                    for t in case.get("trophies", [])]}
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(answer, separators=(",", ":"), allow_nan=False) + "\n",
            encoding="utf-8")


if __name__ == "__main__":
    main()
