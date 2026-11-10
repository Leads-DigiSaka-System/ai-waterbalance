"""Stage 2: daily root-zone water balance and water availability f.

Per pixel, per day (all mm):

  1. irrigation (irrigation='auto'): if f < 1 - p at the start of the day, I = FC - S
  2. runoff      R = SCS curve number runoff of P (I is applied to the field, not run off)
  3. infiltrate  S += P - R + I;  water above saturation runs off as well (R += S - SAT)
  4. drainage    D = tau * (S - FC) when S > FC
  5. ET          ETc = Ks * Kc * ET0, Ks = min(1, f / (1 - p)), limited to S - WP
  6. f           (S - WP) / (FC - WP)   at the end of the day

so  dS = P + I - R - ET - D  holds exactly every day and over every period
(the user's P - R - ET = dS, plus irrigation and deep drainage).

Per period:  sums of P, I, R, ET, D;  dS = S_end - S_start;  f at the period end, its minimum,
and the days with f < 1 - p (crop stress). Status class from f at the period end:

    4  surplus           f > 1          wetter than field capacity (draining / waterlogged)
    3  adequate          1 - p <= f <= 1
    2  moderate deficit  severe_f <= f < 1 - p
    1  severe deficit    f < severe_f
   -1  no data           not cropland / no soil data / outside the area
"""
from __future__ import annotations

import numpy as np

STATUS = {-1: "no data", 1: "severe deficit", 2: "moderate deficit", 3: "adequate", 4: "surplus"}
SURPLUS_EPS = 0.01


def scs_runoff(p, cn):
    """SCS-CN daily runoff, mm (Ia = 0.2 S)."""
    p = np.asarray(p, dtype=np.float64)
    s = 25400.0 / np.asarray(cn, dtype=np.float64) - 254.0
    ia = 0.2 * s
    q = np.where(p > ia, (p - ia) ** 2 / (p + 0.8 * s), 0.0)
    return q


def availability(s, fc, wp):
    return (s - wp) / np.maximum(fc - wp, 1e-6)


def classify(f, p, severe_f=0.25, valid=None):
    """Status codes (see module doc)."""
    f = np.asarray(f, dtype=np.float64)
    cls = np.full(f.shape, -1, dtype=np.int16)
    ok = np.isfinite(f) if valid is None else (np.isfinite(f) & valid)
    adequate_from = 1.0 - p
    cls[ok & (f < severe_f)] = 1
    cls[ok & (f >= severe_f) & (f < adequate_from)] = 2
    cls[ok & (f >= adequate_from)] = 3
    cls[ok & (f > 1.0 + SURPLUS_EPS)] = 4
    return cls


def run_bucket(P, ET0, FC, WP, SAT, CN, S0, kc, p, tau=0.5, irrigation="none", periods=()):
    """
    P, ET0     (t, h, w) mm/day
    FC, WP, SAT, CN, S0  (h, w)
    periods    [(label, i0, i1)]: day indices [i0, i1) to summarise
    -> (S_end (h, w), {label: dict of (h, w) arrays})
    """
    if irrigation not in ("none", "auto"):
        raise ValueError("irrigation must be 'none' or 'auto'")
    fc = FC.astype(np.float64)
    wp = WP.astype(np.float64)
    sat = np.maximum(SAT.astype(np.float64), fc)
    s = np.clip(S0.astype(np.float64), wp, sat)
    stress_f = 1.0 - p
    nt = P.shape[0]

    acc = {}
    for label, i0, i1 in periods:
        if not (0 <= i0 < i1 <= nt):
            raise ValueError(f"period {label} [{i0}, {i1}) outside 0..{nt}")
        z = np.zeros_like(fc)
        acc[label] = {"i0": i0, "i1": i1, "P": z.copy(), "I": z.copy(), "R": z.copy(),
                      "ET": z.copy(), "D": z.copy(), "stress_days": z.copy(),
                      "f_min": np.full_like(fc, np.inf), "S_start": None}

    for t in range(nt):
        for a in acc.values():
            if t == a["i0"]:
                a["S_start"] = s.copy()

        pt = np.nan_to_num(P[t].astype(np.float64), nan=0.0)
        e0 = np.nan_to_num(ET0[t].astype(np.float64), nan=0.0)

        irr = np.zeros_like(s)
        if irrigation == "auto":
            need = availability(s, fc, wp) < stress_f
            irr = np.where(need, np.maximum(fc - s, 0.0), 0.0)

        r = scs_runoff(pt, CN)
        s = s + pt - r + irr
        excess = np.maximum(s - sat, 0.0)
        r = r + excess
        s = s - excess

        d = np.where(s > fc, tau * (s - fc), 0.0)
        s = s - d

        f_now = availability(s, fc, wp)
        ks = np.clip(f_now / stress_f, 0.0, 1.0) if stress_f > 0 else np.ones_like(s)
        et = np.minimum(ks * kc * e0, np.maximum(s - wp, 0.0))
        s = s - et

        f = availability(s, fc, wp)
        for a in acc.values():
            if a["i0"] <= t < a["i1"]:
                a["P"] += pt
                a["I"] += irr
                a["R"] += r
                a["ET"] += et
                a["D"] += d
                a["stress_days"] += f < stress_f
                a["f_min"] = np.minimum(a["f_min"], f)
        for a in acc.values():
            if t == a["i1"] - 1:
                a["S_end"] = s.copy()
                a["f_end"] = f.copy()

    out = {}
    for label, a in acc.items():
        res = {k: a[k].astype(np.float32) for k in ("P", "I", "R", "ET", "D", "stress_days",
                                                    "f_min", "S_end", "f_end")}
        res["S_start"] = a["S_start"].astype(np.float32)
        res["dS"] = (a["S_end"] - a["S_start"]).astype(np.float32)
        res["residual"] = (a["P"] + a["I"] - a["R"] - a["ET"] - a["D"]
                           - (a["S_end"] - a["S_start"])).astype(np.float32)
        out[label] = res
    return s.astype(np.float32), out
