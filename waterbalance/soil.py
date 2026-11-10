"""Soil water limits (FC, WP, saturation) in mm over the root zone, and runoff curve numbers.

  theta at -33 kPa (FC), -1500 kPa (WP) and saturation: Saxton & Rawls (2006) pedotransfer
  functions from sand, clay (fractions) and organic matter (%).
  mm over the root zone: the profile values (given at depths 0, 10, 30, 60, 100, 200 cm) are
  integrated with the trapezoid rule from 0 to Zr.
"""
from __future__ import annotations

import numpy as np


def saxton_rawls(sand, clay, om_pct):
    """-> (theta_wp, theta_fc, theta_sat), m3 m-3. sand, clay: 0-1; om_pct: % (clipped to 0-8)."""
    S = np.clip(np.asarray(sand, dtype=np.float64), 0.0, 1.0)
    C = np.clip(np.asarray(clay, dtype=np.float64), 0.0, 1.0)
    OM = np.clip(np.asarray(om_pct, dtype=np.float64), 0.0, 8.0)

    t1500t = (-0.024 * S + 0.487 * C + 0.006 * OM + 0.005 * S * OM - 0.013 * C * OM
              + 0.068 * S * C + 0.031)
    wp = t1500t + (0.14 * t1500t - 0.02)

    t33t = (-0.251 * S + 0.195 * C + 0.011 * OM + 0.006 * S * OM - 0.027 * C * OM
            + 0.452 * S * C + 0.299)
    fc = t33t + (1.283 * t33t ** 2 - 0.374 * t33t - 0.015)

    ts33t = (0.278 * S + 0.034 * C + 0.022 * OM - 0.018 * S * OM - 0.027 * C * OM
             - 0.584 * S * C + 0.078)
    ts33 = ts33t + (0.636 * ts33t - 0.107)
    sat = fc + ts33 - 0.097 * S + 0.043

    wp = np.clip(wp, 0.01, 0.6)
    fc = np.maximum(np.clip(fc, 0.02, 0.7), wp + 0.01)
    sat = np.maximum(np.clip(sat, 0.2, 0.8), fc + 0.01)
    return wp.astype(np.float32), fc.astype(np.float32), sat.astype(np.float32)


def profile_to_mm(theta_by_depth, depths_cm, zr_m):
    """Integrate theta(z) from 0 to zr (trapezoid) -> mm.
    theta_by_depth: array (n_depths, ...) at depths_cm (ascending, first = 0)."""
    th = np.asarray(theta_by_depth, dtype=np.float64)
    z = np.asarray(depths_cm, dtype=np.float64) / 100.0
    zr = float(zr_m)
    if zr <= 0:
        raise ValueError("root depth must be > 0")
    if zr > z[-1]:
        raise ValueError(f"root depth {zr} m is below the deepest soil value ({z[-1]} m)")
    total = np.zeros(th.shape[1:])
    for i in range(len(z) - 1):
        top, bot = z[i], z[i + 1]
        if top >= zr:
            break
        b = min(bot, zr)
        th_b = th[i] + (th[i + 1] - th[i]) * (b - top) / (bot - top)
        total = total + (th[i] + th_b) / 2.0 * (b - top)
    return (total * 1000.0).astype(np.float32)


def hydrologic_group(sand, clay):
    """Texture-based hydrologic soil group, coded 1=A, 2=B, 3=C, 4=D (simplified USDA NEH-630):
    A sands / loamy sands (sand >= 70 %, clay < 10 %); B loams, sandy and silt loams (clay < 27 %);
    C clay loams (clay 27-40 %); D clays (clay >= 40 %)."""
    S = np.asarray(sand, dtype=np.float32)
    C = np.asarray(clay, dtype=np.float32)
    g = np.full(S.shape, 4, dtype=np.int8)
    g[C < 0.40] = 3
    g[C < 0.27] = 2
    g[(C < 0.10) & (S >= 0.70)] = 1
    return g


def curve_number(hsg, cn_by_hsg):
    """Curve number per pixel from the hydrologic group codes (1-4)."""
    lut = np.array([np.nan, cn_by_hsg["A"], cn_by_hsg["B"], cn_by_hsg["C"], cn_by_hsg["D"]],
                   dtype=np.float32)
    return lut[np.clip(np.asarray(hsg, dtype=np.int64), 0, 4)]
