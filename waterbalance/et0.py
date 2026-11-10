"""FAO-56 Penman-Monteith reference evapotranspiration (daily, mm/day).

Inputs are already harmonized (deg C, kPa, MJ m-2 day-1, m s-1 at 2 m). Net radiation comes straight
from ERA5-Land (net solar + net thermal), so no clear-sky radiation or elevation is needed; soil heat
flux G = 0 at the daily step (FAO-56 eq. 42).
"""
from __future__ import annotations

import numpy as np


def sat_vp(t_c):
    """Saturation vapour pressure, kPa (FAO-56 eq. 11)."""
    t = np.asarray(t_c, dtype=np.float64)
    return 0.6108 * np.exp(17.27 * t / (t + 237.3))


def et0_pm(tmax_c, tmin_c, ea_kpa, u2, rn_mj, p_kpa, g_mj=0.0):
    """FAO-56 eq. 6 with es from Tmax/Tmin (eq. 12) and ea given. Returns float32, >= 0."""
    tmax = np.asarray(tmax_c, dtype=np.float64)
    tmin = np.asarray(tmin_c, dtype=np.float64)
    tmean = (tmax + tmin) / 2.0
    delta = 4098.0 * sat_vp(tmean) / (tmean + 237.3) ** 2         # eq. 13
    gamma = 0.000665 * np.asarray(p_kpa, dtype=np.float64)       # eq. 8
    es = (sat_vp(tmax) + sat_vp(tmin)) / 2.0
    vpd = np.maximum(es - np.asarray(ea_kpa, dtype=np.float64), 0.0)
    u2 = np.asarray(u2, dtype=np.float64)
    num = 0.408 * delta * (np.asarray(rn_mj, dtype=np.float64) - g_mj) \
        + gamma * (900.0 / (tmean + 273.0)) * u2 * vpd
    den = delta + gamma * (1.0 + 0.34 * u2)
    return np.maximum(num / den, 0.0).astype(np.float32)


def et0_from_era5(tmin_c, tmax_c, tdew_c, u2, rn_mj, p_kpa):
    """ET0 with actual vapour pressure from the dew point (FAO-56 eq. 14)."""
    return et0_pm(tmax_c, tmin_c, sat_vp(tdew_c), u2, rn_mj, p_kpa)
