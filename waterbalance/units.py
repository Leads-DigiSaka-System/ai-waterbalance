"""Unit harmonization: every dataset's native units -> the units the water balance uses.

  water depths      mm (per day for fluxes)
  temperature       deg C
  pressure          kPa
  radiation         MJ m-2 day-1
  wind              m s-1 at 2 m
  soil fractions    0-1 (sand, clay); organic matter in % by weight
"""
from __future__ import annotations

import numpy as np


def k_to_c(t_k):
    return np.asarray(t_k, dtype=np.float32) - 273.15


def pa_to_kpa(p_pa):
    return np.asarray(p_pa, dtype=np.float32) / 1000.0


def j_to_mj(x_j):
    """J m-2 day-1 -> MJ m-2 day-1."""
    return np.asarray(x_j, dtype=np.float32) / 1e6


def m_to_mm(x_m):
    """m of water -> mm."""
    return np.asarray(x_m, dtype=np.float32) * 1000.0


def era5_evaporation_to_mm(e_m):
    """ERA5 total_evaporation (m, NEGATIVE = evaporation) -> ET mm, positive."""
    return -m_to_mm(e_m)


def wind_speed(u, v):
    return np.hypot(np.asarray(u, dtype=np.float32), np.asarray(v, dtype=np.float32))


def wind_to_2m(uz, z=10.0):
    """FAO-56 eq. 47: wind speed at z m -> 2 m (log profile)."""
    return np.asarray(uz, dtype=np.float32) * 4.87 / np.log(67.8 * z - 5.42)


def mm_per_hr_to_mm_per_day(rate):
    """Mean precipitation rate over a day (mm/hr, e.g. GPM IMERG) -> daily total (mm/day)."""
    return np.asarray(rate, dtype=np.float32) * 24.0


def percent_to_fraction(x_pct):
    return np.asarray(x_pct, dtype=np.float32) / 100.0


def olm_oc_to_om_pct(oc_5gkg):
    """OpenLandMap organic carbon (units of 5 g/kg) -> organic matter, % by weight
    (OM = 1.724 x OC, van Bemmelen)."""
    oc_g_kg = np.asarray(oc_5gkg, dtype=np.float32) * 5.0
    return oc_g_kg / 10.0 * 1.724


def theta_to_mm(theta, thickness_m):
    """Volumetric water content (m3 m-3) over a layer -> mm of water."""
    return np.asarray(theta, dtype=np.float32) * float(thickness_m) * 1000.0
