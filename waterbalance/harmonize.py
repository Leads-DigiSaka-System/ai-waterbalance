"""Stage 1: harmonize every input onto the 1 ha grid, in water-balance units.

One harmonized tile = an xarray Dataset on the tile's Grid:

  static (y, x)
    crop_frac   share of the 1 ha cell that is WorldCover cropland (0-1)
    crop_mask   1 where crop_frac >= crop_fraction_min (and soil data exists), else 0
    FC, WP, SAT soil water at field capacity / wilting point / saturation over the root zone, mm
    TAW         FC - WP, mm
    hsg, CN     hydrologic soil group (1-4 = A-D) and runoff curve number
  daily (time, y, x)
    P           precipitation, mm/day (CHIRPS; ERA5-Land on days CHIRPS lacks)
    ET0         FAO-56 reference ET from ERA5-Land, mm/day
  daily (time)
    p_fallback  1 on days whose P came from ERA5-Land (provisional), else 0

Steps: fetch raw (native units, already on the grid) -> convert units -> fill small coastal gaps
(nearest valid pixel) -> derive (ET0, soil limits, CN).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from . import et0 as _et0
from . import soil as _soil
from . import units as u
from .config import DATASETS


class DataNotReady(RuntimeError):
    """A dataset does not reach the end of the requested window yet (latency)."""


# ------------------------------------------------------------------ helpers
def fill_nearest(arr):
    """Fill NaN pixels from the nearest valid pixel, per (y, x) plane. Works on (h, w) or
    (t, h, w) (one spatial pattern from all days; days with no data at all stay NaN)."""
    from scipy.ndimage import distance_transform_edt
    a = np.asarray(arr, dtype=np.float32)
    plane = np.isfinite(a) if a.ndim == 2 else np.isfinite(a).any(axis=0)
    if plane.all() or not plane.any():
        return a
    _, (iy, ix) = distance_transform_edt(~plane, return_indices=True)
    return a[iy, ix] if a.ndim == 2 else a[:, iy, ix]


def _reindex_days(dates, data, full):
    """(dates, {var: (t,h,w)}) -> same on the full daily index (missing days NaN)."""
    pos = {d: i for i, d in enumerate(pd.DatetimeIndex(dates))}
    out = {}
    for v, arr in data.items():
        a = np.full((len(full),) + arr.shape[1:], np.nan, np.float32)
        for k, d in enumerate(full):
            if d in pos:
                a[k] = arr[pos[d]]
        out[v] = a
    return out


def _day_missing(a):
    return ~np.isfinite(a).any(axis=(1, 2))


# ------------------------------------------------------------------ static layers
def harmonize_static(raw, cfg):
    """raw: dict from Fetcher.static -> dict of (h, w) arrays (see module doc)."""
    depths = DATASETS["soil"]["depths_cm"]
    crop = cfg.crop_params()

    crop_frac = np.clip(np.nan_to_num(raw["crop_frac"], nan=0.0), 0.0, 1.0).astype(np.float32)
    sand = np.stack([fill_nearest(u.percent_to_fraction(raw[f"sand_{d}"])) for d in depths])
    clay = np.stack([fill_nearest(u.percent_to_fraction(raw[f"clay_{d}"])) for d in depths])
    om = np.stack([fill_nearest(u.olm_oc_to_om_pct(raw[f"oc_{d}"])) for d in depths])

    wp_t, fc_t, sat_t = _soil.saxton_rawls(sand, clay, om)
    fc = _soil.profile_to_mm(fc_t, depths, crop["zr"])
    wp = _soil.profile_to_mm(wp_t, depths, crop["zr"])
    sat = _soil.profile_to_mm(sat_t, depths, crop["zr"])

    # texture of the root zone for the hydrologic group: top-layer values
    hsg = _soil.hydrologic_group(sand[0], clay[0])
    cn = _soil.curve_number(hsg, cfg.cn_by_hsg)

    soil_ok = np.isfinite(fc) & np.isfinite(wp) & (fc > wp)
    mask = (crop_frac >= cfg.crop_fraction_min) & soil_ok
    return {
        "crop_frac": crop_frac,
        "crop_mask": mask.astype(np.uint8),
        "FC": fc, "WP": wp, "SAT": sat,
        "TAW": (fc - wp).astype(np.float32),
        "hsg": hsg, "CN": cn,
    }


# ------------------------------------------------------------------ daily layers
def harmonize_daily(chirps, era5, start, end_excl, cfg, final=True, prev=None):
    """chirps / era5: (DatetimeIndex, {var: (t,h,w)}) from Fetcher.daily, native units.
    final : this chunk ends the window — ERA5 days missing at its end mean the data is not
            published yet (DataNotReady). In earlier chunks they are carried forward.
    prev  : ERA5 harmonized values of the day before the chunk (to carry into a leading gap).
    -> (DatetimeIndex full daily, P mm, ET0 mm, p_fallback (t,) uint8, last-day ERA5 values)."""
    full = pd.date_range(start, pd.Timestamp(end_excl) - pd.Timedelta(days=1), freq="D")
    e = _reindex_days(*era5, full)
    c = _reindex_days(*chirps, full)

    # ---- ERA5-Land: units
    tmin = u.k_to_c(e["tmin"])
    tmax = u.k_to_c(e["tmax"])
    tdew = u.k_to_c(e["tdew"])
    u2 = u.wind_to_2m(u.wind_speed(e["u10"], e["v10"]), z=10.0)
    rn = u.j_to_mj(e["rns"]) + u.j_to_mj(e["rnl"])
    pk = u.pa_to_kpa(e["psurf"])
    tp = u.m_to_mm(e["tp"])

    arrays = [tmin, tmax, tdew, u2, rn, pk, tp]
    missing_e = _day_missing(tmin)
    if missing_e.all() and prev is None:
        raise DataNotReady(f"ERA5-Land has no data for {full[0].date()}..{full[-1].date()}")
    if final and missing_e[-1]:
        last_ok = full[~missing_e][-1] if (~missing_e).any() else None
        raise DataNotReady(f"ERA5-Land has no data after "
                           f"{last_ok.date() if last_ok is not None else full[0].date()}"
                           f" (window ends {full[-1].date()})")
    if missing_e.any():
        # gaps: carry the previous day forward (first day: the day before the chunk, or the
        # first available day)
        first_ok = int(np.argmax(~missing_e)) if (~missing_e).any() else None
        for k in np.where(missing_e)[0]:
            for i, a in enumerate(arrays):
                if k > 0:
                    a[k] = a[k - 1]
                elif prev is not None:
                    a[k] = prev[i]
                else:
                    a[k] = a[first_ok]

    tmin, tmax, tdew, u2, rn, pk, tp = (fill_nearest(a) for a in arrays)
    last = [a[-1].copy() for a in (tmin, tmax, tdew, u2, rn, pk, tp)]
    et0 = _et0.et0_from_era5(tmin, tmax, tdew, u2, rn, pk)

    # ---- CHIRPS (mm/day already), ERA5 fallback on missing days
    p = np.asarray(c["precip"], dtype=np.float32)
    p = fill_nearest(p)
    missing_c = _day_missing(p)
    if missing_c.any():
        if cfg.precip_fallback != "era5":
            raise DataNotReady(f"CHIRPS missing {int(missing_c.sum())} day(s) in "
                               f"{full[0].date()}..{full[-1].date()}")
        p[missing_c] = np.maximum(tp[missing_c], 0.0)
    p = np.maximum(p, 0.0)
    return full, p.astype(np.float32), et0, missing_c.astype(np.uint8), last


# ------------------------------------------------------------------ tile
HARMONIZE_CHUNK_DAYS = 90      # raw ERA5 (9 variables) is converted chunk by chunk to bound memory


def harmonize_tile(fetcher, grid, start, end_excl, cfg, static=None, log=print,
                   chunk_days=HARMONIZE_CHUNK_DAYS):
    """Fetch + harmonize one tile. Returns an xarray Dataset, or None when the tile has no
    cropland (the daily data is then never downloaded). `static` reuses harmonize_static output.
    The window is processed in chunks of chunk_days, so only P and ET0 are kept for the whole
    window, never the raw weather variables."""
    from .gee import chunk_dates
    if static is None:
        static = harmonize_static(fetcher.static(grid), cfg)
    if not static["crop_mask"].any():
        return None
    parts, prev = [], None
    chunks = chunk_dates(start, end_excl, chunk_days)
    for k, (a, b) in enumerate(chunks):
        era5 = fetcher.daily("era5", a, b, grid)
        chirps = fetcher.daily("chirps", a, b, grid)
        full, p, et0, fb, prev = harmonize_daily(chirps, era5, a, b, cfg,
                                                 final=k == len(chunks) - 1, prev=prev)
        parts.append((full, p, et0, fb))
    full = pd.DatetimeIndex(np.concatenate([x[0].values for x in parts]))
    p = np.concatenate([x[1] for x in parts])
    et0 = np.concatenate([x[2] for x in parts])
    fb = np.concatenate([x[3] for x in parts])
    return to_dataset(grid, static, full, p, et0, fb)


def to_dataset(grid, static, full, p, et0, fb):
    coords = {"time": full, "y": grid.ys, "x": grid.xs}
    data = {k: (("y", "x"), v) for k, v in static.items()}
    data.update({"P": (("time", "y", "x"), p), "ET0": (("time", "y", "x"), et0),
                 "p_fallback": (("time",), fb)})
    ds = xr.Dataset(data, coords=coords)
    ds.attrs.update({"crs": grid.crs, "res": grid.res, "x0": grid.x0, "y1": grid.y1,
                     "units": "mm (P, ET0 per day; FC, WP, SAT, TAW over the root zone)"})
    for v, unit in (("P", "mm/day"), ("ET0", "mm/day"), ("FC", "mm"), ("WP", "mm"),
                    ("SAT", "mm"), ("TAW", "mm"), ("crop_frac", "1")):
        ds[v].attrs["units"] = unit
    return ds
