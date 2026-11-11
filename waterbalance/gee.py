"""Earth Engine reads with computePixels -> numpy / xarray, tiled to stay under the request limits.

computePixels refuses a request above ~48 MB (and very many bands), so every read is split:

  space  the AOI is cut into tile_px x tile_px tiles (pipeline.py)
  time   each tile's daily stack is requested in chunks of days sized so that
         days x variables x height x width x 4 bytes <= max_request_mb and
         days x variables <= max_bands_per_request
  retry  a chunk that still fails with a size / memory / timeout error is halved and retried;
         rate-limit and transient server errors are retried with exponential backoff.

Earth Engine does the spatial harmonization on the server: every image is requested directly on
the harmonized grid (Grid.affine_dict), with
  * bilinear resampling for the continuous weather and soil fields
  * reduceResolution(mean) for the WorldCover cropland share of each 1 ha cell
Masked pixels come back as NODATA and are turned into NaN here.

The Fetcher interface (static / daily) is what harmonize.py consumes; tests use a fake one.
"""
from __future__ import annotations

import math
import re
import time

import numpy as np
import pandas as pd

from .config import DATASETS

# Masked pixels are unmasked to this value and turned back into NaN after reading. It must be far
# outside every variable's range: ERA5 net thermal radiation alone is about -4e6 J m-2 day-1, so a
# sentinel like -9999 would wipe it out (ET0 -> NaN -> no evapotranspiration).
NODATA = -3.0e38
NODATA_BELOW = -1.0e38
BYTES_PER_VALUE = 4                      # float32

_SIZE_ERRORS = ("request size", "must be less than or equal", "too large", "memory limit",
                "timed out", "too many pixels", "too many bands", "payload")
_TRANSIENT_ERRORS = ("429", "too many requests", "quota", "rate limit", "503", "502", "500",
                     "unavailable", "deadline", "internal error", "connection", "reset by peer")


class RequestTooLarge(RuntimeError):
    pass


def _is_size_error(e):
    m = str(e).lower()
    return any(s in m for s in _SIZE_ERRORS)


def _is_transient(e):
    m = str(e).lower()
    return any(s in m for s in _TRANSIENT_ERRORS)


# ------------------------------------------------------------------ chunk planning (pure)
def days_per_request(n_vars, height, width, max_request_mb=32.0, max_bands=256):
    """Days of an n_vars daily stack that fit one computePixels request."""
    per_day = n_vars * height * width * BYTES_PER_VALUE
    budget = max_request_mb * 1024 * 1024
    if per_day > budget:
        raise RequestTooLarge(
            f"one day of {n_vars} variable(s) on a {height}x{width} tile is "
            f"{per_day / 2**20:.1f} MB > {max_request_mb} MB: lower tile_px")
    return max(1, min(int(budget // per_day), max(1, max_bands // n_vars)))


def chunk_dates(start, end_excl, days):
    """[(chunk_start, chunk_end_excl)] covering [start, end_excl) in steps of `days`."""
    out, d = [], pd.Timestamp(start)
    end_excl = pd.Timestamp(end_excl)
    while d < end_excl:
        e = min(d + pd.Timedelta(days=days), end_excl)
        out.append((d, e))
        d = e
    return out


def fetch_split(fetch, start, end_excl, days, retries=4, sleep=time.sleep, log=print):
    """Call fetch(chunk_start, chunk_end_excl) over [start, end_excl) in chunks of `days`.
    A chunk failing with a size error is halved (down to one day); transient errors are retried
    with backoff. Returns the list of results in date order."""
    todo = list(reversed(chunk_dates(start, end_excl, days)))
    out = []
    while todo:
        a, b = todo.pop()
        for attempt in range(retries + 1):
            try:
                out.append(fetch(a, b))
                break
            except Exception as e:                     # noqa: BLE001 - classify below
                n = (b - a).days
                if _is_size_error(e) and n > 1:
                    mid = a + pd.Timedelta(days=n // 2)
                    log(f"    request too large ({n} days) — splitting")
                    todo += [(mid, b), (a, mid)]       # popped in date order
                    break
                if (_is_transient(e) or _is_size_error(e)) and attempt < retries:
                    sleep(2 ** attempt)
                    continue
                raise
    return out


# ------------------------------------------------------------------ Earth Engine fetcher
def initialize(project=None):
    import ee
    try:
        ee.Initialize(project=project)
    except Exception:                                  # noqa: BLE001
        ee.Authenticate()
        ee.Initialize(project=project)
    return ee


def structured_to_stack(arr, names):
    """computePixels NUMPY_NDARRAY (structured, one field per band) -> float32 (n, h, w),
    NODATA -> NaN."""
    out = np.stack([np.asarray(arr[n], dtype=np.float32) for n in names])
    out[out <= NODATA_BELOW] = np.nan
    return out


_BAND_RE = re.compile(r"^(\d{8})_(.+)$")


def parse_daily_bands(names):
    """['20240101_tmin', ...] -> list of (Timestamp, var)."""
    out = []
    for n in names:
        m = _BAND_RE.match(n)
        if not m:
            raise ValueError(f"unexpected band name {n!r} (want YYYYMMDD_var)")
        out.append((pd.Timestamp(m.group(1)), m.group(2)))
    return out


class EEFetcher:
    """Reads the datasets in config.DATASETS onto a Grid with ee.data.computePixels."""

    def __init__(self, cfg, log=print):
        self.cfg = cfg
        self.log = log
        self.ee = initialize(cfg.ee_project)

    # ---------------------------------------------------------- low level
    def _pixels(self, image, grid):
        ee = self.ee
        last = None
        for attempt in range(self.cfg.request_retries + 1):
            try:
                arr = ee.data.computePixels({
                    "expression": image,
                    "fileFormat": "NUMPY_NDARRAY",
                    "grid": grid.affine_dict(),
                })
                return arr
            except Exception as e:                     # noqa: BLE001
                last = e
                if _is_size_error(e):
                    raise
                if _is_transient(e) and attempt < self.cfg.request_retries:
                    time.sleep(2 ** attempt)
                    continue
                raise
        raise last

    # ---------------------------------------------------------- static layers
    def static(self, grid):
        """Raw static layers on the grid: crop_frac (0-1) and soil sand/clay/oc at each depth
        (native OpenLandMap units). -> dict name -> (h, w) float32."""
        ee = self.ee
        wc_cfg, soil = DATASETS["worldcover"], DATASETS["soil"]
        wc = ee.ImageCollection(wc_cfg["id"]).first().select(wc_cfg["band"])
        crop = (wc.eq(wc_cfg["cropland_class"]).unmask(0)
                .setDefaultProjection(wc.projection())
                .reduceResolution(reducer=ee.Reducer.mean(), maxPixels=1024)
                .rename("crop_frac"))
        img = crop
        names = ["crop_frac"]
        for var in ("sand", "clay", "oc"):
            src = [f"b{d}" for d in soil["depths_cm"]]
            dst = [f"{var}_{d}" for d in soil["depths_cm"]]
            img = img.addBands(ee.Image(soil[var]).select(src, dst).resample("bilinear"))
            names += dst
        img = img.toFloat().unmask(NODATA, False)
        arr = self._pixels(img, grid)
        stack = structured_to_stack(arr, names)
        return {n: stack[i] for i, n in enumerate(names)}

    # ---------------------------------------------------------- daily stacks
    def daily(self, source, start, end_excl, grid):
        """Daily stack of `source` ('gpm', 'chirps' or 'era5') on the grid for [start, end_excl),
        in the dataset's native units (GPM: the day's mean rate, mm/hr).
        -> (DatetimeIndex, dict var -> (t, h, w) float32). Days missing from the collection are
        absent from the index (harmonize.py reindexes them). GPM also returns '_provisional'
        (t,) bool: days with Late-run images or fewer than 48 half-hourly images."""
        if source == "gpm":
            return self._gpm_daily(start, end_excl, grid)
        ee = self.ee
        ds = DATASETS[source]
        bands = ds["bands"]
        src, dst = list(bands.values()), list(bands.keys())
        days = days_per_request(len(dst), grid.height, grid.width,
                                self.cfg.max_request_mb, self.cfg.max_bands_per_request)

        def fetch(a, b):
            col = (ee.ImageCollection(ds["id"])
                   .filterDate(a.strftime("%Y-%m-%d"), b.strftime("%Y-%m-%d"))
                   .select(src, dst))

            def prep(im):
                day = ee.Date(im.get("system:time_start")).format("YYYYMMdd")
                return (im.resample("bilinear").toFloat().unmask(NODATA, False)
                        .set("system:index", day))

            col = col.map(prep)
            names = col.aggregate_array("system:index").getInfo()
            if not names:
                return pd.DatetimeIndex([]), {v: np.empty((0, grid.height, grid.width),
                                                          np.float32) for v in dst}
            arr = self._pixels(col.toBands(), grid)
            fields = list(arr.dtype.names)
            parsed = parse_daily_bands(fields)
            dates = sorted({d for d, _ in parsed})
            stack = structured_to_stack(arr, fields)
            out = {v: np.full((len(dates), grid.height, grid.width), np.nan, np.float32)
                   for v in dst}
            pos = {d: i for i, d in enumerate(dates)}
            for k, (d, v) in enumerate(parsed):
                if v in out:
                    out[v][pos[d]] = stack[k]
            return pd.DatetimeIndex(dates), out

        parts = fetch_split(fetch, start, end_excl, days, self.cfg.request_retries, log=self.log)
        return merge_daily_parts(parts, dst, grid)

    def _gpm_daily(self, start, end_excl, grid):
        """GPM IMERG half-hourly -> one image per UTC day (mean rate, mm/hr), built on the server,
        then read like any daily stack."""
        ee = self.ee
        ds = DATASETS["gpm"]
        band = ds["bands"]["precip"]
        proj = ee.ImageCollection(ds["id"]).first().select(band).projection()
        days = days_per_request(1, grid.height, grid.width, self.cfg.max_request_mb,
                                self.cfg.max_bands_per_request)

        def fetch(a, b):
            n = (b - a).days
            col = ee.ImageCollection(ds["id"]).filterDate(a.strftime("%Y-%m-%d"),
                                                          b.strftime("%Y-%m-%d")).select(band)
            d0 = ee.Date(a.strftime("%Y-%m-%d"))

            def day_image(i):
                s = d0.advance(i, "day")
                c = col.filterDate(s, s.advance(1, "day"))
                img = (c.mean().rename("precip").setDefaultProjection(proj)
                       .resample("bilinear").toFloat().unmask(NODATA, False))
                return img.set({"system:index": s.format("YYYYMMdd"), "n": c.size(),
                                "n_prov": c.filter(ee.Filter.eq(ds["status_property"],
                                                                ds["provisional_value"])).size()})

            daily = ee.ImageCollection.fromImages(
                ee.List.sequence(0, n - 1).map(day_image)).filter(ee.Filter.gt("n", 0))
            info = ee.Dictionary({"names": daily.aggregate_array("system:index"),
                                  "n": daily.aggregate_array("n"),
                                  "n_prov": daily.aggregate_array("n_prov")}).getInfo()
            if not info["names"]:
                return pd.DatetimeIndex([]), {"precip": np.empty((0, grid.height, grid.width),
                                                                 np.float32),
                                              "_provisional": np.empty(0, bool)}
            arr = self._pixels(daily.select(["precip"]).toBands(), grid)
            fields = list(arr.dtype.names)
            parsed = parse_daily_bands(fields)
            stack = structured_to_stack(arr, fields)
            meta = {pd.Timestamp(k): (cnt, pv) for k, cnt, pv in
                    zip(info["names"], info["n"], info["n_prov"])}
            dates = [d for d, _ in parsed]
            prov = np.array([meta[d][1] > 0 or meta[d][0] < ds["slots_per_day"] for d in dates])
            return pd.DatetimeIndex(dates), {"precip": stack, "_provisional": prov}

        parts = fetch_split(fetch, start, end_excl, days, self.cfg.request_retries, log=self.log)
        return merge_daily_parts(parts, ["precip", "_provisional"], grid)


def merge_daily_parts(parts, vars_, grid):
    dates = [p[0] for p in parts if len(p[0])]
    if not dates:
        return pd.DatetimeIndex([]), {v: (np.empty(0, bool) if v.startswith("_") else
                                          np.empty((0, grid.height, grid.width), np.float32))
                                      for v in vars_}
    idx = pd.DatetimeIndex(np.concatenate([d.values for d in dates]))
    data = {v: np.concatenate([p[1][v] for p in parts if len(p[0])], axis=0) for v in vars_}
    return idx, data


def request_count(n_vars, n_days, grid, cfg):
    """Requests needed for one tile's daily stack (for plans / logs)."""
    d = days_per_request(n_vars, grid.height, grid.width, cfg.max_request_mb,
                         cfg.max_bands_per_request)
    return math.ceil(n_days / d)
