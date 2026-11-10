"""Disk storage: harmonized-tile cache (.npz, no netCDF backend needed), soil-state files, and the
tile-by-tile GeoTIFF writer (finished as a Cloud-Optimized GeoTIFF)."""
from __future__ import annotations

import json
import os
import threading

import numpy as np
import xarray as xr


# ------------------------------------------------------------------ dataset <-> npz
def save_dataset(path, ds):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    arrays = {f"var__{v}": ds[v].values for v in ds.data_vars}
    arrays.update({f"coord__{c}": ds[c].values for c in ds.coords})
    meta = {"dims": {v: list(ds[v].dims) for v in ds.data_vars},
            "units": {v: ds[v].attrs.get("units") for v in ds.data_vars},
            "attrs": {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in ds.attrs.items()}}
    arrays["meta"] = np.array(json.dumps(meta, default=str))
    tmp = path + ".part.npz"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def save_skip(path, reason):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".part.npz"
    np.savez_compressed(tmp, meta=np.array(json.dumps({"skip": reason})))
    os.replace(tmp, path)


def load_dataset(path):
    """Dataset, or None for a skip marker."""
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["meta"]))
        if "skip" in meta:
            return None
        coords = {k[len("coord__"):]: z[k] for k in z.files if k.startswith("coord__")}
        data = {k[len("var__"):]: (meta["dims"][k[len("var__"):]], z[k])
                for k in z.files if k.startswith("var__")}
    ds = xr.Dataset(data, coords=coords)
    ds.attrs.update(meta["attrs"])
    for v, unit in meta.get("units", {}).items():
        if unit and v in ds:
            ds[v].attrs["units"] = unit
    return ds


# ------------------------------------------------------------------ soil state
def save_state(path, s):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".part.npy"
    np.save(tmp, np.asarray(s, dtype=np.float32))
    os.replace(tmp, path)


def load_state(path):
    return np.load(path) if os.path.exists(path) else None


# ------------------------------------------------------------------ raster writer
class RasterWriter:
    """GeoTIFF over a Grid, filled tile by tile (sparse until written), then turned into a COG."""

    def __init__(self, path, grid, band_names, dtype="float32", nodata=np.nan, tags=None):
        import rasterio
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.path, self.grid, self.bands = path, grid, list(band_names)
        self.tmp = path + ".part.tif"
        self.dtype, self.nodata = np.dtype(dtype), nodata
        profile = dict(driver="GTiff", width=grid.width, height=grid.height,
                       count=len(self.bands), dtype=self.dtype.name, crs=grid.crs,
                       transform=grid.transform, nodata=nodata, tiled=True,
                       blockxsize=256, blockysize=256, sparse_ok=True, bigtiff="IF_SAFER")
        self._ds = rasterio.open(self.tmp, "w+", **profile)
        for i, n in enumerate(self.bands, 1):
            self._ds.set_band_description(i, n)
        if tags:
            self._ds.update_tags(**{k: str(v) for k, v in tags.items()})
        self._lock = threading.Lock()

    def write(self, tile_grid, arrays):
        """arrays: dict band -> (h, w) on tile_grid."""
        from rasterio.windows import Window
        col, row = tile_grid.offset_in(self.grid)
        win = Window(col, row, tile_grid.width, tile_grid.height)
        with self._lock:
            for i, n in enumerate(self.bands, 1):
                self._ds.write(np.asarray(arrays[n]).astype(self.dtype), i, window=win)

    def close(self, cog=True):
        import rasterio
        from rasterio.shutil import copy as rio_copy
        self._ds.close()
        if not cog:
            os.replace(self.tmp, self.path)
            return self.path
        try:
            with rasterio.Env():
                rio_copy(self.tmp, self.path, driver="COG", compress="DEFLATE",
                         overview_resampling="nearest", BIGTIFF="IF_SAFER")
            os.remove(self.tmp)
        except Exception:                               # noqa: BLE001 - GDAL without COG driver
            os.replace(self.tmp, self.path)
        return self.path
