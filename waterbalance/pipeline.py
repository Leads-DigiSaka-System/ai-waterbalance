"""End-to-end runs: AOI -> tiles -> stage 1 (harmonize) -> stage 2 (water balance, f) -> maps.

  Runner.plan(periods)         tiles, requests and dates, nothing downloaded
  Runner.harmonize(periods)    stage 1 only: harmonized 1 ha tiles cached under work_dir
  Runner.run(periods)          stage 1 (from cache when present) + stage 2, writes the outputs

Soil state carries over between runs: the root-zone storage S at each period's end is saved per
tile, so the next 14-day run starts from it and needs only that period's data. Without a saved
state (first run, or settings changed), the tile is spun up over `spinup_days` before the period,
starting from field capacity.

Outputs per period, under output_dir/<AOI>/<period>/:
  wa_<AOI>_<period>_status.tif    int16  status class (-1 no data, 1 severe, 2 moderate,
                                         3 adequate, 4 surplus)
  wa_<AOI>_<period>_balance.tif   float32 bands: f_end, f_min, stress_days, P, I, R, ET, D, dS,
                                         S_end, TAW, crop_frac, residual (mm unless noted)
  wa_<AOI>_<period>_summary.csv   hectares per status class
  wa_<AOI>_<period>_stats.csv     crop-area-weighted means of every balance term
  wa_<AOI>_<period>_run.json      settings, tiles, failures, provisional-precipitation days
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

from . import balance as bal
from . import gee
from . import harmonize as hz
from . import periods as per
from . import store
from .config import DATASETS, RunConfig
from .grid import Grid, geometry_mask, lonlat_to_crs, res_in_crs

BALANCE_BANDS = ["f_end", "f_min", "stress_days", "P", "I", "R", "ET", "D", "dS", "S_end",
                 "TAW", "crop_frac", "residual"]


def _h(obj, n=10):
    return hashlib.md5(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:n]


class Runner:
    def __init__(self, cfg: RunConfig, fetcher=None, log=print):
        self.cfg = cfg
        self.log = log
        self._fetcher = fetcher
        self._aoi = None

    # ---------------------------------------------------------- inputs
    @property
    def fetcher(self):
        if self._fetcher is None:
            self._fetcher = gee.EEFetcher(self.cfg, log=self.log)
        return self._fetcher

    def aoi(self):
        """(geometry in EPSG:4326, label)."""
        if self._aoi is not None:
            return self._aoi
        cfg = self.cfg
        from shapely.geometry import box
        if cfg.bbox:
            geom, label = box(*cfg.bbox), cfg.aoi_label or "AOI"
        elif cfg.aoi_path:
            import geopandas as gpd
            gdf = gpd.read_file(cfg.aoi_path).to_crs("EPSG:4326")
            if cfg.aoi_names:
                if not cfg.aoi_name_col:
                    raise ValueError("aoi_names needs aoi_name_col")
                want = {str(n).strip().upper() for n in cfg.aoi_names}
                col = gdf[cfg.aoi_name_col].astype(str).str.strip().str.upper()
                gdf = gdf[col.isin(want)]
                found = set(col[col.isin(want)])
                if want - found:
                    raise ValueError(f"not found in {cfg.aoi_name_col}: {sorted(want - found)}")
            if gdf.empty:
                raise ValueError("the AOI selection is empty")
            geom = gdf.geometry.union_all() if hasattr(gdf.geometry, "union_all") \
                else gdf.geometry.unary_union
            label = cfg.aoi_label or ("_".join(cfg.aoi_names) if cfg.aoi_names else "AOI")
        else:
            raise ValueError("set bbox or aoi_path")
        label = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in label.upper())
        self._aoi = (geom, label)
        return self._aoi

    def grid(self):
        geom, _ = self.aoi()
        g = lonlat_to_crs(geom, self.cfg.crs)
        return Grid.for_bounds(g.bounds, res_in_crs(self.cfg.resolution_m, self.cfg.crs),
                               self.cfg.crs), g

    def tiles(self):
        grid, g = self.grid()
        return grid, g, grid.tiles(self.cfg.tile_px, g)

    def resolve_periods(self, periods=None, as_of=None):
        """periods: None (last complete period as of as_of / today), a label, a list of labels,
        or Period tuples."""
        c = self.cfg
        if periods is None:
            as_of = as_of or _dt.date.today().isoformat()
            return [per.last_complete_period(as_of, c.cadence, c.period_anchor)]
        if isinstance(periods, (str, per.Period)):
            periods = [periods]
        out = [p if isinstance(p, per.Period) else per.parse_label(p, c.cadence, c.period_anchor)
               for p in periods]
        return sorted(out, key=lambda p: p.start)

    # ---------------------------------------------------------- cache paths
    def _tile_key(self, tgrid):
        return _h(tgrid.key(), 12)

    def _static_path(self, tgrid):
        c = self.cfg
        k = _h({"wc": DATASETS["worldcover"], "soil": DATASETS["soil"], "crop": c.crop_params(),
                "cn": c.cn_by_hsg, "cmin": c.crop_fraction_min})
        return os.path.join(c.work_dir, "static", k, f"{self._tile_key(tgrid)}.npz")

    def _harmonized_path(self, tgrid, start, end_excl):
        c = self.cfg
        k = _h({"chirps": DATASETS["chirps"], "era5": DATASETS["era5"], "pf": c.precip_fallback})
        return os.path.join(c.work_dir, "harmonized", k,
                            f"{self._tile_key(tgrid)}_{start:%Y%m%d}_{end_excl:%Y%m%d}.npz")

    def _state_path(self, tgrid, date):
        return os.path.join(self.cfg.work_dir, "state", self.cfg.model_hash(),
                            self._tile_key(tgrid), f"S_{pd.Timestamp(date):%Y%m%d}.npy")

    # ---------------------------------------------------------- stage 1
    def static_tile(self, tgrid):
        """Harmonized static layers (dict) of a tile, cached; None when the tile has no cropland."""
        path = self._static_path(tgrid)
        if os.path.exists(path):
            ds = store.load_dataset(path)
            return None if ds is None else {v: ds[v].values for v in ds.data_vars}
        st = hz.harmonize_static(self.fetcher.static(tgrid), self.cfg)
        if not st["crop_mask"].any():
            store.save_skip(path, "no_cropland")
            return None
        import xarray as xr
        store.save_dataset(path, xr.Dataset({k: (("y", "x"), v) for k, v in st.items()},
                                            coords={"y": tgrid.ys, "x": tgrid.xs}))
        return st

    def harmonized_tile(self, tgrid, start, end_excl):
        """Harmonized Dataset of a tile over [start, end_excl) (cached), or None (no cropland)."""
        start, end_excl = pd.Timestamp(start), pd.Timestamp(end_excl)
        st = self.static_tile(tgrid)
        if st is None:
            return None
        path = self._harmonized_path(tgrid, start, end_excl)
        if os.path.exists(path):
            return store.load_dataset(path)
        ds = hz.harmonize_tile(self.fetcher, tgrid, start, end_excl, self.cfg, static=st,
                               log=self.log)
        if ds is not None:
            store.save_dataset(path, ds)
        return ds

    def _window(self, tgrid, period):
        """(start, S0 or None): from the saved state at the period start, else spin-up."""
        s0 = store.load_state(self._state_path(tgrid, period.start))
        if s0 is not None:
            return period.start, s0
        return period.start - pd.Timedelta(days=self.cfg.spinup_days), None

    # ---------------------------------------------------------- planning
    def plan(self, periods=None, as_of=None):
        grid, _, tiles = self.tiles()
        ps = self.resolve_periods(periods, as_of)
        n_era5 = len(DATASETS["era5"]["bands"])
        rows = []
        for p in ps:
            starts = [self._window(tg, p)[0] for _, tg in tiles]
            n_spin = sum(s < p.start for s in starts)
            days = [(p.end - s).days for s in starts]
            req = sum(gee.request_count(n_era5, d, tg, self.cfg) + gee.request_count(1, d, tg, self.cfg)
                      for d, (_, tg) in zip(days, tiles))
            rows.append({"period": p.label, "start": str(p.start.date()),
                         "end": str((p.end - pd.Timedelta(days=1)).date()),
                         "tiles": len(tiles), "tiles_spinup": int(n_spin),
                         "max_daily_requests": int(req)})
        _, label = self.aoi()
        return {"aoi": label, "grid": {"crs": grid.crs, "res": grid.res, "width": grid.width,
                                       "height": grid.height}, "periods": rows}

    def harmonize(self, periods=None, as_of=None):
        """Stage 1 only: build (and cache) every harmonized tile the periods need."""
        grid, _, tiles = self.tiles()
        done = []
        for p in self.resolve_periods(periods, as_of):
            res = self._for_tiles(tiles, lambda tg: self.harmonized_tile(tg, self._window(tg, p)[0],
                                                                           p.end))
            done.append({"period": p.label, "tiles": len(tiles),
                         "with_cropland": sum(r is not None for r, _ in res.values()),
                         "failed": {t: e for t, (_, e) in res.items() if e}})
        return done

    # ---------------------------------------------------------- stage 2
    def _tile_period(self, tgrid, period, geom_crs):
        start, s0 = self._window(tgrid, period)
        ds = self.harmonized_tile(tgrid, start, period.end)
        if ds is None:
            return None
        crop = self.cfg.crop_params()
        fc, wp, sat = ds["FC"].values, ds["WP"].values, ds["SAT"].values
        if s0 is None:
            s0 = fc
        i0 = (period.start - start).days
        i1 = i0 + (period.end - period.start).days
        s_end, res = bal.run_bucket(ds["P"].values, ds["ET0"].values, fc, wp, sat,
                                    ds["CN"].values, s0, crop["kc"], crop["p"],
                                    self.cfg.drainage_tau, self.cfg.irrigation,
                                    periods=[(period.label, i0, i1)])
        store.save_state(self._state_path(tgrid, period.end), s_end)
        r = res[period.label]
        valid = (ds["crop_mask"].values == 1) & geometry_mask(tgrid, geom_crs)
        r["TAW"] = ds["TAW"].values
        r["crop_frac"] = ds["crop_frac"].values
        status = bal.classify(r["f_end"], crop["p"], self.cfg.severe_f, valid)
        bands = {b: np.where(valid, r[b], np.nan).astype(np.float32) for b in BALANCE_BANDS}
        fb = ds["p_fallback"].values[i0:i1]
        return {"status": status, "bands": bands, "valid": valid,
                "p_fallback_days": int(fb.sum()), "spinup": start < period.start}

    def _for_tiles(self, tiles, fn):
        """Run fn(tile_grid) over tiles in parallel -> {tile_id: (result, error)}."""
        out = {}

        def wrap(tg):
            last = None
            for attempt in range(3):
                try:
                    return fn(tg), None
                except hz.DataNotReady as e:
                    return None, f"DataNotReady: {e}"
                except Exception as e:                     # noqa: BLE001
                    last = e
                    self.log(f"  tile error ({e.__class__.__name__}: {e}) — retry {attempt + 1}/2")
                    time.sleep(2 * (attempt + 1))
            traceback.print_exception(last)
            return None, repr(last)

        workers = max(1, int(self.cfg.tile_workers))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(wrap, tg): tid for tid, tg in tiles}
            for k, f in enumerate(as_completed(futs), 1):
                out[futs[f]] = f.result()
                if k % 10 == 0 or k == len(futs):
                    self.log(f"  {k}/{len(futs)} tiles")
        return out

    def run(self, periods=None, as_of=None):
        """Both stages for each period, in date order (so soil state chains). -> [summary dict]."""
        grid, geom_crs, tiles = self.tiles()
        _, label = self.aoi()
        crop = self.cfg.crop_params()
        self.log(f"{label}: grid {grid.width} x {grid.height} px ({grid.crs}, "
                 f"{self.cfg.resolution_m:g} m), {len(tiles)} tile(s)")
        summaries = []
        for p in self.resolve_periods(periods, as_of):
            self.log(f"period {p.label} ({p.start.date()} .. {(p.end - pd.Timedelta(days=1)).date()})")
            out_dir = os.path.join(self.cfg.output_dir, label, p.label)
            stem = os.path.join(out_dir, f"wa_{label}_{p.label}")
            tags = {"period_start": p.start.date(), "period_end_exclusive": p.end.date(),
                    "crop": self.cfg.crop, "kc": crop["kc"], "zr_m": crop["zr"], "p": crop["p"],
                    "irrigation": self.cfg.irrigation, "f": "(S - WP) / (FC - WP)",
                    "status": json.dumps(bal.STATUS)}
            w_status = store.RasterWriter(stem + "_status.tif", grid, ["status"], "int16", -1, tags)
            w_bal = store.RasterWriter(stem + "_balance.tif", grid, BALANCE_BANDS, "float32",
                                       np.nan, tags)
            area_ha = {s: 0.0 for s in bal.STATUS if s != -1}
            pix = {s: 0 for s in bal.STATUS if s != -1}
            crop_ha = {s: 0.0 for s in bal.STATUS if s != -1}
            sums, wsum, fb_days, n_spin = {b: 0.0 for b in BALANCE_BANDS}, 0.0, 0, 0
            tile_grids = dict(tiles)

            results = self._for_tiles(tiles, lambda tg: self._tile_period(tg, p, geom_crs))
            failed = {t: e for t, (_, e) in results.items() if e}
            for tid, (r, _) in results.items():
                if r is None:
                    continue
                tg = tile_grids[tid]
                w_status.write(tg, {"status": r["status"]})
                w_bal.write(tg, r["bands"])
                ha = tg.pixel_area_ha()
                cf = r["bands"]["crop_frac"]
                for s in pix:
                    m = r["status"] == s
                    pix[s] += int(m.sum())
                    area_ha[s] += float(ha[m].sum())
                    crop_ha[s] += float((ha * np.nan_to_num(cf))[m].sum())
                wts = np.where(r["valid"], ha * np.nan_to_num(cf), 0.0)
                for b in BALANCE_BANDS:
                    sums[b] += float(np.nansum(np.nan_to_num(r["bands"][b]) * wts))
                wsum += float(wts.sum())
                fb_days = max(fb_days, r["p_fallback_days"])
                n_spin += int(r["spinup"])
            w_status.close()
            w_bal.close()

            summary = pd.DataFrame([{"status": s, "class": bal.STATUS[s], "pixels": pix[s],
                                     "hectares": round(area_ha[s], 2),
                                     "cropland_hectares": round(crop_ha[s], 2)} for s in pix])
            tot = summary["hectares"].sum()
            summary["share_pct"] = (100 * summary["hectares"] / tot).round(2) if tot else 0.0
            summary.to_csv(stem + "_summary.csv", index=False)
            stats = pd.DataFrame([{"variable": b, "crop_area_weighted_mean":
                                   (sums[b] / wsum if wsum else np.nan)} for b in BALANCE_BANDS])
            stats.to_csv(stem + "_stats.csv", index=False)

            info = {"aoi": label, "period": p.label, "start": str(p.start.date()),
                    "end": str((p.end - pd.Timedelta(days=1)).date()),
                    "tiles": len(tiles),
                    "tiles_with_cropland": sum(r is not None for r, _ in results.values()),
                    "tiles_spun_up": n_spin, "failed_tiles": failed,
                    "precip_from_era5_days": fb_days, "provisional": bool(fb_days or failed),
                    "outputs": {"status": stem + "_status.tif", "balance": stem + "_balance.tif",
                                "summary": stem + "_summary.csv", "stats": stem + "_stats.csv"},
                    "config": {k: v for k, v in vars(self.cfg).items()}}
            with open(stem + "_run.json", "w", encoding="utf-8") as fh:
                json.dump(info, fh, indent=2, default=str)
            if failed:
                self.log(f"  {len(failed)} tile(s) failed — rerun to retry (finished tiles are cached)")
            self.log(f"  wrote {stem}_status.tif")
            info["summary"] = summary.to_dict("records")
            summaries.append(info)
        return summaries
