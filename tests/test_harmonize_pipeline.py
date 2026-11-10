import json
import os

import numpy as np
import pandas as pd
import pytest
import rasterio

from conftest import CHIRPS_UNTIL, FakeFetcher
from waterbalance import harmonize as hz
from waterbalance.config import RunConfig
from waterbalance.grid import Grid, res_in_crs
from waterbalance.pipeline import Runner

BBOX = [120.90, 15.60, 120.93, 15.63]          # ~33 x 33 px at 1 ha


def _cfg(tmp_path, **kw):
    d = dict(bbox=BBOX, aoi_label="TEST", tile_px=16, tile_workers=1, spinup_days=120,
             work_dir=str(tmp_path / "work"), output_dir=str(tmp_path / "out"))
    d.update(kw)
    return RunConfig(**d)


def _grid():
    return Grid.for_bounds((120.9, 15.6, 120.91, 15.61), res_in_crs(100, "EPSG:4326"), "EPSG:4326")


# ------------------------------------------------------------------ harmonization
def test_static_harmonization_units_and_mask(tmp_path):
    cfg = _cfg(tmp_path)
    st = hz.harmonize_static(FakeFetcher().static(_grid()), cfg)
    assert st["crop_mask"][:, 0].all() and not st["crop_mask"][:, -1].any()
    # loam, Zr 0.6 m: FC ~ 0.28 x 600 mm, WP ~ 0.12 x 600 mm
    assert np.nanmean(st["FC"]) == pytest.approx(168, abs=15)
    assert np.nanmean(st["WP"]) == pytest.approx(72, abs=15)
    assert (st["SAT"] > st["FC"]).all() and (st["TAW"] > 0).all()
    assert (st["CN"] == 78).all()                                   # loam -> group B


def test_daily_harmonization_units_and_fallback(tmp_path):
    cfg, g, f = _cfg(tmp_path), _grid(), FakeFetcher()
    a, b = pd.Timestamp("2026-10-10"), pd.Timestamp("2026-10-30")
    days, p, et0, fb, _ = hz.harmonize_daily(f.daily("chirps", a, b, g), f.daily("era5", a, b, g),
                                             a, b, cfg)
    assert len(days) == 20 and p.shape[0] == 20
    assert 3.0 < float(et0.mean()) < 7.0                            # warm humid tropics, mm/day
    late = days > CHIRPS_UNTIL
    assert fb[late].all() and not fb[~late].any()                   # ERA5 fills missing CHIRPS


def test_era5_latency_raises(tmp_path):
    cfg, g, f = _cfg(tmp_path), _grid(), FakeFetcher()
    a, b = pd.Timestamp("2026-12-20"), pd.Timestamp("2027-01-05")
    with pytest.raises(hz.DataNotReady):
        hz.harmonize_daily(f.daily("chirps", a, b, g), f.daily("era5", a, b, g), a, b, cfg)


def test_chirps_missing_without_fallback_raises(tmp_path):
    cfg, g, f = _cfg(tmp_path, precip_fallback="none"), _grid(), FakeFetcher()
    a, b = pd.Timestamp("2026-10-10"), pd.Timestamp("2026-10-30")
    with pytest.raises(hz.DataNotReady):
        hz.harmonize_daily(f.daily("chirps", a, b, g), f.daily("era5", a, b, g), a, b, cfg)


def test_fill_nearest():
    a = np.array([[1.0, np.nan], [np.nan, 4.0]], np.float32)
    out = hz.fill_nearest(a)
    assert np.isfinite(out).all() and out[0, 0] == 1 and out[1, 1] == 4


# ------------------------------------------------------------------ pipeline
def test_run_writes_maps_and_chains_soil_state(tmp_path):
    fake = FakeFetcher()
    r = Runner(_cfg(tmp_path), fetcher=fake, log=lambda *a: None)
    grid, _, tiles = r.tiles()
    assert len(tiles) == 9                                          # 33 px / 16 -> 3 x 3

    s1 = r.run(["20260910"])[0]
    assert s1["tiles_spun_up"] == s1["tiles_with_cropland"] > 0
    first_starts = {c[1] for c in fake.daily_calls}
    assert min(first_starts) == pd.Timestamp("2026-09-10") - pd.Timedelta(days=120)

    with rasterio.open(s1["outputs"]["status"]) as src:
        st = src.read(1)
        assert (src.width, src.height) == (grid.width, grid.height)
    assert set(np.unique(st)) <= {-1, 1, 2, 3, 4}
    assert (st[:, : grid.width // 3] != -1).any() and (st[:, -2:] == -1).all()
    with rasterio.open(s1["outputs"]["balance"]) as src:
        names = list(src.descriptions)
        resid = src.read(names.index("residual") + 1)
        f_end = src.read(names.index("f_end") + 1)
    assert np.nanmax(np.abs(resid)) < 1e-2
    assert np.nanmin(f_end) >= 0
    summary = pd.read_csv(s1["outputs"]["summary"])
    assert summary["hectares"].sum() > 0

    # next period starts from the saved state: data requested from the period start only
    fake.daily_calls.clear()
    s2 = r.run(["20260924"])[0]
    assert s2["tiles_spun_up"] == 0
    assert {c[1] for c in fake.daily_calls} == {pd.Timestamp("2026-09-24")}
    assert s2["precip_from_era5_days"] == 0

    # dry since August -> f falls between the two periods
    f2 = pd.read_csv(s2["outputs"]["stats"]).set_index("variable")["crop_area_weighted_mean"]
    f1 = pd.read_csv(s1["outputs"]["stats"]).set_index("variable")["crop_area_weighted_mean"]
    assert f2["f_end"] < f1["f_end"]

    # a period with CHIRPS missing is flagged provisional
    s3 = r.run(["20261022"])[0]
    assert s3["precip_from_era5_days"] > 0 and s3["provisional"]
    with open(s3["outputs"]["summary"].replace("_summary.csv", "_run.json")) as fh:
        assert json.load(fh)["provisional"] is True


def test_rerun_uses_cache(tmp_path):
    fake = FakeFetcher()
    r = Runner(_cfg(tmp_path), fetcher=fake, log=lambda *a: None)
    r.harmonize(["20260910"])
    n_static, n_daily = len(fake.static_calls), len(fake.daily_calls)
    assert n_daily > 0
    # no saved state yet -> run() needs the same spin-up window, served from the cache
    r.run(["20260910"])
    assert len(fake.static_calls) == n_static and len(fake.daily_calls) == n_daily


def test_tiles_without_cropland_skip_daily_data(tmp_path):
    fake = FakeFetcher(no_crop=True)
    r = Runner(_cfg(tmp_path), fetcher=fake, log=lambda *a: None)
    s = r.run(["20260910"])[0]
    assert s["tiles_with_cropland"] == 0 and fake.daily_calls == []
    assert os.path.exists(s["outputs"]["status"])


def test_plan(tmp_path):
    r = Runner(_cfg(tmp_path), fetcher=FakeFetcher(), log=lambda *a: None)
    plan = r.plan(["20260910"])
    assert plan["periods"][0]["tiles"] == 9 and plan["periods"][0]["tiles_spinup"] == 9


def test_provisional_tiles_are_fetched_again(tmp_path):
    fake = FakeFetcher()
    r = Runner(_cfg(tmp_path), fetcher=fake, log=lambda *a: None)
    r.harmonize(["20261022"])                       # CHIRPS ends 2026-10-20 -> ERA5 fallback
    n = len(fake.daily_calls)
    r.harmonize(["20261022"])
    assert len(fake.daily_calls) == 2 * n           # not served from the cache
    r.harmonize(["20260910"])                       # complete CHIRPS -> cached
    m = len(fake.daily_calls)
    r.harmonize(["20260910"])
    assert len(fake.daily_calls) == m
