import os

import numpy as np
import pandas as pd
import pytest
import rasterio

from conftest import FakeFetcher
from waterbalance.config import RunConfig
from waterbalance.pipeline import Runner


@pytest.fixture
def boundaries(tmp_path):
    import geopandas as gpd
    from shapely.geometry import box
    gdf = gpd.GeoDataFrame(
        {"Pro_Name": ["ALPHA", "BETA", "GAMMA"], "Reg_Name": ["REGION X", "REGION X", "REGION Y"]},
        geometry=[box(120.90, 15.60, 120.92, 15.62), box(120.92, 15.60, 120.94, 15.62),
                  box(121.50, 16.00, 121.52, 16.02)], crs="EPSG:4326")
    path = tmp_path / "prov.gpkg"
    gdf.to_file(path, driver="GPKG")
    return str(path)


def _cfg(tmp_path, boundaries, **kw):
    d = dict(aoi_path=boundaries, aoi_name_col="Pro_Name", region_col="Reg_Name", tile_px=16,
             tile_workers=1, spinup_days=60, work_dir=str(tmp_path / "work"),
             output_dir=str(tmp_path / "out"))
    d.update(kw)
    return RunConfig(**d)


def test_select_areas(tmp_path, boundaries):
    r = Runner(_cfg(tmp_path, boundaries), fetcher=FakeFetcher(), log=lambda *a: None)
    assert r.list_areas() == ["ALPHA", "BETA", "GAMMA"]
    assert [n for n, _ in r.select_areas("regional", ["region x"])] == ["ALPHA", "BETA"]
    assert [n for n, _ in r.select_areas("provincial", ["gamma"])] == ["GAMMA"]
    assert len(r.select_areas("national")) == 3
    with pytest.raises(ValueError):
        r.select_areas("provincial", ["NOWHERE"])


def test_run_areas_writes_each_area_and_a_mosaic(tmp_path, boundaries):
    r = Runner(_cfg(tmp_path, boundaries), fetcher=FakeFetcher(), log=lambda *a: None)
    res = r.run_areas("regional", ["REGION X"], ["20260910"])
    assert [s["area"] for s in res["success"]] == ["ALPHA", "BETA"] and not res["failed"]
    for s in res["success"]:
        assert os.path.exists(s["outputs"]["status"])
    m = res["mosaics"][0]
    assert m["areas"] == 2 and not m["missing"]
    with rasterio.open(m["status"]) as src:
        mos = src.read(1)
        assert src.width >= 2 * 20                                  # both provinces side by side
    with rasterio.open(res["success"][0]["outputs"]["status"]) as a:
        n_a = int((a.read(1) > 0).sum())
    with rasterio.open(res["success"][1]["outputs"]["status"]) as b:
        n_b = int((b.read(1) > 0).sum())
    assert int((mos > 0).sum()) == pytest.approx(n_a + n_b, abs=40)   # shared border column
    summ = pd.DataFrame(m["summary"])
    assert summ["hectares"].sum() > 0


def test_retry_only(tmp_path, boundaries):
    r = Runner(_cfg(tmp_path, boundaries), fetcher=FakeFetcher(), log=lambda *a: None)
    res = r.run_areas("national", None, ["20260910"], only=["beta"])
    assert [s["area"] for s in res["success"]] == ["BETA"] and res["mosaics"] == []


class FakeBlob:
    def __init__(self, store, name):
        self.store, self.name = store, name

    def upload_from_filename(self, path):
        with open(path, "rb") as fh:
            self.store[self.name] = len(fh.read())


class FakeGCS:
    def __init__(self):
        self.objects = {}

    def bucket(self, name):
        client = self

        class B:
            def blob(self, n):
                return FakeBlob(client.objects, f"{name}/{n}")
        return B()


def test_gcs_target_uploads_and_cleans_staging(tmp_path, boundaries):
    gcs = FakeGCS()
    cfg = _cfg(tmp_path, boundaries, output_target="gcs", gcs_bucket="bkt", gcs_prefix="wa")
    r = Runner(cfg, fetcher=FakeFetcher(), log=lambda *a: None, gcs_client=gcs)
    res = r.run_areas("provincial", ["ALPHA"], ["20260910"])
    s = res["success"][0]
    assert s["uris"][0] == "gs://bkt/wa/ALPHA/20260910/wa_ALPHA_20260910_status.tif"
    assert "bkt/wa/ALPHA/20260910/wa_ALPHA_20260910_balance.tif" in gcs.objects
    assert not os.path.exists(s["outputs"]["status"])                # staging removed


def test_gcs_needs_bucket(tmp_path, boundaries):
    with pytest.raises(ValueError):
        Runner(_cfg(tmp_path, boundaries, output_target="gcs"), fetcher=FakeFetcher())


def test_quick_check_daily_series(tmp_path):
    cfg = RunConfig(spinup_days=60, crop="rice", work_dir=str(tmp_path / "w"))
    r = Runner(cfg, fetcher=FakeFetcher(), log=lambda *a: None)
    ds = r.quick_check((120.90, 15.60, 120.91, 15.61), "20260910")
    assert ds.sizes["time"] == 60 + 14
    assert {"f", "S", "P", "ET0", "ET", "status", "f_end"} <= set(ds.data_vars)
    crop = ds["crop_mask"].values == 1
    f = ds["f"].values[:, crop]
    assert np.isfinite(f).all() and np.isnan(ds["f"].values[:, ~crop]).all()
    np.testing.assert_allclose(ds["f"].values[-1][crop], ds["f_end"].values[crop])
