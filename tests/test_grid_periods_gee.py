import numpy as np
import pandas as pd
import pytest

from waterbalance import gee, periods
from waterbalance.grid import Grid, res_in_crs


# ------------------------------------------------------------------ grid
def test_grid_is_globally_aligned_and_tiled():
    res = res_in_crs(100, "EPSG:4326")
    g = Grid.for_bounds((120.9012, 15.6003, 120.9567, 15.6511), res, "EPSG:4326")
    assert abs(g.x0 / res - round(g.x0 / res)) < 1e-6
    assert abs(g.y1 / res - round(g.y1 / res)) < 1e-6
    tiles = g.tiles(16)
    assert sum(t.width * t.height for _, t in tiles) == g.width * g.height
    _, t = tiles[-1]
    col, row = t.offset_in(g)
    assert 0 <= col < g.width and 0 <= row < g.height


def test_pixel_area_is_about_one_hectare():
    res = res_in_crs(100, "EPSG:4326")
    eq = Grid.for_bounds((0, -0.001, 0.001, 0.001), res, "EPSG:4326").pixel_area_ha()
    assert eq.mean() == pytest.approx(1.0, abs=0.01)
    lat15 = Grid.for_bounds((120, 15, 120.001, 15.001), res, "EPSG:4326").pixel_area_ha()
    assert lat15.mean() == pytest.approx(np.cos(np.radians(15)), abs=0.01)
    utm = Grid.for_bounds((300000, 1650000, 301000, 1651000), 100.0, "EPSG:32651")
    assert (utm.pixel_area_ha() == 1.0).all()


# ------------------------------------------------------------------ periods
def test_14d_periods():
    p = periods.period_of("2026-01-20", "14d", "2026-01-01")
    assert (p.start, p.end, p.label) == (pd.Timestamp("2026-01-15"), pd.Timestamp("2026-01-29"),
                                         "20260115")
    last = periods.last_complete_period("2026-01-28", "14d", "2026-01-01")
    assert last.label == "20260115"                               # 28th is its last day
    assert periods.last_complete_period("2026-01-27", "14d", "2026-01-01").label == "20260101"
    assert periods.parse_label("20260115").start == pd.Timestamp("2026-01-15")
    with pytest.raises(ValueError):
        periods.parse_label("20260116")


def test_semimonthly_periods():
    p = periods.period_of("2026-02-20", "semimonthly")
    assert p.label == "202602H2" and p.end == pd.Timestamp("2026-03-01")
    assert [x.label for x in periods.periods_between("2026-01-10", "2026-02-01", "semimonthly")] \
        == ["202601H1", "202601H2", "202602H1"]


# ------------------------------------------------------------------ request planning
def test_days_per_request_respects_bytes_and_bands():
    # 9 vars x 128 x 128 x 4 B = 0.56 MB/day -> band limit 256 // 9 = 28 days
    assert gee.days_per_request(9, 128, 128, 32, 256) == 28
    # 1 var at 1024 x 1024 = 4 MB/day -> 8 days in 32 MB
    assert gee.days_per_request(1, 1024, 1024, 32, 256) == 8
    with pytest.raises(gee.RequestTooLarge):
        gee.days_per_request(9, 2048, 2048, 32, 256)


def test_fetch_split_halves_too_large_requests():
    calls = []

    def fetch(a, b):
        calls.append((a, b))
        if (b - a).days > 5:
            raise RuntimeError("Total request size (60000000 bytes) must be less than or equal "
                               "to 50331648 bytes.")
        return (a, b)

    out = gee.fetch_split(fetch, "2026-01-01", "2026-01-21", 20, sleep=lambda s: None,
                          log=lambda *a: None)
    spans = [(a, b) for a, b in out]
    assert spans[0][0] == pd.Timestamp("2026-01-01") and spans[-1][1] == pd.Timestamp("2026-01-21")
    assert all((b - a).days <= 5 for a, b in spans)
    assert all(spans[i][1] == spans[i + 1][0] for i in range(len(spans) - 1))   # contiguous


def test_fetch_split_retries_transient_errors():
    n = {"k": 0}

    def fetch(a, b):
        n["k"] += 1
        if n["k"] < 3:
            raise RuntimeError("429 Too Many Requests")
        return "ok"

    assert gee.fetch_split(fetch, "2026-01-01", "2026-01-03", 10, sleep=lambda s: None) == ["ok"]


def test_band_parsing_and_nodata():
    assert gee.parse_daily_bands(["20260101_tmin"]) == [(pd.Timestamp("2026-01-01"), "tmin")]
    with pytest.raises(ValueError):
        gee.parse_daily_bands(["tmin"])
    arr = np.zeros((2, 2), dtype=[("a", "f4"), ("b", "f4")])
    arr["a"] = 1.0
    arr["b"][0, 0] = gee.NODATA
    st = gee.structured_to_stack(arr, ["a", "b"])
    assert st.shape == (2, 2, 2) and np.isnan(st[1, 0, 0]) and st[0].sum() == 4
