"""A fake Earth Engine fetcher serving synthetic raw data in each dataset's NATIVE units, so the
tests exercise the unit harmonization and run offline (no earthengine-api login, no network).

Cropland: the west half of any tile (crop_frac 0.8), none in the east half. Soil: loam
(sand 40 %, clay 20 %, OM ~2.5 %). Weather: warm and humid; rain 12 mm every 4th day before
RAIN_UNTIL, dry afterwards.
  GPM    half-hourly mean rate in mm/hr (rain / 24) up to GPM_UNTIL; days after GPM_FINAL_UNTIL
         are Late run (flagged provisional)
  CHIRPS mm/day up to CHIRPS_UNTIL
Later days must come from ERA5-Land."""
import numpy as np
import pandas as pd
import pytest

from waterbalance.config import DATASETS

RAIN_UNTIL = pd.Timestamp("2026-08-01")
CHIRPS_UNTIL = pd.Timestamp("2026-10-20")      # last CHIRPS day
GPM_FINAL_UNTIL = pd.Timestamp("2026-10-01")   # last GPM Final-run day; later = Late run
GPM_UNTIL = pd.Timestamp("2026-11-09")         # last GPM day (about a day of latency)
ERA5_UNTIL = pd.Timestamp("2026-12-31")


class FakeFetcher:
    def __init__(self, no_crop=False):
        self.no_crop = no_crop
        self.static_calls, self.daily_calls = [], []

    def static(self, grid):
        self.static_calls.append(grid.key())
        h, w = grid.height, grid.width
        crop = np.zeros((h, w), np.float32)
        if not self.no_crop:
            crop[:, : max(1, w // 2)] = 0.8
        out = {"crop_frac": crop}
        for d in DATASETS["soil"]["depths_cm"]:
            out[f"sand_{d}"] = np.full((h, w), 40.0, np.float32)
            out[f"clay_{d}"] = np.full((h, w), 20.0, np.float32)
            out[f"oc_{d}"] = np.full((h, w), 2.9, np.float32)     # x5 g/kg -> OM ~2.5 %
        return out

    def daily(self, source, start, end_excl, grid):
        self.daily_calls.append((source, pd.Timestamp(start), pd.Timestamp(end_excl)))
        until = {"chirps": CHIRPS_UNTIL, "gpm": GPM_UNTIL}.get(source, ERA5_UNTIL)
        days = pd.date_range(start, pd.Timestamp(end_excl) - pd.Timedelta(days=1), freq="D")
        days = days[days <= until]
        shape = (len(days), grid.height, grid.width)
        rain = np.array([12.0 if (d < RAIN_UNTIL and d.day_of_year % 4 == 0) else 0.0
                         for d in days], np.float32)[:, None, None]
        if source == "chirps":
            return days, {"precip": np.broadcast_to(rain, shape).astype(np.float32).copy()}
        if source == "gpm":
            return days, {"precip": np.broadcast_to(rain / 24.0, shape).astype(np.float32).copy(),
                          "_provisional": np.asarray(days > GPM_FINAL_UNTIL)}
        full = lambda v: np.full(shape, v, np.float32)          # noqa: E731
        return days, {
            "tmin": full(296.0), "tmax": full(305.0), "tdew": full(294.0),
            "u10": full(2.0), "v10": full(1.0),
            "rns": full(17e6), "rnl": full(-4e6), "psurf": full(100_500.0),
            "tp": np.broadcast_to(rain / 1000.0, shape).astype(np.float32).copy(),
        }


@pytest.fixture
def fake():
    return FakeFetcher()
