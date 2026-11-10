"""Settings: datasets, crop parameters, model and run options.

Everything a run needs is in RunConfig. Dataset ids and band names live in DATASETS so they can be
changed without touching the code (Earth Engine catalogue ids do get versioned).
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field

# ------------------------------------------------------------------ datasets (Earth Engine)
DATASETS = {
    # CHIRPS v2.0 daily, 0.05 deg, mm/day. Final product: ~3-6 weeks behind real time.
    "chirps": {
        "id": "UCSB-CHG/CHIRPS/DAILY",
        "bands": {"precip": "precipitation"},
    },
    # ERA5-Land daily aggregates, 0.1 deg. Units as served: K, Pa, J m-2 day-1, m day-1.
    "era5": {
        "id": "ECMWF/ERA5_LAND/DAILY_AGGR",
        "bands": {
            "tmin": "temperature_2m_min",                   # K
            "tmax": "temperature_2m_max",                   # K
            "tdew": "dewpoint_temperature_2m",              # K (daily mean)
            "u10": "u_component_of_wind_10m",               # m s-1
            "v10": "v_component_of_wind_10m",               # m s-1
            "rns": "surface_net_solar_radiation_sum",       # J m-2 day-1
            "rnl": "surface_net_thermal_radiation_sum",     # J m-2 day-1 (negative = loss)
            "psurf": "surface_pressure",                    # Pa
            "tp": "total_precipitation_sum",                # m day-1 (fallback for missing CHIRPS days)
        },
    },
    # ESA WorldCover 10 m (v200 = 2021). Class 40 = cropland.
    "worldcover": {
        "id": "ESA/WorldCover/v200",
        "band": "Map",
        "cropland_class": 40,
    },
    # OpenLandMap soil, 250 m, values at depths 0, 10, 30, 60, 100, 200 cm (bands b0 ... b200).
    #   sand / clay : % (kg/kg x 100)
    #   oc          : organic carbon in units of 5 g/kg (value x 5 = g/kg)
    "soil": {
        "sand": "OpenLandMap/SOL/SOL_SAND-WFRACTION_USDA-3A1A1A_M/v02",
        "clay": "OpenLandMap/SOL/SOL_CLAY-WFRACTION_USDA-3A1A1A_M/v02",
        "oc": "OpenLandMap/SOL/SOL_ORGANIC-CARBON_USDA-6A1C_M/v02",
        "depths_cm": [0, 10, 30, 60, 100, 200],
    },
}

# ------------------------------------------------------------------ crops
# Season-average constants (FAO-56 Tables 12 and 22). A crop-calendar Kc curve is a later step.
#   kc : crop coefficient (ETc = Ks * Kc * ET0)
#   zr : root-zone depth, m
#   p  : depletion fraction before stress -> stress starts when f < 1 - p
CROPS = {
    "generic": {"kc": 1.00, "zr": 0.60, "p": 0.50},
    "rice":    {"kc": 1.10, "zr": 0.60, "p": 0.20},   # non-flooded (AWD / upland); paddies need ponding
    "maize":   {"kc": 1.00, "zr": 1.00, "p": 0.55},
}

# SCS Curve Number for cropland by hydrologic soil group (row crops, straight row, good condition,
# antecedent moisture II). Bunded paddies run off less: calibrate locally.
CN_CROPLAND = {"A": 67, "B": 78, "C": 85, "D": 89}


@dataclass
class RunConfig:
    # ---- area
    bbox: list | None = None                  # [west, south, east, north] lon/lat
    aoi_path: str | None = None               # vector file with boundaries
    aoi_name_col: str | None = None           # column holding the area names
    aoi_names: list | None = None             # names to process (None = every feature)
    aoi_label: str | None = None              # name used in output files (default from names / "AOI")
    region_col: str | None = None             # column used to pick areas by region (run_areas)

    # ---- grid (harmonized resolution)
    crs: str = "EPSG:4326"                    # EPSG:4326 = global grid (like ai-cropgrowth);
                                              # a UTM code (e.g. EPSG:32651) gives exact 1 ha pixels
    resolution_m: float = 100.0               # 100 m = 1 ha
    tile_px: int = 128                        # tile side in pixels (128 px x 100 m = 12.8 km)

    # ---- Earth Engine
    ee_project: str | None = None
    max_request_mb: float = 32.0              # computePixels hard limit is 48 MB per request
    max_bands_per_request: int = 256
    request_retries: int = 4
    tile_workers: int = 2

    # ---- crop mask
    crop_fraction_min: float = 0.5            # 1 ha cell is cropland when >= this share of its
                                              # 10 m WorldCover pixels are class 40

    # ---- periods
    cadence: str = "14d"                      # '14d' (fixed 14-day windows from period_anchor) or
                                              # 'semimonthly' (1st-15th, 16th-end of month)
    period_anchor: str = "2026-01-01"
    spinup_days: int = 365                    # used when no saved soil state exists for the tile
    precip_fallback: str = "era5"             # 'era5': days CHIRPS lacks use ERA5-Land precipitation
                                              # (flagged provisional); 'none': fail instead

    # ---- model
    crop: str = "generic"                     # key of CROPS
    kc: float | None = None                   # overrides of the crop preset
    zr: float | None = None
    p: float | None = None
    cn_by_hsg: dict = field(default_factory=lambda: dict(CN_CROPLAND))
    drainage_tau: float = 0.5                 # share of the water above field capacity drained per day
    irrigation: str = "none"                  # 'none' (rainfed) or 'auto' (refill to FC when f < 1 - p)
    severe_f: float = 0.25                    # f below this = severe deficit

    # ---- storage
    work_dir: str = "work"                    # harmonized tiles, soil state, caches
    output_dir: str = "outputs"               # local outputs ('local'), staging folder for 'gcs'
    output_target: str = "local"              # 'local' | 'gdrive' | 'gcs' | 'both' (Drive + GCS)
    drive_root: str = "/content/drive/MyDrive/AI-WaterBalance/outputs"   # 'gdrive' / 'both'
    gcs_bucket: str | None = None             # 'gcs' / 'both'
    gcs_prefix: str = "products/water_availability"
    mosaic_after: bool = True                 # run_areas with several areas: one mosaic per period
    mosaic_label: str | None = None           # mosaic folder / file name (default PHILIPPINES or region)

    # ------------------------------------------------------------------
    def crop_params(self):
        if self.crop not in CROPS:
            raise ValueError(f"crop must be one of {sorted(CROPS)}")
        c = dict(CROPS[self.crop])
        for k in ("kc", "zr", "p"):
            v = getattr(self, k)
            if v is not None:
                c[k] = float(v)
        return c

    def output_root(self):
        """Folder the outputs are written to (Drive for 'gdrive' / 'both')."""
        if self.output_target not in ("local", "gdrive", "gcs", "both"):
            raise ValueError("output_target must be 'local', 'gdrive', 'gcs' or 'both'")
        return self.drive_root if self.output_target in ("gdrive", "both") else self.output_dir

    def model_hash(self):
        """Hash of everything that changes the soil-water state (for state-file keys)."""
        parts = dict(crop=self.crop_params(), cn=self.cn_by_hsg, tau=self.drainage_tau,
                     irr=self.irrigation, crs=self.crs, res=self.resolution_m,
                     tile=self.tile_px, cmin=self.crop_fraction_min, pf=self.precip_fallback,
                     ds=DATASETS)
        return hashlib.md5(json.dumps(parts, sort_keys=True).encode()).hexdigest()[:10]

    @classmethod
    def from_json(cls, path, **overrides):
        with open(path, encoding="utf-8") as fh:
            d = json.load(fh)
        d.update({k: v for k, v in overrides.items() if v is not None})
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown config field(s): {sorted(unknown)}")
        return cls(**d)
