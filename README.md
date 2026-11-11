# Water availability (f) from a daily water balance

Every 14 days, this maps how much plant-available water is left in the root zone of cropland, on a
1 ha grid:

```
P + I − R − ET − D = ΔS          (precipitation, irrigation, runoff, evapotranspiration, drainage)
f = (S − WP) / (FC − WP)          (share of the root zone's available water still there)
```

| Status | Code | f at the end of the period |
|---|---|---|
| no data | −1 | not cropland, no soil data, outside the area |
| severe deficit | 1 | f < 0.25 (`severe_f`) |
| moderate deficit | 2 | 0.25 ≤ f < 1 − p |
| adequate | 3 | 1 − p ≤ f ≤ 1 |
| surplus | 4 | f > 1 (wetter than field capacity: draining / waterlogged) |

p is the crop's depletion fraction (FAO-56): stress starts at f < 0.8 for rice, < 0.45 for maize.

## Pipeline

```
Stage 1 — harmonize (harmonize.py), per tile, cached under work_dir/
  Earth Engine (ee.data.computePixels -> numpy -> xarray), every layer requested ON the 1 ha grid
  ├─ ESA WorldCover 10 m ── reduceResolution(mean) ──> crop_frac (share of class 40 per 1 ha cell)
  │                                                    crop_mask = crop_frac >= 0.5
  ├─ OpenLandMap sand/clay/OC 250 m ── bilinear ──> % -> fraction, OC (x5 g/kg) -> OM %
  │                                                 Saxton & Rawls -> θFC, θWP, θSAT per depth
  │                                                 integrate 0..Zr -> FC, WP, SAT, TAW (mm)
  │                                                 texture -> hydrologic group -> curve number
  ├─ ERA5-Land daily 0.1° ── bilinear ──> K -> °C, Pa -> kPa, J m-2 -> MJ m-2, u10/v10 -> u2,
  │                                       m -> mm  =>  FAO-56 Penman-Monteith ET0 (mm/day)
  └─ GPM IMERG V07 half-hourly 0.1° ── daily mean per UTC day (server) ── bilinear
                                      ──> mm/hr x 24 -> P (mm/day); Late-run days flagged
                                      provisional; days with no data yet use ERA5-Land P
                                      (precip_source='chirps': CHIRPS daily 0.05°, mm/day)

Stage 2 — water balance (balance.py), per pixel per day
  irrigation (optional) -> SCS-CN runoff -> infiltration (excess over saturation runs off)
  -> drainage above FC -> ETc = Ks·Kc·ET0 -> S, f
  per period: ΣP, ΣI, ΣR, ΣET, ΣD, ΔS, f at the end, min f, days in stress
```

### Tiling (staying under the computePixels limits)

computePixels refuses requests over ~48 MB. Each read is split two ways:

- **Space:** the area is cut into `tile_px` × `tile_px` tiles (default 128 px = 12.8 km). Tiles outside
  the area's boundary are never requested.
- **Time:** each tile's daily stack is requested in chunks of days sized so that
  days × variables × pixels × 4 bytes ≤ `max_request_mb` (32) and days × variables ≤
  `max_bands_per_request` (256).

A chunk that still fails with a size, memory or timeout error is halved and retried. Rate-limit and
server errors are retried with backoff. Tiles without cropland skip the daily data entirely. Raw
ERA5 variables are converted in 90-day chunks, so a tile keeps only P and ET0 in memory.

### Soil state between runs

The root-zone storage S at each period's end is saved per tile. The next 14-day run starts from it
and downloads only that period's 14 days. A tile with no saved state (the first run, or after a
model setting changed) is spun up over `spinup_days` (365) before the period, starting at field
capacity. Run periods in date order, or backfill with several labels in one call.

### Grid

- Default `EPSG:4326` with pixel edges at multiples of 100 m / 111 320 m per degree, the same global
  grid scheme as ai-cropgrowth. A pixel is 1 ha at the equator and about 0.97 ha at 15° N.
  Hectares in the summaries use each pixel's exact area.
- Set `crs` to a UTM code (Luzon/Visayas/Mindanao: `EPSG:32651`; Palawan west of 120° E:
  `EPSG:32650`) for pixels of exactly 1 ha.

## Outputs

`output_dir/<AOI>/<period>/`

| File | Content |
|---|---|
| `wa_<AOI>_<period>_status.tif` | int16 status class (table above) |
| `wa_<AOI>_<period>_balance.tif` | float32 bands: `f_end`, `f_min`, `stress_days`, `P`, `I`, `R`, `ET`, `D`, `dS`, `S_end`, `TAW`, `crop_frac`, `residual` (mm unless noted) |
| `wa_<AOI>_<period>_summary.csv` | pixels, hectares, cropland hectares and share per class |
| `wa_<AOI>_<period>_stats.csv` | cropland-area-weighted mean of every band |
| `wa_<AOI>_<period>_run.json` | settings, tiles, failures, `precip_from_era5_days`, `provisional` |

`residual` = P + I − R − ET − D − ΔS, which is zero by construction. It is kept as a check.

## Periods

| `cadence` | Periods | Label |
|---|---|---|
| `14d` (default) | 14-day windows from `period_anchor` (2026-01-01) | `20261022` (first day) |
| `semimonthly` | 1st–15th, 16th–end of month | `202610H1`, `202610H2` |

With no `--periods`, the run makes the last period that has fully ended by today (or `--as-of`).

## Run it

**Colab:** open `notebooks/run_water_availability.ipynb`, the full pipeline notebook (the
ai-cropgrowth layout without the agent). It covers:

1. clone and install, Drive, Earth Engine and Cloud Storage sign-in
2. settings
3. a **quick check** on a small bbox, with maps of cropland share, TAW, f and status, and the daily
   P / ETc / f of a few pixels
4. the plan
5. the periodic run per area (national / regional / provincial / municipal / bbox), with results,
   hectares per class and the mosaic
6. the maps read back from the GeoTIFFs

**Command line:**

```bash
pip install -r requirements.txt
earthengine authenticate            # once
```

```bash
python -m waterbalance plan      --config config.json --periods 20261022
python -m waterbalance harmonize --config config.json --periods 20261022   # stage 1 only
python -m waterbalance run       --config config.json --periods 20261008 20261022
python -m waterbalance run       --config config.json                       # last complete period
python -m waterbalance run       --bbox 120.85 15.55 121.05 15.75 --crop rice --ee-project my-project
```

`config.json` holds `RunConfig` fields (`waterbalance/config.py`); flags override it. See
`config.example.json`.

```python
from waterbalance import RunConfig, Runner
# one map per province of a region + one mosaic per period, saved to Drive
r = Runner(RunConfig(aoi_path="boundary_province.gpkg", aoi_name_col="Pro_Name", region_col="Reg_Name",
                     crop="rice", ee_project="my-project", output_target="gdrive"))
r.run_areas("regional", ["REGION III"], ["20261022"])
r.quick_check((120.90, 15.60, 120.95, 15.65), "20261022")      # daily f series, in memory

r = Runner(RunConfig(aoi_path="boundary_municipal.gpkg", aoi_name_col="Mun_Name",
                     aoi_names=["SCIENCE CITY OF MUÑOZ"], crop="rice", ee_project="my-project"))
r.run(["20261022"])
```

**Main settings:**

| Setting | Default | What it does |
|---|---|---|
| `bbox` / `aoi_path` + `aoi_name_col` + `aoi_names` | — | Area to map |
| `crs`, `resolution_m` | `EPSG:4326`, 100 | Harmonized grid |
| `tile_px` | 128 | Tile side in pixels |
| `max_request_mb`, `max_bands_per_request` | 32, 256 | computePixels request budget |
| `crop` | `generic` | `generic`, `rice`, `maize` (Kc, Zr, p in `config.CROPS`); `kc`, `zr`, `p` override |
| `crop_fraction_min` | 0.5 | Cropland share for a 1 ha cell to count as cropland |
| `irrigation` | `none` | `auto`: refill to FC whenever f < 1 − p (an irrigation-requirement scenario) |
| `cn_by_hsg` | 67/78/85/89 | Curve numbers for groups A–D (calibrate for bunded paddies) |
| `drainage_tau` | 0.5 | Share of the water above FC drained per day |
| `spinup_days` | 365 | Spin-up when no saved state exists |
| `precip_source` | `gpm` | `gpm` (IMERG V07, ~1 day behind) or `chirps` (CHIRPS daily, weeks behind) |
| `precip_fallback` | `era5` | Use ERA5-Land P on days the source lacks (`none`: fail instead) |
| `tile_workers` | 2 | Tiles processed in parallel |
| `output_target` | `local` | `local` (`output_dir`), `gdrive` (`drive_root`), `gcs` (`gcs_bucket`/`gcs_prefix`), `both` |
| `region_col`, `mosaic_after`, `mosaic_label` | —, true, — | `run_areas`: pick areas by region; one mosaic per period |

Dataset ids and band names are in `config.DATASETS`.

## Tests

```bash
pip install pytest && pytest
```

The tests run offline with a fake Earth Engine fetcher that serves synthetic data in each dataset's
native units. They cover:

- unit conversions
- FAO-56 Example 18 (ET0)
- Saxton & Rawls loam values
- the curve-number runoff formula
- water-balance closure, drying to the wilting point, surplus after heavy rain and auto irrigation
- request splitting and retries
- grid alignment and pixel areas
- periods
- the full pipeline: maps, the soil-state carry-over, the cache, and provisional flags

## Limitations / next steps

- Kc, Zr and p are season-average constants for each crop. A crop-calendar Kc curve (e.g. from
  ai-cropgrowth's growth stages) is the natural next step.
- Flooded paddies need ponded-water storage, and percolation in place of the drainage term.
- `irrigation='auto'` is a scenario (what full irrigation would add), not observed irrigation.
  Irrigated-area masks, measured volumes and SMAP detection are planned.
- GPM IMERG arrives within about a day, but recent days come from its Late run (Earth Engine
  property `status = 'provisional'`). They are counted in `precip_provisional_days`, and the period
  is flagged `provisional`. The Final run replaces them a few months later. Rerun those periods
  then, and every later one, in order. Provisional tiles are always fetched again, never served
  from the cache.
- GPM days are UTC days (08:00–08:00 Philippine time). This doesn't matter for 14-day totals.
- The GPM `status` property and the 48 half-hourly images per day are assumptions from the
  catalog; check them on the first live run. If the property is absent, nothing is flagged as Late
  run, and only incomplete days are flagged.
- The Earth Engine calls have been checked against the API documentation but not yet run live
  from this repository. Run `plan`, then one small bbox first.
