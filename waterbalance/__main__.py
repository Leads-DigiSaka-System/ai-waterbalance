"""Command line.

  python -m waterbalance plan      --config config.json --periods 20261027
  python -m waterbalance harmonize --config config.json --periods 20261027     # stage 1 only
  python -m waterbalance run       --config config.json                         # last complete period
  python -m waterbalance run       --config config.json --periods 20260929 20261013 20261027
  python -m waterbalance run       --config config.json --bbox 120.85 15.55 121.05 15.75 --crop rice

Settings come from --config (JSON of RunConfig fields, see waterbalance/config.py); flags override.
"""
import argparse
import json
import sys

from .config import RunConfig

FLAGS = {
    "--aoi-path": ("aoi_path", str), "--aoi-name-col": ("aoi_name_col", str),
    "--aoi-label": ("aoi_label", str), "--crs": ("crs", str),
    "--resolution": ("resolution_m", float), "--tile-px": ("tile_px", int),
    "--tile-workers": ("tile_workers", int), "--ee-project": ("ee_project", str),
    "--cadence": ("cadence", str), "--period-anchor": ("period_anchor", str),
    "--spinup-days": ("spinup_days", int), "--crop": ("crop", str),
    "--irrigation": ("irrigation", str), "--work-dir": ("work_dir", str),
    "--output-dir": ("output_dir", str), "--precip-fallback": ("precip_fallback", str),
}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="waterbalance", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["plan", "harmonize", "run"])
    ap.add_argument("--config")
    ap.add_argument("--periods", nargs="*", help="period labels (default: last complete period)")
    ap.add_argument("--as-of", help="date the default period is computed from (default today)")
    ap.add_argument("--bbox", nargs=4, type=float, metavar=("W", "S", "E", "N"))
    ap.add_argument("--names", nargs="*", help="AOI names in --aoi-name-col")
    for flag, (field, typ) in FLAGS.items():
        ap.add_argument(flag, dest=field, type=typ)
    a = ap.parse_args(argv)

    over = {field: getattr(a, field) for field, _ in FLAGS.values()}
    over["bbox"] = a.bbox
    over["aoi_names"] = a.names
    cfg = RunConfig.from_json(a.config, **over) if a.config else \
        RunConfig(**{k: v for k, v in over.items() if v is not None})

    from .pipeline import Runner
    r = Runner(cfg)
    periods = a.periods or None
    if a.command == "plan":
        out = r.plan(periods, a.as_of)
    elif a.command == "harmonize":
        out = r.harmonize(periods, a.as_of)
    else:
        out = [{k: v for k, v in s.items() if k != "config"} for s in r.run(periods, a.as_of)]
    json.dump(out, sys.stdout, indent=2, default=str)
    print()


if __name__ == "__main__":
    main()
