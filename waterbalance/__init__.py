"""Water availability (f) from a daily root-zone water balance on a harmonized 1 ha grid.

  P - R - ET (+ I - D) = dS       f = (S - WP) / (FC - WP)

Inputs (Google Earth Engine, read with computePixels): CHIRPS daily precipitation, ERA5-Land daily
weather (FAO-56 reference ET), ESA WorldCover cropland, OpenLandMap soil texture.
"""
from .config import CROPS, DATASETS, RunConfig
from .pipeline import Runner

__all__ = ["RunConfig", "Runner", "CROPS", "DATASETS"]
