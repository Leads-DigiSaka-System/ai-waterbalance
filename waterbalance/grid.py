"""The harmonized pixel grid and its tiles.

Every dataset is delivered onto ONE grid: pixel edges at integer multiples of the resolution in the
grid CRS (as in ai-cropgrowth), so separate runs and areas line up exactly and can be mosaicked
without resampling.

  EPSG:4326 (default)  resolution = resolution_m / 111 320 deg; a pixel is 1 ha at the equator and
                       cos(latitude) ha elsewhere (0.97 ha at 15 N). Areas are computed exactly.
  projected (UTM ...)  resolution = resolution_m metres; 100 m = exactly 1 ha.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

M_PER_DEG = 111_320.0
EARTH_R = 6_371_008.8


def is_geographic(crs):
    return str(crs).upper() in ("EPSG:4326", "WGS84", "OGC:CRS84")


def res_in_crs(resolution_m, crs):
    return resolution_m / M_PER_DEG if is_geographic(crs) else float(resolution_m)


@dataclass(frozen=True)
class Grid:
    """Pixel grid: top-left corner (x0, y1), size in pixels, square pixels of `res` CRS units."""
    crs: str
    res: float
    x0: float
    y1: float
    width: int
    height: int

    # ---------------------------------------------------------- construction
    @classmethod
    def for_bounds(cls, bounds, res, crs):
        """Global-aligned grid covering bounds (in grid CRS units)."""
        w, s, e, n = bounds
        eps = res * 1e-6
        x0 = math.floor((w + eps) / res) * res
        y1 = math.ceil((n - eps) / res) * res
        x1 = math.ceil((e - eps) / res) * res
        y0 = math.floor((s + eps) / res) * res
        return cls(crs, res, x0, y1, max(1, int(round((x1 - x0) / res))),
                   max(1, int(round((y1 - y0) / res))))

    # ---------------------------------------------------------- geometry
    @property
    def bounds(self):
        return (self.x0, self.y1 - self.height * self.res, self.x0 + self.width * self.res, self.y1)

    @property
    def xs(self):
        return self.x0 + (np.arange(self.width) + 0.5) * self.res

    @property
    def ys(self):
        return self.y1 - (np.arange(self.height) + 0.5) * self.res

    @property
    def transform(self):
        from rasterio.transform import from_origin
        return from_origin(self.x0, self.y1, self.res, self.res)

    def affine_dict(self):
        """Earth Engine computePixels grid."""
        return {
            "dimensions": {"width": int(self.width), "height": int(self.height)},
            "affineTransform": {"scaleX": self.res, "shearX": 0, "translateX": self.x0,
                                "shearY": 0, "scaleY": -self.res, "translateY": self.y1},
            "crsCode": self.crs,
        }

    def subgrid(self, col0, row0, width, height):
        return Grid(self.crs, self.res, self.x0 + col0 * self.res, self.y1 - row0 * self.res,
                    width, height)

    def offset_in(self, parent):
        """(col, row) of this grid's top-left pixel inside `parent` (same res, aligned)."""
        return (int(round((self.x0 - parent.x0) / self.res)),
                int(round((parent.y1 - self.y1) / self.res)))

    def pixel_area_ha(self):
        """(height, width) pixel areas in hectares."""
        if is_geographic(self.crs):
            lat_top = np.radians(self.y1 - np.arange(self.height) * self.res)
            lat_bot = np.radians(self.y1 - (np.arange(self.height) + 1) * self.res)
            row = EARTH_R ** 2 * np.radians(self.res) * np.abs(np.sin(lat_top) - np.sin(lat_bot))
            return np.repeat((row / 1e4)[:, None], self.width, axis=1)
        return np.full((self.height, self.width), self.res ** 2 / 1e4)

    # ---------------------------------------------------------- tiles
    def tiles(self, tile_px, geometry=None):
        """[(tile_id, Grid)] of tile_px x tile_px blocks; with geometry (in grid CRS), blocks not
        touching it are dropped."""
        out = []
        for r in range(0, self.height, tile_px):
            for c in range(0, self.width, tile_px):
                g = self.subgrid(c, r, min(tile_px, self.width - c), min(tile_px, self.height - r))
                out.append((f"r{r // tile_px:04d}c{c // tile_px:04d}", g))
        if geometry is not None:
            from shapely.geometry import box
            from shapely.prepared import prep
            pg = prep(geometry)
            out = [(t, g) for t, g in out if pg.intersects(box(*g.bounds))]
        return out

    def key(self):
        return f"{self.crs}|{self.res:.10g}|{self.x0:.10g}|{self.y1:.10g}|{self.width}|{self.height}"


def lonlat_to_crs(geom, crs):
    """Shapely geometry from EPSG:4326 to crs."""
    if is_geographic(crs):
        return geom
    from pyproj import Transformer
    from shapely.ops import transform
    tf = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    return transform(tf.transform, geom)


def geometry_mask(grid, geometry):
    """(height, width) bool: True inside geometry (grid CRS), all_touched."""
    from rasterio.features import geometry_mask as gm
    return ~gm([geometry], out_shape=(grid.height, grid.width), transform=grid.transform,
               all_touched=True)
