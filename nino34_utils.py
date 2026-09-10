"""
nino34_utils.py — 60S–60N four-variable ENSO region utilities
=============================================================

This file is the single source of truth for low-dimensional regional indices
used by the cleaned four-variable experiment:

    sst, hc, slp, tauv

The current processed grid is expected to be cropped to 60S–60N, usually with
longitude in 0–360 coordinates (0, 2, ..., 358).  All default regions below are
inside 60S–60N.  Longitude bounds are still defined in 0–360 degrees and are
automatically adapted when a ±180 longitude array is provided.

Core supervised targets
-----------------------
Nino3.4:
    5S–5N, 170W–120W  -> lon 190–240 in 0–360 coordinates

HC-based WWV / recharge proxy:
    5S–5N, 120E–80W   -> lon 120–280 in 0–360 coordinates

Notes
-----
- ``find_nino34_indices`` and ``find_wwv_indices`` return rectangular slices for
  the dataset target builder.  They are safe for the current 0–360 grid.
- Mask-based helpers such as ``region_mask_bool`` and
  ``area_weighted_mean_series`` are safer for diagnostic regions and for
  longitude intervals that cross the dateline under a ±180 grid.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch


RegionBounds = Dict[str, Tuple[float, float] | Tuple[float, float]]


KEY_REGIONS: Dict[str, Dict[str, Tuple[float, float]]] = {
    "Nino34": {"lat": (-5.0, 5.0), "lon_360": (190.0, 240.0)},
    "Nino3": {"lat": (-5.0, 5.0), "lon_360": (210.0, 270.0)},
    "Nino4": {"lat": (-5.0, 5.0), "lon_360": (160.0, 210.0)},

    "WWV": {"lat": (-5.0, 5.0), "lon_360": (120.0, 280.0)},
    "WP_HC": {"lat": (-5.0, 5.0), "lon_360": (120.0, 170.0)},
    "CP_HC": {"lat": (-5.0, 5.0), "lon_360": (170.0, 210.0)},
    "EP_HC": {"lat": (-5.0, 5.0), "lon_360": (210.0, 280.0)},

    "EQ_SLP": {"lat": (-10.0, 10.0), "lon_360": (120.0, 280.0)},
    "WP_SLP": {"lat": (-10.0, 10.0), "lon_360": (120.0, 170.0)},
    "EP_SLP": {"lat": (-10.0, 10.0), "lon_360": (210.0, 280.0)},

    "EQ_TAUV": {"lat": (-5.0, 5.0), "lon_360": (150.0, 240.0)},
    "NH_TAUV": {"lat": (5.0, 15.0), "lon_360": (150.0, 240.0)},
    "SH_TAUV": {"lat": (-15.0, -5.0), "lon_360": (150.0, 240.0)},
    "WP_TAUV": {"lat": (-10.0, 10.0), "lon_360": (120.0, 170.0)},
    "EP_TAUV": {"lat": (-10.0, 10.0), "lon_360": (210.0, 280.0)},

    "IOB": {"lat": (-20.0, 20.0), "lon_360": (40.0, 100.0)},
    "IOD_E": {"lat": (-10.0, 0.0), "lon_360": (90.0, 110.0)},
    "IOD_W": {"lat": (-10.0, 10.0), "lon_360": (50.0, 70.0)},
    "TNA": {"lat": (5.0, 25.0), "lon_360": (305.0, 345.0)},
    "ATL3": {"lat": (-3.0, 3.0), "lon_360": (340.0, 360.0)},
    "SASD": {"lat": (-35.0, -20.0), "lon_360": (300.0, 360.0)},
}


REGION_ALIASES = {
    "Nino3.4": "Nino34",
    "WP_WarmPool": "WP_HC",
    "IO_East": "IOD_E",
    "IO_West": "IOD_W",
    "TA_Atlantic": "TNA",
}


def _as_1d_coord(values: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1-D coordinate array; got shape={arr.shape}")
    if arr.size == 0:
        raise ValueError(f"{name} coordinate is empty.")
    return arr


def _to_pm180(lon_360: float) -> float:
    """Convert a longitude in 0–360 coordinates to [-180, 180)."""
    return ((float(lon_360) + 180.0) % 360.0) - 180.0


def _resolve_lon_mask(lon: np.ndarray, lon_lo_360: float, lon_hi_360: float) -> np.ndarray:
    """Return a boolean longitude mask for bounds supplied in 0–360 degrees."""
    lon = _as_1d_coord(lon, "lon")
    lo = float(lon_lo_360)
    hi = float(lon_hi_360)

    if lo <= 0.0 and hi >= 360.0:
        return np.ones(lon.shape, dtype=bool)

    if np.nanmax(lon) > 180.0:
        lon_mod = np.mod(lon, 360.0)
        lo_mod = lo % 360.0
        hi_mod = hi % 360.0 if hi < 360.0 else 360.0
        if lo_mod <= hi_mod:
            return (lon_mod >= lo_mod) & (lon_mod <= hi_mod)
        return (lon_mod >= lo_mod) | (lon_mod <= hi_mod)

    lo_pm = _to_pm180(lo)
    hi_pm = _to_pm180(hi)
    if lo_pm <= hi_pm:
        return (lon >= lo_pm) & (lon <= hi_pm)
    return (lon >= lo_pm) | (lon <= hi_pm)


def _resolve_lat_mask(lat: np.ndarray, lat_lo: float, lat_hi: float) -> np.ndarray:
    lat = _as_1d_coord(lat, "lat")
    return (lat >= float(lat_lo)) & (lat <= float(lat_hi))


def _contiguous_slice(mask: np.ndarray, coord_name: str) -> Tuple[int, int]:
    idx = np.where(mask)[0]
    if idx.size == 0:
        raise ValueError(f"No {coord_name} points selected by region mask.")
    if np.any(np.diff(idx) != 1):
        raise ValueError(
            f"{coord_name} selection is not contiguous. This usually means the "
            "region crosses the longitude seam in the current coordinate system. "
            "Use mask-based utilities or convert longitudes to 0–360 before using "
            "slice-based dataset targets."
        )
    return int(idx[0]), int(idx[-1]) + 1


def region_bounds(region_name: str) -> Dict[str, Tuple[float, float]]:
    """Return canonical region bounds by name, resolving backward-compatible aliases."""
    key = REGION_ALIASES.get(region_name, region_name)
    if key not in KEY_REGIONS:
        raise ValueError(f"Unknown region '{region_name}'. Valid: {sorted(KEY_REGIONS)}")
    return KEY_REGIONS[key]


def region_mask_bool(
    lat: np.ndarray,
    lon: np.ndarray,
    lat_lo: float,
    lat_hi: float,
    lon_lo_360: float,
    lon_hi_360: float,
) -> np.ndarray:
    """Return an ``(H, W)`` boolean mask for a rectangular region."""
    lat_m = _resolve_lat_mask(lat, lat_lo, lat_hi)
    lon_m = _resolve_lon_mask(lon, lon_lo_360, lon_hi_360)
    if not lat_m.any():
        raise ValueError(
            f"No latitude points in [{lat_lo}, {lat_hi}]. "
            f"Lat range: [{float(np.min(lat)):.1f}, {float(np.max(lat)):.1f}]"
        )
    if not lon_m.any():
        raise ValueError(
            f"No longitude points in [{lon_lo_360}, {lon_hi_360}] (0–360). "
            f"Lon range: [{float(np.min(lon)):.1f}, {float(np.max(lon)):.1f}]"
        )
    return lat_m[:, None] & lon_m[None, :]


def get_region_mask(lat: np.ndarray, lon: np.ndarray, region_name: str) -> np.ndarray:
    """Return an ``(H, W)`` boolean mask for a named region."""
    r = region_bounds(region_name)
    lat_lo, lat_hi = r["lat"]
    lon_lo, lon_hi = r["lon_360"]
    return region_mask_bool(lat, lon, lat_lo, lat_hi, lon_lo, lon_hi)


def make_region_mask(
    lat: np.ndarray,
    lon: np.ndarray,
    lat_lo: float,
    lat_hi: float,
    lon_lo_360: float,
    lon_hi_360: float,
) -> torch.Tensor:
    """Return an ``(H, W)`` float32 torch mask for a region."""
    return torch.from_numpy(
        region_mask_bool(lat, lon, lat_lo, lat_hi, lon_lo_360, lon_hi_360).astype(np.float32)
    )


def find_region_indices(
    lat: np.ndarray,
    lon: np.ndarray,
    lat_lo: float,
    lat_hi: float,
    lon_lo_360: float,
    lon_hi_360: float,
) -> Tuple[int, int, int, int]:
    """Return contiguous slice indices ``lat_s, lat_e, lon_s, lon_e``.

    This function is kept for the dataset target builder.  It is correct for the
    current 0–360 grid and the target regions Nino3.4 / WWV.  For wrapped
    longitude selections under a ±180 grid, use mask-based helpers instead.
    """
    mask = region_mask_bool(lat, lon, lat_lo, lat_hi, lon_lo_360, lon_hi_360)
    lat_s, lat_e = _contiguous_slice(mask.any(axis=1), "latitude")
    lon_s, lon_e = _contiguous_slice(mask.any(axis=0), "longitude")
    return lat_s, lat_e, lon_s, lon_e


def find_nino34_indices(lat: np.ndarray, lon: np.ndarray) -> Tuple[int, int, int, int]:
    """Return slice indices for Nino3.4: 5S–5N, 170W–120W."""
    r = KEY_REGIONS["Nino34"]
    return find_region_indices(lat, lon, *r["lat"], *r["lon_360"])


def find_wwv_indices(lat: np.ndarray, lon: np.ndarray) -> Tuple[int, int, int, int]:
    """Return slice indices for HC-based WWV/recharge proxy: 5S–5N, 120E–80W."""
    r = KEY_REGIONS["WWV"]
    return find_region_indices(lat, lon, *r["lat"], *r["lon_360"])


def nino34_mask_bool(lat: np.ndarray, lon: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return separate latitude and longitude boolean masks for Nino3.4."""
    r = KEY_REGIONS["Nino34"]
    lat_mask = _resolve_lat_mask(lat, *r["lat"])
    lon_mask = _resolve_lon_mask(lon, *r["lon_360"])
    return lat_mask, lon_mask


def area_weighted_mean_series(
    field: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    lat_lo: float,
    lat_hi: float,
    lon_lo_360: float,
    lon_hi_360: float,
) -> np.ndarray:
    """Compute a cosine-latitude-weighted regional mean time series.

    Parameters
    ----------
    field : np.ndarray, shape (T, H, W)
        Time-varying spatial field.
    lat, lon : np.ndarray
        1-D coordinates.
    region bounds : float
        Region definition in degrees, longitude in 0–360 coordinates.

    Returns
    -------
    np.ndarray, shape (T,)
        Regional mean series.
    """
    arr = np.asarray(field, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"field must have shape (T, H, W); got {arr.shape}")
    lat = _as_1d_coord(lat, "lat")
    lon = _as_1d_coord(lon, "lon")
    if arr.shape[-2:] != (lat.size, lon.size):
        raise ValueError(
            f"field spatial shape {arr.shape[-2:]} does not match "
            f"lat/lon sizes {(lat.size, lon.size)}"
        )

    mask = region_mask_bool(lat, lon, lat_lo, lat_hi, lon_lo_360, lon_hi_360)
    weights = np.cos(np.deg2rad(lat)).astype(np.float32)[:, None] * mask.astype(np.float32)
    denom = float(weights.sum())
    if denom <= 0.0:
        raise ValueError("Invalid regional weights: sum <= 0")
    return (np.nansum(arr * weights[None, :, :], axis=(1, 2)) / denom).astype(np.float32)


def named_area_weighted_mean_series(
    field: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    region_name: str,
) -> np.ndarray:
    """Compute a cosine-latitude-weighted time series for a named region."""
    r = region_bounds(region_name)
    return area_weighted_mean_series(field, lat, lon, *r["lat"], *r["lon_360"])


def validate_key_regions(lat: np.ndarray, lon: np.ndarray) -> Dict[str, Dict[str, float | int | Tuple[float, float]]]:
    """Validate that all named regions have at least one grid cell.

    Returns a small report dictionary with selected cell counts and the actual
    coordinate span covered by each mask.  This is useful after changing the
    spatial crop or longitude convention.
    """
    lat = _as_1d_coord(lat, "lat")
    lon = _as_1d_coord(lon, "lon")
    report: Dict[str, Dict[str, float | int | Tuple[float, float]]] = {}
    for name, r in KEY_REGIONS.items():
        mask = get_region_mask(lat, lon, name)
        lat_idx = np.where(mask.any(axis=1))[0]
        lon_idx = np.where(mask.any(axis=0))[0]
        if lat_idx.size == 0 or lon_idx.size == 0:
            raise ValueError(f"Region {name} has no selected grid cells.")
        report[name] = {
            "n_cells": int(mask.sum()),
            "n_lat": int(lat_idx.size),
            "n_lon": int(lon_idx.size),
            "lat_span": (float(lat[lat_idx[0]]), float(lat[lat_idx[-1]])),
            "lon_span": (float(lon[lon_idx[0]]), float(lon[lon_idx[-1]])),
        }
    return report


if __name__ == "__main__":
    lat_demo = np.arange(-60.0, 60.0 + 1.0, 1.0)
    lon_demo = np.arange(0.0, 360.0, 2.0)
    rep = validate_key_regions(lat_demo, lon_demo)
    print(f"Validated {len(rep)} regions on demo 60S–60N / 0–358 grid.")
    print("Nino34:", find_nino34_indices(lat_demo, lon_demo), rep["Nino34"])
    print("WWV:", find_wwv_indices(lat_demo, lon_demo), rep["WWV"])
