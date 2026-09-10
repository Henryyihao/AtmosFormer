"""
Flexible ENSO global-mechanism dataset for CTM experiments.

This replaces the old fixed four-variable loader.  It supports the new global
mechanism NetCDF files described by the user:

- CMIP6 map variables: (model, time, lat, lon)
- OBS map variables:   (time, lat, lon)
- physical indices:    (model, time) or (time)
- optional lead features are intentionally not fed as history features; target
  month / SPB information is reconstructed from the calendar in the model.

Dataset output:
    x_map:        (input_len, C, H, W)
    x_phys:       (input_len, P)       P may be zero
    y:            (output_len, D)      default D=2: [nino34, wwv]
    init_month:   scalar long, 0=Jan, month of the last input frame
    target_years: (output_len,)
    segment_id:   scalar long
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset

from nino34_utils import find_nino34_indices, find_wwv_indices

FOUR_VAR_NAMES = ["sst", "hc300", "slp", "tauv"]

MAP_VAR_GROUPS: Dict[str, List[str]] = {
    "core4": ["sst", "hc300", "slp", "tauv"],
    "core5": ["sst", "hc300", "slp", "tauv", "tauu"],
    "full8": ["sst", "hc300", "slp", "tauv", "tauu", "zos", "hfds", "mld"],
}

PHYS_INDEX_GROUPS: Dict[str, List[str]] = {
    "none": [],
    "core": [
        "nino34", "nino3", "nino4", "nino12", "emi", "cp_ep_contrast",
    ],
    "recharge": [
        "hc_eq", "hc_west", "hc_central", "hc_east", "wwv", "recharge_tendency",
        "thermocline_tilt", "hc_gradient_we",
    ],
    "wind": [
        "tauu_wcp", "tauu_cp", "tauv_eq", "tauv_nh", "tauv_sh", "tauv_antisym",
        "wwb_proxy", "wwb_state_dependent", "bjerknes_proxy", "thermocline_feedback",
        "soi_like",
    ],
    "cross_basin": [
        "dmi", "iod_w", "iod_e", "iob", "tna", "atl3", "sasd",
        "pmm_like", "npmm_like", "npo_like_slp", "pdo_like", "ipo_tripole_like",
    ],
}
PHYS_INDEX_GROUPS["mechanism"] = (
    PHYS_INDEX_GROUPS["core"]
    + PHYS_INDEX_GROUPS["recharge"]
    + PHYS_INDEX_GROUPS["wind"]
    + PHYS_INDEX_GROUPS["cross_basin"]
)

VAR_ALIASES: Dict[str, List[str]] = {
    "hc300": ["hc300", "hc", "ohc300", "heat_content_300"],
    "hc": ["hc", "hc300", "ohc300", "heat_content_300"],
    "sst": ["sst", "tos", "ssta"],
    "slp": ["slp", "psl", "msl"],
    "tauu": ["tauu", "uflx", "uas_tau", "tx"],
    "tauv": ["tauv", "vflx", "vas_tau", "ty"],
    "zos": ["zos", "ssh", "sshg"],
    "hfds": ["hfds", "qnet", "thflx", "heat_flux"],
    "mld": ["mld", "dbss_obml", "mixed_layer_depth"],
    "wwv": ["wwv", "hc_eq", "wwv_proxy"],
    "nino34": ["nino34", "nino3.4", "nino_34"],
    "thermocline_tilt": ["thermocline_tilt", "hc_gradient_we", "tc_tilt"],
}

DEFAULT_NOISE_STD = {
    "sst": 0.03,
    "hc300": 0.015,
    "hc": 0.015,
    "slp": 0.04,
    "tauu": 0.04,
    "tauv": 0.04,
    "zos": 0.02,
    "hfds": 0.03,
    "mld": 0.03,
}

CALENDAR_FEATURE_NAMES = [
    "input_month_sin",
    "input_month_cos",
    "init_month_sin",
    "init_month_cos",
]

BASIN_REGIONS = {
    "trop_pacific": (-10.0, 10.0, 120.0, 280.0),
    "north_pacific": (10.0, 40.0, 120.0, 280.0),
    "south_pacific": (-40.0, -10.0, 120.0, 290.0),
    "indian": (-30.0, 30.0, 30.0, 120.0),
    "atlantic": (-30.0, 30.0, 280.0, 20.0),
}


def _region_mask(
    lat: np.ndarray,
    lon: np.ndarray,
    region_name: str,
) -> np.ndarray:
    if region_name not in BASIN_REGIONS:
        raise ValueError(
            f"Unknown basin {region_name!r}; available={list(BASIN_REGIONS)}"
        )
    lat_min, lat_max, lon_min, lon_max = BASIN_REGIONS[region_name]
    lat_keep = (lat >= lat_min) & (lat <= lat_max)
    lon_360 = lon % 360.0
    if lon_min <= lon_max:
        lon_keep = (lon_360 >= lon_min) & (lon_360 <= lon_max)
    else:
        lon_keep = (lon_360 >= lon_min) | (lon_360 <= lon_max)
    return lat_keep[:, None] & lon_keep[None, :]


def build_spatial_input_mask(
    map_vars: Sequence[str],
    lat: np.ndarray,
    lon: np.ndarray,
    specification: str,
) -> np.ndarray:
    """Build a `(variable, lat, lon)` keep mask for basin experiments.

    Supported forms are:

    - `full`: all grid cells;
    - `state_only`: no map cells (shared scalar-state baseline);
    - `five_basins`: union of all configured basins;
    - `basins=trop_pacific+indian`: union of selected basins;
    - `only_variables=slp+sst`: selected variables on the full grid;
    - `without_basin=atlantic`: all configured basins except one;
    - `only_group=indian__sst`: one basin-variable group;
    - `without_group=indian__sst`: all groups except one;
    - `only_groups=indian__sst+atlantic__sst`: selected groups;
    - `without_groups=indian__sst+atlantic__sst`: remove selected groups.
    """
    specification = str(specification or "full").strip().lower()
    shape = (len(map_vars), len(lat), len(lon))
    if specification == "full":
        return np.ones(shape, dtype=np.float32)
    if specification == "state_only":
        return np.zeros(shape, dtype=np.float32)

    basin_masks = {
        name: _region_mask(lat, lon, name) for name in BASIN_REGIONS
    }
    five_basin_union = np.logical_or.reduce(list(basin_masks.values()))

    def parse_names(value: str) -> List[str]:
        return [item.strip() for item in value.split("+") if item.strip()]

    def parse_groups(value: str) -> List[Tuple[str, str]]:
        groups = []
        for item in parse_names(value):
            pieces = item.split("__", 1)
            if len(pieces) != 2:
                raise ValueError(
                    f"Invalid basin-variable group {item!r}; expected basin__variable"
                )
            basin, variable = pieces
            if basin not in basin_masks:
                raise ValueError(f"Unknown basin in group {item!r}")
            if variable not in map_vars:
                raise ValueError(
                    f"Variable {variable!r} is not loaded; loaded={list(map_vars)}"
                )
            groups.append((basin, variable))
        if not groups:
            raise ValueError(f"No groups in spatial mask {specification!r}")
        return groups

    if specification == "five_basins":
        return np.broadcast_to(five_basin_union, shape).astype(np.float32).copy()
    if specification.startswith("basins="):
        names = parse_names(specification.split("=", 1)[1])
        unknown = [name for name in names if name not in basin_masks]
        if unknown:
            raise ValueError(f"Unknown basins {unknown}; available={list(basin_masks)}")
        active = np.logical_or.reduce([basin_masks[name] for name in names])
        return np.broadcast_to(active, shape).astype(np.float32).copy()
    if specification.startswith("without_basin="):
        excluded = specification.split("=", 1)[1].strip()
        if excluded not in basin_masks:
            raise ValueError(f"Unknown excluded basin {excluded!r}")
        names = [name for name in basin_masks if name != excluded]
        active = np.logical_or.reduce([basin_masks[name] for name in names])
        return np.broadcast_to(active, shape).astype(np.float32).copy()

    variable_mode, variable_separator, variable_value = specification.partition("=")
    if variable_separator and variable_mode in {
        "only_variable", "only_variables", "without_variable", "without_variables",
    }:
        selected = parse_names(variable_value)
        unknown = [name for name in selected if name not in map_vars]
        if unknown:
            raise ValueError(
                f"Unknown variables {unknown}; loaded={list(map_vars)}"
            )
        mask = (
            np.zeros(shape, dtype=np.float32)
            if variable_mode.startswith("only")
            else np.ones(shape, dtype=np.float32)
        )
        variable_lookup = {name: index for index, name in enumerate(map_vars)}
        for variable in selected:
            mask[variable_lookup[variable], :, :] = (
                1.0 if variable_mode.startswith("only") else 0.0
            )
        return mask

    mode, separator, value = specification.partition("=")
    if not separator or mode not in {
        "only_group",
        "only_groups",
        "without_group",
        "without_groups",
    }:
        raise ValueError(f"Unsupported spatial mask specification {specification!r}")
    groups = parse_groups(value)
    if mode.startswith("only"):
        mask = np.zeros(shape, dtype=np.float32)
    else:
        mask = np.broadcast_to(five_basin_union, shape).astype(np.float32).copy()
    variable_lookup = {name: index for index, name in enumerate(map_vars)}
    for basin, variable in groups:
        variable_index = variable_lookup[variable]
        mask[variable_index, basin_masks[basin]] = (
            1.0 if mode.startswith("only") else 0.0
        )
    return mask


@dataclass
class MechanismData:
    maps: Dict[str, np.ndarray]
    physics: Dict[str, np.ndarray]
    targets: Dict[str, np.ndarray]
    segments: List[Tuple[int, int]]
    lat: np.ndarray
    lon: np.ndarray
    time: Optional[object]
    map_vars: List[str]
    phys_vars: List[str]
    target_vars: List[str]


def _parse_csv(value: Optional[str]) -> Optional[List[str]]:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [str(x).strip() for x in value if str(x).strip()]
    value = str(value).strip()
    if value == "":
        return []
    return [x.strip() for x in value.split(",") if x.strip()]


def _dedupe_names(names: Sequence[str]) -> List[str]:
    """Remove duplicate variable names while preserving channel order."""
    return list(dict.fromkeys(str(name).strip() for name in names if str(name).strip()))


def resolve_map_vars(var_group: str = "full8", map_vars: Optional[str] = None) -> List[str]:
    custom = _parse_csv(map_vars)
    if custom is not None:
        custom = _dedupe_names(custom)
        if not custom:
            raise ValueError("--map_vars must contain at least one map variable.")
        return custom
    if var_group not in MAP_VAR_GROUPS:
        raise ValueError(f"Unknown var_group={var_group}. Available={sorted(MAP_VAR_GROUPS)}")
    return _dedupe_names(MAP_VAR_GROUPS[var_group])


def resolve_phys_vars(phys_group: str = "mechanism", phys_vars: Optional[str] = None, no_phys: bool = False) -> List[str]:
    if no_phys:
        return []
    custom = _parse_csv(phys_vars)
    if custom is not None:
        if len(custom) == 1 and custom[0].lower() == "none":
            return []
        return _dedupe_names(custom)
    if phys_group not in PHYS_INDEX_GROUPS:
        raise ValueError(f"Unknown phys_group={phys_group}. Available={sorted(PHYS_INDEX_GROUPS)}")
    return _dedupe_names(PHYS_INDEX_GROUPS[phys_group])


def resolve_target_vars(target_vars: Optional[str] = None) -> List[str]:
    custom = _parse_csv(target_vars)
    return _dedupe_names(custom) if custom else ["nino34", "wwv"]


def _canonical_candidates(name: str) -> List[str]:
    return VAR_ALIASES.get(name, [name])


def _find_var(ds: xr.Dataset, canonical: str) -> Optional[str]:
    for cand in _canonical_candidates(canonical):
        if cand in ds.data_vars:
            return cand
    return canonical if canonical in ds.data_vars else None


def _detect_model_dim(ds: xr.Dataset) -> Optional[str]:
    for d in ("model", "source_id", "member_id", "member", "ensemble"):
        if d in ds.dims:
            return d
    return None


def _da_to_numpy_map(da: xr.DataArray, model_dim: Optional[str]) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    if "time" not in da.dims:
        raise ValueError(f"Map variable {da.name} has no time dimension: dims={da.dims}")
    if "lat" not in da.dims or "lon" not in da.dims:
        raise ValueError(
            f"Map variable {da.name} must have lat/lon dimensions: dims={da.dims}"
        )
    allowed_dims = {"time", "lat", "lon"}
    if model_dim is not None:
        allowed_dims.add(model_dim)
    extra_dims = set(da.dims) - allowed_dims
    if extra_dims:
        raise ValueError(
            f"Map variable {da.name} has unsupported dimensions {sorted(extra_dims)}: "
            f"dims={da.dims}"
        )
    if model_dim is not None and model_dim in da.dims:
        vals = np.asarray(da.transpose(model_dim, "time", "lat", "lon").values, dtype=np.float32)
        n_model, n_time = vals.shape[0], vals.shape[1]
        arr = vals.reshape(n_model * n_time, vals.shape[-2], vals.shape[-1])
        segs = [(i * n_time, (i + 1) * n_time) for i in range(n_model)]
        return arr.astype(np.float32), segs
    vals = np.asarray(da.transpose("time", "lat", "lon").values, dtype=np.float32)
    return vals.astype(np.float32), [(0, vals.shape[0])]


def _da_to_numpy_series(da: xr.DataArray, model_dim: Optional[str]) -> Optional[np.ndarray]:
    if "time" not in da.dims:
        return None
    if "lead" in da.dims:
        return None
    allowed_dims = {"time"}
    if model_dim is not None:
        allowed_dims.add(model_dim)
    if set(da.dims) - allowed_dims:
        return None
    if model_dim is not None and model_dim in da.dims:
        vals = np.asarray(da.transpose(model_dim, "time").values, dtype=np.float32)
        return vals.reshape(-1).astype(np.float32)
    vals = np.asarray(da.transpose("time").values, dtype=np.float32)
    return vals.reshape(-1).astype(np.float32)


def _crop_lat_arrays(maps: Dict[str, np.ndarray], lat: np.ndarray, lat_south: Optional[float], lat_north: Optional[float]):
    if lat_south is None or lat_north is None:
        return maps, lat
    mask = (lat >= lat_south) & (lat <= lat_north)
    idx = np.where(mask)[0]
    if len(idx) == 0:
        raise ValueError(f"No latitude points in [{lat_south}, {lat_north}], available {lat.min()}..{lat.max()}")
    s, e = int(idx[0]), int(idx[-1] + 1)
    return {
        k: np.ascontiguousarray(v[:, s:e, :], dtype=np.float32)
        for k, v in maps.items()
    }, np.ascontiguousarray(lat[s:e], dtype=np.float32)


def _make_region_weights(lat: np.ndarray, slices: Tuple[int, int, int, int]) -> Tuple[np.ndarray, float]:
    lat_s, lat_e, lon_s, lon_e = slices
    lat_region = lat[lat_s:lat_e]
    w = np.cos(np.deg2rad(lat_region)).astype(np.float32)[:, None]
    denom = float(w.sum() * (lon_e - lon_s))
    if denom <= 0:
        raise ValueError("Invalid regional weights.")
    return w, denom


def _calc_region_index(field: np.ndarray, lat: np.ndarray, lon: np.ndarray, region: str) -> np.ndarray:
    slices = find_nino34_indices(lat, lon) if region == "nino34" else find_wwv_indices(lat, lon)
    w, denom = _make_region_weights(lat, slices)
    lat_s, lat_e, lon_s, lon_e = slices
    sub = field[:, lat_s:lat_e, lon_s:lon_e]
    return (np.nansum(sub * w[None, :, :], axis=(1, 2)) / denom).astype(np.float32)


def _find_region_slice(lat: np.ndarray, lon: np.ndarray,
                       lat_s: float, lat_n: float, lon_w: float, lon_e: float) -> Tuple[int, int, int, int]:
    """Find array indices for a lat/lon box. Returns (lat_start, lat_end, lon_start, lon_end)."""
    lat_mask = (lat >= lat_s) & (lat <= lat_n)
    lon_360 = lon % 360
    lon_mask = (lon_360 >= lon_w) & (lon_360 <= lon_e)
    lat_idx = np.where(lat_mask)[0]
    lon_idx = np.where(lon_mask)[0]
    if len(lat_idx) == 0 or len(lon_idx) == 0:
        raise ValueError(f"No grid points in region [{lat_s},{lat_n}]×[{lon_w},{lon_e}]")
    return int(lat_idx[0]), int(lat_idx[-1] + 1), int(lon_idx[0]), int(lon_idx[-1] + 1)


def load_mechanism_file(
    path: str,
    map_vars: Sequence[str],
    phys_vars: Sequence[str],
    target_vars: Sequence[str],
    lat_south: Optional[float] = -60.0,
    lat_north: Optional[float] = 60.0,
    strict_phys: bool = False,
) -> MechanismData:
    ds = xr.open_dataset(path)
    model_dim = _detect_model_dim(ds)
    lat_full = np.asarray(ds["lat"].values, dtype=np.float32)
    lon = np.asarray(ds["lon"].values, dtype=np.float32)
    lat_slice = slice(None)
    if lat_south is not None and lat_north is not None:
        selected = np.flatnonzero((lat_full >= lat_south) & (lat_full <= lat_north))
        if selected.size == 0:
            raise ValueError(
                f"No latitude points in [{lat_south}, {lat_north}], "
                f"available {lat_full.min()}..{lat_full.max()}"
            )
        lat_slice = slice(int(selected[0]), int(selected[-1]) + 1)
    lat = np.ascontiguousarray(lat_full[lat_slice], dtype=np.float32)

    maps: Dict[str, np.ndarray] = {}
    segments: Optional[List[Tuple[int, int]]] = None
    for var in map_vars:
        name = _find_var(ds, var)
        if name is None:
            raise KeyError(f"Missing map variable '{var}'. Available variables include: {list(ds.data_vars)[:30]}")
        map_da = ds[name].isel(lat=lat_slice)
        arr, segs = _da_to_numpy_map(map_da, model_dim)
        maps[var] = arr
        if segments is None:
            segments = segs
        elif segments != segs:
            raise ValueError(f"Variable {var} has incompatible segmentation.")

    if segments is None:
        raise ValueError("No map variables loaded.")
    n_time_total = next(iter(maps.values())).shape[0]

    physics: Dict[str, np.ndarray] = {}
    skipped_phys: List[str] = []
    for var in phys_vars:
        name = _find_var(ds, var)
        if name is None:
            skipped_phys.append(f"{var} (missing)")
            continue
        arr = _da_to_numpy_series(ds[name], model_dim)
        if arr is None:
            skipped_phys.append(f"{var} (not a one-dimensional history series)")
            continue
        if arr.shape[0] == n_time_total:
            physics[var] = arr.astype(np.float32)
        else:
            skipped_phys.append(
                f"{var} (length {arr.shape[0]} != expected {n_time_total})"
            )
    if skipped_phys:
        message = f"Physical indices not loaded from {path}: {', '.join(skipped_phys)}"
        if strict_phys:
            raise KeyError(message)
        print(f"  [Warning] {message}")

    targets: Dict[str, np.ndarray] = {}
    for var in target_vars:
        name = _find_var(ds, var)
        arr = None
        if name is not None:
            arr = _da_to_numpy_series(ds[name], model_dim)
        if arr is None or arr.shape[0] != n_time_total:
            if var == "nino34":
                if "sst" not in maps:
                    raise KeyError("Target nino34 missing and sst map is not loaded for fallback calculation.")
                arr = _calc_region_index(maps["sst"], lat, lon, "nino34")
            elif var == "wwv":
                hc_name = "hc300" if "hc300" in maps else ("hc" if "hc" in maps else None)
                if hc_name is None:
                    raise KeyError("Target wwv missing and hc300 map is not loaded for fallback calculation.")
                arr = _calc_region_index(maps[hc_name], lat, lon, "wwv")
            elif var == "thermocline_tilt":
                hc_name = "hc300" if "hc300" in maps else ("hc" if "hc" in maps else None)
                if hc_name is None:
                    raise KeyError("Target thermocline_tilt missing and hc300 map not loaded for fallback.")
                hc_field = maps[hc_name]
                wp_slices = _find_region_slice(lat, lon, -5.0, 5.0, 120.0, 170.0)
                ep_slices = _find_region_slice(lat, lon, -5.0, 5.0, 210.0, 280.0)
                wp_w, wp_d = _make_region_weights(lat, wp_slices)
                ep_w, ep_d = _make_region_weights(lat, ep_slices)
                la_s, la_e, lo_s, lo_e = wp_slices
                hc_west = (np.nansum(hc_field[:, la_s:la_e, lo_s:lo_e] * wp_w[None, :, :], axis=(1, 2)) / wp_d)
                la_s, la_e, lo_s, lo_e = ep_slices
                hc_east = (np.nansum(hc_field[:, la_s:la_e, lo_s:lo_e] * ep_w[None, :, :], axis=(1, 2)) / ep_d)
                arr = (hc_west - hc_east).astype(np.float32)
            else:
                raise KeyError(f"Target variable '{var}' is missing or lead-dependent and cannot be built.")
        targets[var] = np.asarray(arr, dtype=np.float32)

    time_coord = ds["time"] if "time" in ds.coords else None
    print(f"  Loaded {path}")
    print(f"  model_dim={model_dim}, total_months={n_time_total}, segments={len(segments)}, spatial={len(lat)}x{len(lon)}")
    print(f"  maps={list(maps)}, physics={list(physics)}, targets={list(targets)}")
    return MechanismData(
        maps=maps, physics=physics, targets=targets, segments=list(segments), lat=lat, lon=lon,
        time=time_coord, map_vars=list(map_vars), phys_vars=list(physics.keys()), target_vars=list(targets.keys())
    )


def split_segments(segments: Sequence[Tuple[int, int]], val_months_per_model: int, window: int) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    val_months_per_model = max(int(val_months_per_model), int(window))
    train: List[Tuple[int, int]] = []
    val: List[Tuple[int, int]] = []
    for s, e in segments:
        s, e = int(s), int(e)
        if e - s <= val_months_per_model + window:
            train.append((s, e))
        else:
            split = e - val_months_per_model
            train.append((s, split))
            val.append((split, e))
    if not val:
        i = max(range(len(segments)), key=lambda k: segments[k][1] - segments[k][0])
        s, e = segments[i]
        split = max(s + window, e - val_months_per_model)
        train[i] = (s, split)
        val.append((split, e))
    print(f"  Split: train_months={sum(e-s for s,e in train)}, val_months={sum(e-s for s,e in val)}")
    return train, val


def _concat_segments(arr: np.ndarray, segments: Sequence[Tuple[int, int]]) -> np.ndarray:
    pieces = [arr[int(s):int(e)] for s, e in segments if int(e) > int(s)]
    if not pieces:
        return arr
    return np.concatenate(pieces, axis=0)


class ENSOGlobalMechanismDataset(Dataset):
    CLIP_SIGMA = 5.0

    def __init__(
        self,
        data: MechanismData,
        input_len: int,
        output_len: int,
        is_train: bool = True,
        stats: Optional[Dict[str, float]] = None,
        segments: Optional[Sequence[Tuple[int, int]]] = None,
        start_month: int = 0,
        start_year: int = 1900,
        phys_scaler: str = "robust",
        include_calendar_features: bool = True,
        map_noise: bool = True,
        exclude_target_years: Optional[Sequence[int]] = None,
        keep_only_target_years: Optional[Sequence[int]] = None,
        spatial_mask_spec: str = "full",
    ):
        super().__init__()
        self._exclude_target_years = set(int(y) for y in exclude_target_years) if exclude_target_years else None
        self._keep_only_target_years = set(int(y) for y in keep_only_target_years) if keep_only_target_years else None
        self.data = data
        self.input_len = int(input_len)
        self.output_len = int(output_len)
        self.window = self.input_len + self.output_len
        self.is_train = bool(is_train)
        self.start_month = int(start_month)
        self.start_year = int(start_year)
        self.phys_scaler = str(phys_scaler)
        self.include_calendar_features = bool(include_calendar_features)
        self.map_noise = bool(map_noise)
        self.map_vars = list(data.map_vars)
        self.base_phys_vars = list(data.phys_vars)
        self.target_vars = list(data.target_vars)
        self.lat = data.lat
        self.lon = data.lon
        self.n_vars = len(self.map_vars)
        self.target_dim = len(self.target_vars)
        self.spatial_mask_spec = str(spatial_mask_spec or "full")
        self.spatial_input_mask = build_spatial_input_mask(
            self.map_vars,
            self.lat,
            self.lon,
            self.spatial_mask_spec,
        )
        active_fractions = {
            name: float(self.spatial_input_mask[index].mean())
            for index, name in enumerate(self.map_vars)
        }
        print(
            f"  [SpatialMask] spec={self.spatial_mask_spec} "
            f"active_fraction={active_fractions}"
        )
        self.segments = list(segments) if segments is not None else list(data.segments)

        if stats is None:
            self.stats = self._compute_stats(self.segments)
        else:
            self.stats = dict(stats)

        self.phys_feature_names = list(self.base_phys_vars)
        if self.include_calendar_features:
            self.phys_feature_names += CALENDAR_FEATURE_NAMES
        self.phys_dim = len(self.phys_feature_names)

        self._maps = self._clip_maps(data.maps)
        self._physics = data.physics
        self._targets = data.targets

        self._valid_starts: List[int] = []
        self._segment_ids: List[int] = []
        self._segment_starts: List[int] = []
        total_len = next(iter(data.maps.values())).shape[0]
        filter_years = (self._exclude_target_years is not None
                        or self._keep_only_target_years is not None)
        n_dropped = 0
        for seg_id, (s, e) in enumerate(self.segments):
            s = max(0, int(s))
            e = min(int(e), int(total_len))
            self._segment_starts.append(s)
            for start in range(s, e - self.window + 1):
                if filter_years and not self._window_year_ok(start, s):
                    n_dropped += 1
                    continue
                self._valid_starts.append(start)
                self._segment_ids.append(seg_id)
        if filter_years:
            kind = "exclude" if self._exclude_target_years is not None else "keep-only"
            print(f"  [YearFilter:{kind}] kept={len(self._valid_starts)} windows, dropped={n_dropped}")
        if not self._valid_starts:
            raise ValueError(f"No valid windows: total={total_len}, window={self.window}, segments={self.segments}")

    def _window_year_ok(self, start: int, segment_start: int) -> bool:
        """Decide whether a forecast window passes the E1 year filter.

        A window's target trajectory covers calendar years derived the same way
        as in __getitem__.  exclude_target_years drops a window if ANY target
        month falls in a listed year; keep_only_target_years keeps a window only
        if ANY target month falls in a listed year (used for event-only tests).
        """
        start_in_segment = start - segment_start
        target_abs_months = (self.start_month + start_in_segment
                             + self.input_len + np.arange(self.output_len))
        target_years = set(int(y) for y in (self.start_year + (target_abs_months // 12)))
        if self._exclude_target_years is not None:
            if target_years & self._exclude_target_years:
                return False
        if self._keep_only_target_years is not None:
            if not (target_years & self._keep_only_target_years):
                return False
        return True

    def _compute_stats(self, segments: Sequence[Tuple[int, int]]) -> Dict[str, float]:
        stats: Dict[str, float] = {}
        for name, arr in self.data.maps.items():
            vals = _concat_segments(arr, segments)
            stats[f"map:{name}:mean"] = float(np.nanmean(vals))
            stats[f"map:{name}:std"] = max(float(np.nanstd(vals)), 1e-6)
            print(f"  [Map stats] {name:10s} mean={stats[f'map:{name}:mean']:+.4f} std={stats[f'map:{name}:std']:.4f}")
        for name, arr in self.data.targets.items():
            vals = _concat_segments(arr, segments)
            stats[f"target:{name}:mean"] = float(np.nanmean(vals))
            stats[f"target:{name}:std"] = max(float(np.nanstd(vals)), 1e-6)
            if name == "nino34":
                stats["nino34_mean"] = stats[f"target:{name}:mean"]
                stats["nino34_std"] = stats[f"target:{name}:std"]
            if name == "wwv":
                stats["wwv_mean"] = stats[f"target:{name}:mean"]
                stats["wwv_std"] = stats[f"target:{name}:std"]
            print(f"  [Target stats] {name:10s} mean={stats[f'target:{name}:mean']:+.4f} std={stats[f'target:{name}:std']:.4f}")
        for name, arr in self.data.physics.items():
            vals = _concat_segments(arr, segments).astype(np.float64)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                center, scale = 0.0, 1.0
            elif self.phys_scaler == "robust" or name == "thermocline_feedback":
                q25, q50, q75 = np.nanpercentile(vals, [25, 50, 75])
                center = float(q50)
                scale = float((q75 - q25) / 1.349)
                if not np.isfinite(scale) or scale < 1e-6:
                    scale = max(float(np.nanstd(vals)), 1e-6)
            else:
                center = float(np.nanmean(vals))
                scale = max(float(np.nanstd(vals)), 1e-6)
            stats[f"phys:{name}:center"] = center
            stats[f"phys:{name}:scale"] = max(scale, 1e-6)
            print(f"  [Phys stats] {name:22s} center={center:+.4f} scale={max(scale, 1e-6):.4f}")
        return stats

    def _clip_maps(self, maps: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        out: Dict[str, np.ndarray] = {}
        for name, arr in maps.items():
            mu = self.stats[f"map:{name}:mean"]
            sig = self.stats[f"map:{name}:std"]
            arr = np.clip(arr, mu - self.CLIP_SIGMA * sig, mu + self.CLIP_SIGMA * sig)
            out[name] = np.nan_to_num(arr, nan=mu).astype(np.float32)
        return out

    def __len__(self) -> int:
        return len(self._valid_starts)

    def __getitem__(self, idx: int):
        start = self._valid_starts[idx]
        mid = start + self.input_len
        end = mid + self.output_len
        seg_id = self._segment_ids[idx]
        segment_start = self._segment_starts[seg_id]
        start_in_segment = start - segment_start

        init_abs_month = self.start_month + start_in_segment + self.input_len - 1
        init_month = int(init_abs_month % 12)
        input_abs_months = self.start_month + start_in_segment + np.arange(self.input_len)
        target_abs_months = self.start_month + start_in_segment + self.input_len + np.arange(self.output_len)
        target_years = self.start_year + (target_abs_months // 12)

        x_vars = []
        for name in self.map_vars:
            arr = self._maps[name]
            mu = self.stats[f"map:{name}:mean"]
            sig = self.stats[f"map:{name}:std"] + 1e-6
            x = (arr[start:mid] - mu) / sig
            if self.is_train and self.map_noise:
                noise_std = DEFAULT_NOISE_STD.get(name, 0.015)
                x = x + np.random.normal(0.0, noise_std, size=x.shape).astype(np.float32)
            x = x * self.spatial_input_mask[len(x_vars)][None, :, :]
            x_vars.append(x.astype(np.float32))
        x_map = torch.from_numpy(np.stack(x_vars, axis=1)).float()

        phys_cols = []
        for name in self.base_phys_vars:
            arr = self._physics[name]
            center = self.stats[f"phys:{name}:center"]
            scale = self.stats[f"phys:{name}:scale"] + 1e-6
            vals = np.nan_to_num(arr[start:mid], nan=center)
            vals = np.clip((vals - center) / scale, -8.0, 8.0).astype(np.float32)
            phys_cols.append(vals[:, None])
        if self.include_calendar_features:
            input_m = input_abs_months % 12
            init_m = np.full(self.input_len, init_month, dtype=np.int64)
            phys_cols += [
                np.sin(2 * np.pi * input_m / 12.0).astype(np.float32)[:, None],
                np.cos(2 * np.pi * input_m / 12.0).astype(np.float32)[:, None],
                np.sin(2 * np.pi * init_m / 12.0).astype(np.float32)[:, None],
                np.cos(2 * np.pi * init_m / 12.0).astype(np.float32)[:, None],
            ]
        if phys_cols:
            x_phys = torch.from_numpy(np.concatenate(phys_cols, axis=1)).float()
        else:
            x_phys = torch.zeros(self.input_len, 0, dtype=torch.float32)

        ys = []
        for name in self.target_vars:
            arr = self._targets[name]
            mu = self.stats[f"target:{name}:mean"]
            sig = self.stats[f"target:{name}:std"] + 1e-6
            ys.append(((arr[mid:end] - mu) / sig).astype(np.float32)[:, None])
        y = torch.from_numpy(np.concatenate(ys, axis=1)).float()
        return (
            x_map,
            x_phys,
            y,
            torch.tensor(init_month, dtype=torch.long),
            torch.from_numpy(target_years.astype(np.float32)),
            torch.tensor(seg_id, dtype=torch.long),
        )

    def get_stats(self) -> Dict[str, float]:
        return dict(self.stats)

    def get_state_scale_info(self) -> Dict[str, float]:
        info = {k.replace("map:", "").replace(":", "_"): v for k, v in self.stats.items() if k.startswith("map:")}
        info.update({"nino34_mean": self.stats.get("nino34_mean", 0.0), "nino34_std": self.stats.get("nino34_std", 1.0)})
        info.update({"wwv_mean": self.stats.get("wwv_mean", 0.0), "wwv_std": self.stats.get("wwv_std", 1.0)})
        return info

    @property
    def spatial_shape(self) -> Tuple[int, int]:
        arr = next(iter(self._maps.values()))
        return arr.shape[1], arr.shape[2]


ENSODataset = ENSOGlobalMechanismDataset
