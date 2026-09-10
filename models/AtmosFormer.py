"""Coalition-identifiable basin-variable ENSO forecaster.

The model keeps the map-based GL-Geoformer input/output contract but makes the
basin/variable mask an explicit inference-time control.  A single checkpoint
can therefore be evaluated under many coalitions without retraining.

Input
-----
``x_map``: ``(batch, history, channels, lat, lon)``
``x_phys``: optional ``(batch, history, physical_features)``

Output
------
``(prediction, diagnostics)`` with prediction shaped ``(batch, lead, target)``.
The diagnostics expose group and pair contributions for attribution checks.
When ``atmos_horizon_router`` is enabled, they also expose per-lead
``horizon_router`` weights ordered as ``state, map, group``.
When ``atmos_anomaly_calibration`` is enabled, diagnostics additionally expose
the bounded per-lead ``anomaly_gain`` and ``anomaly_offset`` corrections.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .baselines import (
    OptionalPhysEncoder,
    _arg_float,
    _arg_int,
    _choose_heads,
    _month_history_features,
)


BASIN_NAMES: Tuple[str, ...] = (
    "trop_pacific",
    "north_pacific",
    "south_pacific",
    "indian",
    "atlantic",
)

BASIN_REGIONS = {
    "trop_pacific": (-10.0, 10.0, 120.0, 280.0),
    "north_pacific": (10.0, 40.0, 120.0, 280.0),
    "south_pacific": (-40.0, -10.0, 120.0, 290.0),
    "indian": (-30.0, 30.0, 30.0, 120.0),
    "atlantic": (-30.0, 30.0, 280.0, 20.0),
}


def _region_mask(lat: np.ndarray, lon: np.ndarray, name: str) -> np.ndarray:
    lat_min, lat_max, lon_min, lon_max = BASIN_REGIONS[name]
    keep_lat = (lat >= lat_min) & (lat <= lat_max)
    lon_360 = lon % 360.0
    if lon_min <= lon_max:
        keep_lon = (lon_360 >= lon_min) & (lon_360 <= lon_max)
    else:
        keep_lon = (lon_360 >= lon_min) | (lon_360 <= lon_max)
    return keep_lat[:, None] & keep_lon[None, :]


def _build_basin_masks(
    lat: Optional[Sequence[float]], lon: Optional[Sequence[float]],
    height: int, width: int,
) -> np.ndarray:
    """Build fixed basin masks; use a regular fallback for standalone use."""
    if lat is None or lon is None:
        lat_values = np.linspace(-60.0, 60.0, int(height), dtype=np.float32)
        lon_values = np.linspace(0.0, 360.0, int(width), endpoint=False, dtype=np.float32)
    else:
        lat_values = np.asarray(lat, dtype=np.float32).reshape(-1)
        lon_values = np.asarray(lon, dtype=np.float32).reshape(-1)
        if len(lat_values) != int(height) or len(lon_values) != int(width):
            raise ValueError(
                "Coordinate lengths do not match map shape: "
                f"lat={len(lat_values)} vs {height}, lon={len(lon_values)} vs {width}"
            )
    masks = [_region_mask(lat_values, lon_values, name) for name in BASIN_NAMES]
    result = np.stack(masks, axis=0).astype(np.float32)
    if not np.any(result):
        raise ValueError("The configured latitude/longitude grid contains no basin cells")
    return result


def _parse_names(value: str) -> Iterable[str]:
    return (item.strip().lower() for item in str(value).split("+") if item.strip())


def _parse_fixed_group_mask(
    specification: str,
    variable_names: Sequence[str],
) -> np.ndarray:
    """Translate the dataset's spatial-mask syntax to a group mask."""
    n_basins = len(BASIN_NAMES)
    n_variables = len(variable_names)
    mask = np.ones((n_basins, n_variables), dtype=np.float32)
    spec = str(specification or "full").strip().lower()
    if spec in {"full", "five_basins"}:
        return mask
    if spec == "state_only":
        return np.zeros_like(mask)
    basin_index = {name: i for i, name in enumerate(BASIN_NAMES)}
    variable_index = {name.lower(): i for i, name in enumerate(variable_names)}
    if spec.startswith("basins="):
        mask[:] = 0.0
        for name in _parse_names(spec.split("=", 1)[1]):
            if name not in basin_index:
                raise ValueError(f"Unknown basin in spatial_mask_spec={specification!r}")
            mask[basin_index[name], :] = 1.0
        return mask
    if spec.startswith("without_basin="):
        name = spec.split("=", 1)[1].strip()
        if name not in basin_index:
            raise ValueError(f"Unknown basin in spatial_mask_spec={specification!r}")
        mask[basin_index[name], :] = 0.0
        return mask

    variable_mode, variable_separator, variable_value = spec.partition("=")
    if variable_separator and variable_mode in {
        "only_variable", "only_variables", "without_variable", "without_variables",
    }:
        selected = list(_parse_names(variable_value))
        unknown = [name for name in selected if name not in variable_index]
        if unknown:
            raise ValueError(
                f"Unknown variables {unknown}; loaded={list(variable_names)}"
            )
        if variable_mode.startswith("only"):
            mask[:] = 0.0
        for variable in selected:
            mask[:, variable_index[variable]] = (
                1.0 if variable_mode.startswith("only") else 0.0
            )
        return mask

    mode, separator, value = spec.partition("=")
    if not separator or mode not in {
        "only_group", "only_groups", "without_group", "without_groups",
    }:
        raise ValueError(f"Unsupported spatial_mask_spec={specification!r}")
    groups = []
    for item in _parse_names(value):
        pieces = item.split("__", 1)
        if len(pieces) != 2 or pieces[0] not in basin_index or pieces[1] not in variable_index:
            raise ValueError(f"Invalid basin-variable group {item!r}")
        groups.append((basin_index[pieces[0]], variable_index[pieces[1]]))
    if mode.startswith("only"):
        mask[:] = 0.0
        for basin, variable in groups:
            mask[basin, variable] = 1.0
    else:
        for basin, variable in groups:
            mask[basin, variable] = 0.0
    return mask


class _AtmosTemporalEncoder(nn.Module):
    def __init__(self, d_model: int, heads: int, depth: int, ffn_mult: int, dropout: float):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=d_model * ffn_mult,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=max(1, depth))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.encoder(x))


class _AtmosMaskedGroupEncoder(nn.Module):
    """Group self-attention whose inactive keys are padding, not zero tokens."""

    def __init__(self, d_model: int, heads: int, depth: int, ffn_mult: int, dropout: float):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=d_model * ffn_mult,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=max(1, depth))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, keep: torch.Tensor, null_token: torch.Tensor) -> torch.Tensor:
        x = torch.where(keep[..., None], x, null_token.view(1, 1, -1))
        padding = ~keep
        all_masked = padding.all(dim=1)
        if all_masked.any():
            padding = padding.clone()
            padding[all_masked, 0] = False
        x = self.encoder(x, src_key_padding_mask=padding)
        return self.norm(x) * keep[..., None]


class AtmosFormer(nn.Module):
    """Coalition-identifiable map-based ENSO predictor.

    ``set_group_mask`` changes only the inference coalition.  Training uses
    structured coalition dropout so all coalitions are represented by one set
    of weights.  No forecast field is recursively fed back into the model.
    """

    def __init__(self, args):
        super().__init__()
        self.input_len = _arg_int(args, "input_len", 12)
        self.output_len = _arg_int(args, "output_len", 24)
        self.target_dim = _arg_int(args, "target_dim", 2)
        self.input_dim = _arg_int(args, "n_vars", _arg_int(args, "input_dim", 1))
        self.phys_dim = _arg_int(args, "phys_dim", len(getattr(args, "phys_feature_names", [])))
        self.phase_phys_dim = min(
            self.phys_dim,
            max(0, _arg_int(args, "base_phys_dim", self.phys_dim)),
        )
        self.d_model = _arg_int(args, "atmos_d_model", 128)
        self.dropout_p = _arg_float(args, "atmos_dropout", _arg_float(args, "dropout", 0.10))
        self.pool_lat = max(2, _arg_int(args, "atmos_pool_lat", 10))
        self.pool_lon = max(2, _arg_int(args, "atmos_pool_lon", 20))
        self.n_cells = self.pool_lat * self.pool_lon
        self.n_basins = len(BASIN_NAMES)
        self.variable_names = tuple(
            str(x) for x in getattr(args, "map_vars", getattr(args, "map_vars_csv", []))
        )
        if len(self.variable_names) != self.input_dim:
            self.variable_names = tuple(f"var_{i}" for i in range(self.input_dim))

        requested_heads = _arg_int(args, "atmos_n_heads", 4)
        heads = _choose_heads(self.d_model, requested_heads)
        ffn_mult = max(2, _arg_int(args, "atmos_ffn_mult", 3))
        temporal_depth = max(1, _arg_int(args, "atmos_temporal_depth", 2))
        spatial_depth = max(1, _arg_int(args, "atmos_spatial_depth", 2))
        group_depth = max(1, _arg_int(args, "atmos_group_depth", 2))
        decoder_depth = max(1, _arg_int(args, "atmos_decoder_depth", 2))

        height = _arg_int(args, "img_height", 81)
        width = _arg_int(args, "img_width", 180)
        basin_masks = _build_basin_masks(
            getattr(args, "lat_coords", None),
            getattr(args, "lon_coords", None),
            height,
            width,
        )
        self.register_buffer("basin_masks", torch.from_numpy(basin_masks), persistent=True)
        fixed_mask = _parse_fixed_group_mask(
            getattr(args, "spatial_mask_spec", "full"), self.variable_names
        )
        self.register_buffer("fixed_group_mask", torch.from_numpy(fixed_mask), persistent=True)
        self._manual_group_mask: Optional[torch.Tensor] = None

        self.map_projection = nn.Sequential(
            nn.Linear(self.input_dim, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.GELU(),
        )
        self.time_embedding = nn.Embedding(max(self.input_len, 1), self.d_model)
        self.lat_embedding = nn.Embedding(self.pool_lat, self.d_model)
        self.lon_embedding = nn.Embedding(self.pool_lon, self.d_model)
        self.history_month_projection = nn.Sequential(
            nn.Linear(2, self.d_model), nn.GELU(), nn.LayerNorm(self.d_model)
        )
        self.map_temporal = _AtmosTemporalEncoder(
            self.d_model, heads, temporal_depth, ffn_mult, self.dropout_p
        )
        self.map_spatial = _AtmosTemporalEncoder(
            self.d_model, heads, spatial_depth, ffn_mult, self.dropout_p
        )
        self.map_null_token = nn.Parameter(torch.zeros(self.d_model))

        self.group_feature_projection = nn.Sequential(
            nn.Linear(3, self.d_model), nn.LayerNorm(self.d_model), nn.GELU()
        )
        self.basin_embedding = nn.Embedding(self.n_basins, self.d_model)
        self.variable_embedding = nn.Embedding(self.input_dim, self.d_model)
        self.group_temporal = _AtmosTemporalEncoder(
            self.d_model, heads, temporal_depth, ffn_mult, self.dropout_p
        )
        self.group_temporal_pool = nn.Linear(self.d_model, 1)
        self.group_encoder = _AtmosMaskedGroupEncoder(
            self.d_model, heads, group_depth, ffn_mult, self.dropout_p
        )
        self.group_null_token = nn.Parameter(torch.zeros(self.d_model))

        self.use_horizon_router = bool(getattr(args, "atmos_horizon_router", False))
        self.phase_summary_scale = _arg_float(args, "atmos_phase_summary_scale", 0.25)
        self.router_prior_strength = _arg_float(args, "atmos_router_prior_strength", 1.0)
        self.use_multiscale_phase = bool(getattr(args, "atmos_multiscale_phase", False))
        self.use_phase_film = bool(getattr(args, "atmos_phase_film", False))
        self.phase_film_scale = max(
            _arg_float(args, "atmos_phase_film_scale", 0.20), 0.0
        )
        self.phase_film_lead_power = max(
            _arg_float(args, "atmos_phase_film_lead_power", 1.0), 0.0
        )
        self.target_calendar_decay = max(
            _arg_float(args, "atmos_target_calendar_decay", 0.0), 0.0
        )
        if self.use_horizon_router:
            if self.phase_phys_dim > 0:
                phase_windows = (3, 6, self.input_len) if self.use_multiscale_phase else (self.input_len,)
                self.phase_window_sizes = tuple(
                    sorted({max(1, min(int(window), self.input_len)) for window in phase_windows})
                )
                self.phase_summary_projection = nn.Sequential(
                    nn.Linear(
                        self.phase_phys_dim * 4 * len(self.phase_window_sizes),
                        self.d_model,
                    ),
                    nn.LayerNorm(self.d_model),
                    nn.GELU(),
                )
            else:
                self.phase_window_sizes = (self.input_len,)
                self.phase_summary_projection = None
            if self.use_phase_film:
                self.phase_film = nn.Sequential(
                    nn.LayerNorm(self.d_model),
                    nn.Linear(self.d_model, self.d_model * 2),
                )
                nn.init.zeros_(self.phase_film[-1].weight)
                nn.init.zeros_(self.phase_film[-1].bias)
            else:
                self.phase_film = None
            router_hidden = max(32, self.d_model // 2)
            self.horizon_router = nn.Sequential(
                nn.LayerNorm(self.d_model * 4),
                nn.Linear(self.d_model * 4, router_hidden),
                nn.GELU(),
                nn.Linear(router_hidden, 3),
            )
            leads = torch.arange(self.output_len, dtype=torch.float32)
            fraction = leads / max(float(self.output_len - 1), 1.0)
            prior = torch.stack(
                [
                    1.50 * (1.0 - fraction),
                    0.15 + 0.90 * fraction,
                    0.30 + 0.75 * fraction,
                ],
                dim=-1,
            )
            self.register_buffer("horizon_router_prior", prior, persistent=True)
            nn.init.zeros_(self.horizon_router[-1].weight)
            nn.init.zeros_(self.horizon_router[-1].bias)
        else:
            self.phase_window_sizes = (self.input_len,)
            self.phase_summary_projection = None
            self.phase_film = None
            self.horizon_router = None
            self.horizon_router_prior = None

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.d_model,
            nhead=heads,
            dim_feedforward=self.d_model * ffn_mult,
            dropout=self.dropout_p,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.lead_decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_depth)

        self.phys_encoder = OptionalPhysEncoder(self.phys_dim, self.d_model, self.dropout_p)
        self.lead_embedding = nn.Embedding(max(self.output_len, 1), self.d_model)
        self.future_month_projection = nn.Sequential(
            nn.Linear(4, self.d_model), nn.GELU(), nn.LayerNorm(self.d_model)
        )
        self.spb_embedding = nn.Embedding(2, self.d_model)
        self.query_norm = nn.LayerNorm(self.d_model)

        context_dim = self.d_model * (2 if self.use_horizon_router else 3)
        self.baseline_head = nn.Sequential(
            nn.LayerNorm(context_dim),
            nn.Linear(context_dim, self.d_model),
            nn.GELU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(self.d_model, self.target_dim),
        )
        self.use_anomaly_calibration = bool(
            getattr(args, "atmos_anomaly_calibration", False)
        )
        self.anomaly_gain_scale = max(
            _arg_float(args, "atmos_anomaly_gain_scale", 0.40), 0.0
        )
        self.anomaly_offset_scale = max(
            _arg_float(args, "atmos_anomaly_offset_scale", 0.25), 0.0
        )
        if self.use_anomaly_calibration:
            calibration_dim = context_dim + self.d_model
            calibration_hidden = max(32, self.d_model // 2)
            self.anomaly_calibration_head = nn.Sequential(
                nn.LayerNorm(calibration_dim),
                nn.Linear(calibration_dim, calibration_hidden),
                nn.GELU(),
                nn.Dropout(self.dropout_p),
                nn.Linear(calibration_hidden, self.target_dim * 2),
            )
            nn.init.zeros_(self.anomaly_calibration_head[-1].weight)
            nn.init.zeros_(self.anomaly_calibration_head[-1].bias)
        else:
            self.anomaly_calibration_head = None
        self.use_regime_moe = bool(getattr(args, "atmos_regime_moe", False))
        self.spring_scale = _arg_float(args, "atmos_spring_scale", 0.20)
        if self.use_regime_moe:
            self.spring_head = nn.Sequential(
                nn.LayerNorm(context_dim),
                nn.Linear(context_dim, self.d_model),
                nn.GELU(),
                nn.Dropout(self.dropout_p),
                nn.Linear(self.d_model, self.target_dim),
            )
            self.spring_gate = nn.Sequential(
                nn.Linear(self.d_model, max(32, self.d_model // 2)),
                nn.GELU(),
                nn.Linear(max(32, self.d_model // 2), 1),
            )
        else:
            self.spring_head = None
            self.spring_gate = None
        self.uncertainty_head = nn.Sequential(
            nn.LayerNorm(context_dim),
            nn.Linear(context_dim, max(32, self.d_model // 2)),
            nn.GELU(),
            nn.Linear(max(32, self.d_model // 2), self.target_dim),
        )

        joint_dim = self.d_model
        self.group_query = nn.Linear(self.d_model, joint_dim, bias=False)
        self.group_key = nn.Linear(self.d_model, joint_dim, bias=False)
        self.group_joint = nn.Sequential(
            nn.Linear(joint_dim, self.d_model), nn.Tanh()
        )
        self.group_gate = nn.Linear(self.d_model, 1)
        self.group_value = nn.Sequential(
            nn.Linear(self.d_model, self.d_model), nn.GELU(), nn.Linear(self.d_model, self.target_dim)
        )
        self.group_scale = nn.Parameter(torch.tensor(0.50))

        pair_names = (
            ("trop_pacific", "indian"),
            ("trop_pacific", "atlantic"),
            ("indian", "atlantic"),
            ("trop_pacific", "north_pacific"),
            ("trop_pacific", "south_pacific"),
        )
        self.pair_indices = tuple(
            (BASIN_NAMES.index(left), BASIN_NAMES.index(right))
            for left, right in pair_names
        )
        self.use_pairwise = not bool(getattr(args, "atmos_no_pairwise", False))
        if self.use_pairwise:
            self.pair_query = nn.Linear(self.d_model, self.d_model, bias=False)
            self.pair_key = nn.Linear(self.d_model, self.d_model, bias=False)
            self.pair_joint = nn.Sequential(nn.Linear(self.d_model, self.d_model), nn.Tanh())
            self.pair_gate = nn.Linear(self.d_model, 1)
            self.pair_value = nn.Sequential(
                nn.Linear(self.d_model, self.d_model), nn.GELU(), nn.Linear(self.d_model, self.target_dim)
            )
            self.pair_scale = nn.Parameter(torch.tensor(0.05))
        else:
            self.pair_scale = None

        self.coalition_probability = _arg_float(args, "atmos_coalition_probability", 0.70)
        self.group_drop_probability = _arg_float(args, "atmos_group_drop", 0.10)
        self.basin_drop_probability = _arg_float(args, "atmos_basin_drop", 0.05)
        self.variable_drop_probability = _arg_float(args, "atmos_variable_drop", 0.05)
        self.mask_strategy = str(getattr(args, "atmos_mask_strategy", "bernoulli")).lower()
        self.full_mask_probability = _arg_float(args, "atmos_full_mask_probability", 0.50)
        self.leave_basin_probability = _arg_float(args, "atmos_leave_basin_probability", 0.20)
        self.leave_variable_probability = _arg_float(args, "atmos_leave_variable_probability", 0.15)
        self.loss_diagnostics: Dict[str, torch.Tensor] = {}

        nn.init.constant_(self.group_gate.bias, -1.0)
        if self.use_pairwise:
            nn.init.constant_(self.pair_gate.bias, -2.0)
        if self.use_regime_moe:
            nn.init.constant_(self.spring_gate[-1].bias, -1.5)

    @property
    def n_groups(self) -> int:
        return self.n_basins * self.input_dim

    def set_group_mask(self, group_mask: torch.Tensor) -> None:
        """Set a fixed inference coalition shaped ``(5, n_variables)``."""
        mask = torch.as_tensor(group_mask, dtype=torch.float32)
        if mask.shape != (self.n_basins, self.input_dim):
            raise ValueError(
                f"Expected group mask {(self.n_basins, self.input_dim)}, got {tuple(mask.shape)}"
            )
        self._manual_group_mask = mask.detach().clone()

    def clear_group_mask(self) -> None:
        self._manual_group_mask = None

    def full_group_mask(self, device: Optional[torch.device] = None) -> torch.Tensor:
        return self.fixed_group_mask.new_ones(self.n_basins, self.input_dim, device=device)

    def make_group_mask(
        self,
        basins: Optional[Iterable[str]] = None,
        variables: Optional[Iterable[str]] = None,
    ) -> torch.Tensor:
        """Create a named coalition mask for post-training attribution."""
        mask = torch.zeros(self.n_basins, self.input_dim, dtype=torch.float32)
        if basins is None:
            selected_basins = set(BASIN_NAMES)
        elif isinstance(basins, str):
            selected_basins = {str(x).strip().lower() for x in basins.split(",") if str(x).strip()}
        else:
            selected_basins = {str(x).lower() for x in basins}
        if variables is None:
            selected_variables = {str(x).lower() for x in self.variable_names}
        elif isinstance(variables, str):
            selected_variables = {str(x).strip().lower() for x in variables.split(",") if str(x).strip()}
        else:
            selected_variables = {str(x).lower() for x in variables}
        for basin_index, basin in enumerate(BASIN_NAMES):
            for variable_index, variable in enumerate(self.variable_names):
                if basin in selected_basins and variable.lower() in selected_variables:
                    mask[basin_index, variable_index] = 1.0
        return mask

    def sample_coalition_mask(self, batch_size: int, device: torch.device) -> torch.Tensor:
        base = self.fixed_group_mask.to(device=device).expand(batch_size, -1, -1)
        if self.coalition_probability <= 0.0:
            return base.clone()

        if self.mask_strategy in {"balanced", "structured"}:
            mask = base.bool().clone()
            draw = torch.rand(batch_size, device=device)
            full_p = min(max(self.full_mask_probability, 0.0), 1.0)
            basin_p = min(max(self.leave_basin_probability, 0.0), 1.0 - full_p)
            variable_p = min(
                max(self.leave_variable_probability, 0.0),
                1.0 - full_p - basin_p,
            )
            leave_basin = (draw >= full_p) & (draw < full_p + basin_p)
            leave_variable = (draw >= full_p + basin_p) & (
                draw < full_p + basin_p + variable_p
            )
            random_rows = draw >= full_p + basin_p + variable_p

            if leave_basin.any():
                selected = torch.randint(self.n_basins, (batch_size,), device=device)
                for row in torch.nonzero(leave_basin, as_tuple=False).flatten().tolist():
                    basin = int(selected[row].item())
                    active_basins = mask[row].any(dim=1)
                    if bool(active_basins.sum() > 1) and bool(active_basins[basin]):
                        mask[row, basin] = False

            if leave_variable.any():
                selected = torch.randint(self.input_dim, (batch_size,), device=device)
                for row in torch.nonzero(leave_variable, as_tuple=False).flatten().tolist():
                    variable = int(selected[row].item())
                    active_variables = mask[row].any(dim=0)
                    if bool(active_variables.sum() > 1) and bool(active_variables[variable]):
                        mask[row, :, variable] = False

            if random_rows.any():
                random_keep = torch.rand_like(base) >= self.group_drop_probability
                basin_keep = torch.ones(
                    batch_size, self.n_basins, 1, device=device, dtype=torch.bool
                )
                if self.n_basins > 1:
                    basin_keep[:, 1:] = (
                        torch.rand(
                            batch_size, self.n_basins - 1, 1, device=device
                        )
                        >= self.basin_drop_probability
                    )
                variable_keep = (
                    torch.rand(batch_size, 1, self.input_dim, device=device)
                    >= self.variable_drop_probability
                )
                proposed = base.bool() & random_keep & basin_keep & variable_keep
                mask[random_rows] = proposed[random_rows]

            nonempty_base = base.sum(dim=(1, 2)) > 0
            empty = (mask.sum(dim=(1, 2)) == 0) & nonempty_base
            if empty.any():
                first_active = base[empty].reshape(int(empty.sum()), -1).argmax(dim=1)
                flat = mask[empty].reshape(int(empty.sum()), -1).clone()
                flat.scatter_(1, first_active[:, None], True)
                mask[empty] = flat.reshape(-1, self.n_basins, self.input_dim)
            return mask.float()

        apply = torch.rand(batch_size, 1, 1, device=device) < self.coalition_probability
        group_keep = torch.rand_like(base) >= self.group_drop_probability
        basin_keep = torch.ones(batch_size, self.n_basins, 1, device=device, dtype=torch.bool)
        if self.n_basins > 1:
            basin_keep[:, 1:] = (
                torch.rand(batch_size, self.n_basins - 1, 1, device=device)
                >= self.basin_drop_probability
            )
        variable_keep = (
            torch.rand(batch_size, 1, self.input_dim, device=device)
            >= self.variable_drop_probability
        )
        proposed = base.bool() & group_keep & basin_keep & variable_keep
        mask = torch.where(apply, proposed, base.bool()).float()
        nonempty_base = base.sum(dim=(1, 2)) > 0
        empty = (mask.sum(dim=(1, 2)) == 0) & nonempty_base
        if empty.any():
            first_active = base[empty].reshape(int(empty.sum()), -1).argmax(dim=1)
            flat = mask[empty].reshape(int(empty.sum()), -1).clone()
            flat.scatter_(1, first_active[:, None], 1.0)
            mask[empty] = flat.reshape(-1, self.n_basins, self.input_dim)
        return mask

    def _effective_group_mask(self, batch: int, device: torch.device) -> torch.Tensor:
        if self._manual_group_mask is not None:
            base = self._manual_group_mask.to(device=device).expand(batch, -1, -1)
        else:
            base = self.fixed_group_mask.to(device=device).expand(batch, -1, -1)
        if self.training and self._manual_group_mask is None:
            return self.sample_coalition_mask(batch, device)
        return base.clone()

    def _map_channel_mask(self, group_mask: torch.Tensor, height: int, width: int) -> torch.Tensor:
        masks = self.basin_masks.to(device=group_mask.device, dtype=group_mask.dtype)
        if masks.shape[-2:] != (height, width):
            raise ValueError(
                f"Input map shape {(height, width)} does not match trained coordinates "
                f"{tuple(masks.shape[-2:])}"
            )
        result = group_mask.new_zeros(group_mask.shape[0], self.input_dim, height, width)
        all_basins = group_mask.sum(dim=1) >= float(self.n_basins) - 1.0e-4
        for variable in range(self.input_dim):
            selected = group_mask[:, :, variable]
            regional = torch.einsum("bs,shw->bhw", selected, masks).clamp(0.0, 1.0)
            regional = torch.where(all_basins[:, variable, None, None], torch.ones_like(regional), regional)
            result[:, variable] = regional
        return result

    def _encode_map(self, x_map: torch.Tensor, init_month: torch.Tensor, group_mask: torch.Tensor):
        batch, history, channels, height, width = x_map.shape
        channel_mask = self._map_channel_mask(group_mask, height, width)
        masked_map = x_map * channel_mask[:, None]
        pooled = F.adaptive_avg_pool2d(
            masked_map.reshape(batch * history, channels, height, width),
            (self.pool_lat, self.pool_lon),
        ).reshape(batch, history, channels, self.pool_lat, self.pool_lon)
        series = pooled.permute(0, 3, 4, 1, 2).reshape(
            batch * self.n_cells, history, channels
        )
        tokens = self.map_projection(series)
        time_ids = torch.arange(history, device=x_map.device).clamp(max=self.input_len - 1)
        lat_ids = torch.arange(self.pool_lat, device=x_map.device)[:, None].expand(self.pool_lat, self.pool_lon).reshape(-1)
        lon_ids = torch.arange(self.pool_lon, device=x_map.device)[None, :].expand(self.pool_lat, self.pool_lon).reshape(-1)
        spatial = self.lat_embedding(lat_ids) + self.lon_embedding(lon_ids)
        month_features = self.history_month_projection(_month_history_features(init_month, history))
        tokens = tokens + self.time_embedding(time_ids)[None] + spatial[None, :, None, :].expand(batch, -1, history, -1).reshape(batch * self.n_cells, history, self.d_model)
        tokens = tokens + month_features[:, None].expand(batch, self.n_cells, history, self.d_model).reshape(batch * self.n_cells, history, self.d_model)
        local = self.map_temporal(tokens)
        local = local.mean(dim=1).reshape(batch, self.n_cells, self.d_model)
        memory = self.map_spatial(local)
        pooled_mask = F.adaptive_avg_pool2d(
            channel_mask, (self.pool_lat, self.pool_lon)
        ).amax(dim=1).reshape(batch, self.n_cells) > 0
        if (~pooled_mask).any():
            memory = memory.clone()
            memory[~pooled_mask] = self.map_null_token.view(1, 1, -1)
        return memory, channel_mask, pooled_mask

    def _encode_groups(self, x_map: torch.Tensor, init_month: torch.Tensor, group_mask: torch.Tensor):
        batch, history, channels, height, width = x_map.shape
        basin_masks = self.basin_masks.to(device=x_map.device, dtype=x_map.dtype)
        area = basin_masks.sum(dim=(-1, -2)).clamp_min(1.0)
        regional = torch.einsum("btchw,shw->btsc", x_map, basin_masks) / area[None, None, :, None]
        delta = regional[:, -1:] - regional[:, :1]
        delta = delta.expand(-1, history, -1, -1)
        features = torch.stack([regional, delta, regional - regional.mean(dim=1, keepdim=True)], dim=-1)
        features = features.permute(0, 2, 3, 1, 4).reshape(batch * self.n_groups, history, 3)
        tokens = self.group_feature_projection(features)
        time_ids = torch.arange(history, device=x_map.device).clamp(max=self.input_len - 1)
        basin_ids = torch.arange(self.n_basins, device=x_map.device)
        variable_ids = torch.arange(self.input_dim, device=x_map.device)
        tokens = tokens + self.time_embedding(time_ids)[None]
        tokens = tokens + self.basin_embedding(basin_ids)[None, :, None, :].expand(batch, self.n_basins, self.input_dim, self.d_model).reshape(batch * self.n_groups, 1, self.d_model)
        tokens = tokens + self.variable_embedding(variable_ids)[None, None, :, :].expand(batch, self.n_basins, self.input_dim, self.d_model).reshape(batch * self.n_groups, 1, self.d_model)
        months = _month_history_features(init_month, history)
        tokens = tokens + self.history_month_projection(months)[:, None].expand(batch, self.n_groups, history, self.d_model).reshape(batch * self.n_groups, history, self.d_model)
        temporal = self.group_temporal(tokens)
        weights = torch.softmax(self.group_temporal_pool(temporal), dim=1)
        pooled = (weights * temporal).sum(dim=1).reshape(batch, self.n_basins, self.input_dim, self.d_model)
        keep = group_mask.bool()
        pooled = self.group_encoder(pooled.reshape(batch, self.n_groups, self.d_model), keep.reshape(batch, self.n_groups), self.group_null_token)
        return pooled.reshape(batch, self.n_basins, self.input_dim, self.d_model), weights.reshape(batch, self.n_groups, history)

    def _future_features(self, init_month: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = init_month.shape[0]
        leads = torch.arange(1, self.output_len + 1, device=init_month.device).view(1, -1)
        target_months = (init_month.view(batch, 1).long() + leads) % 12
        init_values = init_month.view(batch, 1).float().expand(batch, self.output_len)
        target_values = target_months.float()
        features = torch.stack(
            [
                torch.sin(2 * torch.pi * target_values / 12.0),
                torch.cos(2 * torch.pi * target_values / 12.0),
                torch.sin(2 * torch.pi * init_values / 12.0),
                torch.cos(2 * torch.pi * init_values / 12.0),
            ], dim=-1,
        )
        spb = ((target_months == 2) | (target_months == 3) | (target_months == 4)).long()
        return target_months, features, spb

    def _phase_summary(
        self,
        x_phys: Optional[torch.Tensor],
        batch: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Return normalized mean/trend/amplitude conditions for the lead router."""
        if (
            not self.use_horizon_router
            or self.phase_summary_projection is None
            or self.phase_phys_dim <= 0
            or x_phys is None
            or x_phys.numel() == 0
        ):
            return torch.zeros(batch, self.d_model, device=device, dtype=dtype)
        values = x_phys.to(device=device, dtype=dtype)
        if values.shape[-1] > self.phase_phys_dim:
            values = values[..., : self.phase_phys_dim]
        elif values.shape[-1] < self.phase_phys_dim:
            pad = torch.zeros(
                *values.shape[:-1], self.phase_phys_dim - values.shape[-1],
                device=device, dtype=dtype,
            )
            values = torch.cat([values, pad], dim=-1)
        summaries = []
        for window in self.phase_window_sizes:
            segment = values[:, -window:]
            mean = segment.mean(dim=1)
            last = segment[:, -1]
            trend = last - segment[:, 0]
            spread = segment.std(dim=1, unbiased=False)
            summaries.extend((mean, last, trend, spread))
        summary = torch.cat(summaries, dim=-1)
        return self.phase_summary_projection(summary)

    def _pair_states(self, group_states: torch.Tensor, group_mask: torch.Tensor):
        basin_states = []
        basin_keep = []
        for basin in range(self.n_basins):
            keep = group_mask[:, basin].sum(dim=1) > 0
            denom = group_mask[:, basin].sum(dim=1, keepdim=True).clamp_min(1.0)
            state = (group_states[:, basin] * group_mask[:, basin, :, None]).sum(dim=1) / denom
            basin_states.append(state)
            basin_keep.append(keep)
        basin_states = torch.stack(basin_states, dim=1)
        basin_keep = torch.stack(basin_keep, dim=1)
        left = torch.stack([basin_states[:, a] for a, _ in self.pair_indices], dim=1)
        right = torch.stack([basin_states[:, b] for _, b in self.pair_indices], dim=1)
        keep = torch.stack([basin_keep[:, a] & basin_keep[:, b] for a, b in self.pair_indices], dim=1)
        return left, right, keep

    def forward(
        self,
        x_map: torch.Tensor,
        x_phys: Optional[torch.Tensor] = None,
        init_month: Optional[torch.Tensor] = None,
        target_years: Optional[torch.Tensor] = None,
        return_diagnostics: bool = False,
    ):
        del target_years
        batch = x_map.shape[0]
        device, dtype = x_map.device, x_map.dtype
        if init_month is None:
            init_month = torch.zeros(batch, device=device, dtype=torch.long)
        init_month = init_month.to(device=device).long()
        group_mask = self._effective_group_mask(batch, device).to(dtype=dtype)

        map_memory, channel_mask, map_keep = self._encode_map(x_map, init_month, group_mask)
        group_states, temporal_weights = self._encode_groups(x_map, init_month, group_mask)
        flat_groups = group_states.reshape(batch, self.n_groups, self.d_model)
        aux_state = self.phys_encoder(x_phys, batch, device, dtype)
        phase_state = self._phase_summary(x_phys, batch, device, dtype)
        target_months, month_features, spb = self._future_features(init_month)
        if self.target_calendar_decay > 0.0 and self.output_len > 1:
            lead_fraction = torch.arange(
                self.output_len, device=device, dtype=dtype
            ) / float(self.output_len - 1)
            target_scale = torch.exp(-self.target_calendar_decay * lead_fraction)
            month_features = month_features.clone()
            month_features[..., :2] = month_features[..., :2] * target_scale[None, :, None]
        else:
            target_scale = torch.ones(self.output_len, device=device, dtype=dtype)
        lead_ids = torch.arange(self.output_len, device=device)[None].expand(batch, -1)
        query = self.lead_embedding(lead_ids) + self.future_month_projection(month_features)
        query = query + aux_state[:, None, :]
        if self.use_horizon_router:
            query = query + float(self.phase_summary_scale) * phase_state[:, None, :]
        query = self.query_norm(query + self.spb_embedding(spb))
        if self.phase_film is not None and self.phase_film_scale > 0.0:
            film = self.phase_film(phase_state).view(batch, 2, self.d_model)
            lead_fraction = torch.arange(
                self.output_len, device=device, dtype=dtype
            ) / float(max(self.output_len - 1, 1))
            lead_gain = lead_fraction.pow(self.phase_film_lead_power).view(1, -1, 1)
            gamma = torch.tanh(film[:, 0]).unsqueeze(1) * lead_gain
            beta = torch.tanh(film[:, 1]).unsqueeze(1) * lead_gain
            query = query * (1.0 + float(self.phase_film_scale) * gamma)
            query = query + float(self.phase_film_scale) * beta
        else:
            gamma = query.new_zeros(batch, self.output_len, self.d_model)
            beta = query.new_zeros(batch, self.output_len, self.d_model)
        if self.use_horizon_router:
            map_keep_f = map_keep.to(dtype=dtype)
            map_context = (
                (map_memory * map_keep_f[..., None]).sum(dim=1)
                / map_keep_f.sum(dim=1, keepdim=True).clamp_min(1.0)
            )
            group_keep_f = group_mask.reshape(batch, self.n_groups).to(dtype=dtype)
            group_context = (
                (flat_groups * group_keep_f[..., None]).sum(dim=1)
                / group_keep_f.sum(dim=1, keepdim=True).clamp_min(1.0)
            )
        else:
            map_context = map_memory.mean(dim=1)
            group_context = x_map.new_zeros(batch, self.d_model)
        memory = torch.cat([map_memory, flat_groups], dim=1)
        memory_keep = torch.cat([map_keep, group_mask.reshape(batch, self.n_groups).bool()], dim=1)
        memory_padding = ~memory_keep
        all_masked = memory_padding.all(dim=1)
        if all_masked.any():
            memory_padding = memory_padding.clone()
            memory_padding[all_masked, 0] = False
        decoded = self.lead_decoder(
            query, memory, memory_key_padding_mask=memory_padding
        )
        if self.use_horizon_router:
            route_features = torch.cat(
                [
                    query,
                    aux_state[:, None, :].expand(-1, self.output_len, -1),
                    map_context[:, None, :].expand(-1, self.output_len, -1),
                    group_context[:, None, :].expand(-1, self.output_len, -1),
                ],
                dim=-1,
            )
            route_logits = self.horizon_router(route_features)
            prior = self.horizon_router_prior[: self.output_len].to(
                device=device, dtype=dtype
            )
            route_logits = route_logits + float(self.router_prior_strength) * prior[None]
            horizon_router = torch.softmax(route_logits, dim=-1)
            route_states = torch.stack(
                [
                    aux_state[:, None, :].expand(-1, self.output_len, -1),
                    map_context[:, None, :].expand(-1, self.output_len, -1),
                    group_context[:, None, :].expand(-1, self.output_len, -1),
                ],
                dim=2,
            )
            routed_context = (horizon_router[..., None] * route_states).sum(dim=2)
            context = torch.cat([decoded, routed_context], dim=-1)
        else:
            horizon_router = x_map.new_zeros(batch, self.output_len, 3)
            context = torch.cat(
                [
                    decoded,
                    aux_state[:, None, :].expand(-1, self.output_len, -1),
                    map_context[:, None, :].expand(-1, self.output_len, -1),
                ],
                dim=-1,
            )
        baseline = self.baseline_head(context)
        if self.use_regime_moe:
            spring_gate = torch.sigmoid(self.spring_gate(query)).squeeze(-1)
            spring_gate = spring_gate * spb.to(dtype=query.dtype)
            baseline = baseline + (
                float(self.spring_scale)
                * spring_gate[..., None]
                * self.spring_head(context)
            )
        else:
            spring_gate = query.new_zeros(batch, self.output_len)

        group_query = self.group_query(query)[:, :, None, :]
        group_key = self.group_key(flat_groups)[:, None, :, :]
        group_joint = self.group_joint(group_query + group_key)
        group_gate = torch.sigmoid(self.group_gate(group_joint)).squeeze(-1) * group_mask.reshape(batch, 1, self.n_groups)
        group_contributions = self.group_scale * group_gate[..., None] * self.group_value(group_joint)

        if self.use_pairwise:
            left, right, pair_keep = self._pair_states(group_states, group_mask)
            pair_state = torch.tanh(self.pair_query(left)) * torch.tanh(self.pair_key(right))
            pair_joint = self.pair_joint(pair_state[:, None] + self.pair_query(query)[:, :, None, :])
            pair_gate = torch.sigmoid(self.pair_gate(pair_joint)).squeeze(-1) * pair_keep[:, None]
            pair_contributions = self.pair_scale * pair_gate[..., None] * self.pair_value(pair_joint)
        else:
            pair_keep = x_map.new_zeros(batch, 0)
            pair_gate = x_map.new_zeros(batch, self.output_len, 0)
            pair_contributions = x_map.new_zeros(batch, self.output_len, 0, self.target_dim)

        mean = baseline + group_contributions.sum(dim=2) + pair_contributions.sum(dim=2)
        if self.anomaly_calibration_head is not None:
            calibration_context = torch.cat(
                [context, phase_state[:, None, :].expand(-1, self.output_len, -1)],
                dim=-1,
            )
            calibration = self.anomaly_calibration_head(calibration_context)
            gain_raw, offset_raw = torch.chunk(calibration, 2, dim=-1)
            anomaly_gain = torch.exp(
                float(self.anomaly_gain_scale) * torch.tanh(gain_raw)
            )
            anomaly_offset = float(self.anomaly_offset_scale) * torch.tanh(offset_raw)
            mean = anomaly_gain * mean + anomaly_offset
        else:
            anomaly_gain = torch.ones_like(mean)
            anomaly_offset = torch.zeros_like(mean)
        logvar = self.uncertainty_head(context).clamp(min=-6.0, max=4.0)
        self.loss_diagnostics = {
            "final_logvar": logvar,
            "group_mask": group_mask,
            "channel_mask": channel_mask,
            "spring_gate": spring_gate,
            "horizon_router": horizon_router,
            "target_months": target_months,
            "phase_film_gamma": gamma,
            "phase_film_beta": beta,
            "target_calendar_scale": target_scale,
            "anomaly_gain": anomaly_gain,
            "anomaly_offset": anomaly_offset,
        }
        if return_diagnostics:
            return mean, {
                "final_logvar": logvar.detach(),
                "group_mask": group_mask.detach(),
                "group_contributions": group_contributions.detach(),
                "group_gates": group_gate.detach(),
                "pair_contributions": pair_contributions.detach(),
                "pair_gates": pair_gate.detach(),
                "spring_gate": spring_gate.detach(),
                "horizon_router": horizon_router.detach(),
                "phase_state": phase_state.detach(),
                "phase_film_gamma": gamma.detach(),
                "phase_film_beta": beta.detach(),
                "target_calendar_scale": target_scale.detach(),
                "anomaly_gain": anomaly_gain.detach(),
                "anomaly_offset": anomaly_offset.detach(),
                "target_months": target_months.detach(),
                "spb_flags": spb.detach(),
                "temporal_weights": temporal_weights.detach(),
            }
        return mean, None


