"""Neural baselines for ENSO hindcast experiments.

The classes in this file follow the ``train_v2.py`` model contract:

    forward(x_map, x_phys=None, init_month=None, target_years=None,
            return_diagnostics=False) -> (pred, diagnostics)

where ``x_map`` is ``(B, T, C, H, W)`` and ``pred`` is
``(B, output_len, target_dim)``.  They intentionally do not emit CTM/PINN
diagnostics, so mechanism-specific losses stay inactive for these baselines.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .forecast_decoder import ProgressiveLeadDecoder


def _arg_int(args, name: str, default: int) -> int:
    return int(getattr(args, name, default))


def _arg_float(args, name: str, default: float) -> float:
    return float(getattr(args, name, default))


def _choose_heads(d_model: int, requested: int) -> int:
    requested = max(1, int(requested))
    for h in range(min(requested, d_model), 0, -1):
        if d_model % h == 0:
            return h
    return 1


def _month_history_features(init_month: torch.Tensor, T: int) -> torch.Tensor:
    """Sin/cos month encoding for the input window."""
    device = init_month.device
    offsets = torch.arange(T, device=device).view(1, T)
    months = (init_month.view(-1, 1).long() - (T - 1 - offsets)) % 12
    months = months.float()
    return torch.stack(
        [
            torch.sin(2 * torch.pi * months / 12.0),
            torch.cos(2 * torch.pi * months / 12.0),
        ],
        dim=-1,
    )


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 dropout: float):
        super().__init__()
        pad = kernel_size // 2
        groups = min(8, out_channels)
        while out_channels % groups != 0 and groups > 1:
            groups -= 1
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=pad, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class OptionalPhysEncoder(nn.Module):
    """Encode optional scalar mechanism-history features."""

    def __init__(self, phys_dim: int, d_model: int, dropout: float):
        super().__init__()
        self.phys_dim = max(0, int(phys_dim))
        self.d_model = int(d_model)
        if self.phys_dim > 0:
            self.gru = nn.GRU(
                input_size=self.phys_dim,
                hidden_size=d_model,
                num_layers=1,
                batch_first=True,
            )
            self.out = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Dropout(dropout),
            )
        else:
            self.gru = None
            self.out = None

    def _align_dim(self, x_phys: torch.Tensor) -> torch.Tensor:
        p = x_phys.shape[-1]
        if p == self.phys_dim:
            return x_phys
        if p > self.phys_dim:
            return x_phys[..., :self.phys_dim]
        pad = torch.zeros(*x_phys.shape[:-1], self.phys_dim - p,
                          device=x_phys.device, dtype=x_phys.dtype)
        return torch.cat([x_phys, pad], dim=-1)

    def forward(self, x_phys: Optional[torch.Tensor], B: int, device, dtype) -> torch.Tensor:
        if self.phys_dim <= 0 or x_phys is None or x_phys.numel() == 0 or x_phys.shape[-1] == 0:
            return torch.zeros(B, self.d_model, device=device, dtype=dtype)
        x_phys = self._align_dim(x_phys.to(device=device, dtype=dtype))
        _, h = self.gru(x_phys)
        return self.out(h[-1])


class ConvLSTMCell(nn.Module):
    """Standard convolutional LSTM cell for a fixed spatial grid."""

    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        kernel_size: int = 3,
    ):
        super().__init__()
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("ConvLSTM kernel_size must be a positive odd integer")
        self.hidden_channels = int(hidden_channels)
        self.gates = nn.Conv2d(
            int(input_channels) + self.hidden_channels,
            4 * self.hidden_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        nn.init.zeros_(self.gates.bias)
        nn.init.ones_(
            self.gates.bias[
                self.hidden_channels : 2 * self.hidden_channels
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden, cell = state
        input_gate, forget_gate, output_gate, candidate = self.gates(
            torch.cat([x, hidden], dim=1)
        ).chunk(4, dim=1)
        input_gate = torch.sigmoid(input_gate)
        forget_gate = torch.sigmoid(forget_gate)
        output_gate = torch.sigmoid(output_gate)
        candidate = torch.tanh(candidate)
        cell = forget_gate * cell + input_gate * candidate
        hidden = output_gate * torch.tanh(cell)
        return hidden, cell


class ENSOConvLSTM(nn.Module):
    """Basic spatially recurrent ConvLSTM baseline for ENSO prediction.

    The global fields are pooled to a fixed grid before recurrence so memory
    use is independent of the native data resolution.  Unlike ``ENSOCNN``,
    which applies a 2-D CNN independently to each month and then uses a vector
    GRU, this model keeps a spatial hidden state throughout the 12-month input
    history.  It therefore supplies a distinct, conventional spatiotemporal
    recurrent baseline while retaining the shared physical-state encoder and
    lead-aware forecast decoder used by the other neural baselines.
    """

    def __init__(self, args):
        super().__init__()
        self.input_len = _arg_int(args, "input_len", 12)
        self.output_len = _arg_int(args, "output_len", 24)
        self.target_dim = _arg_int(args, "target_dim", 2)
        self.input_dim = _arg_int(
            args,
            "n_vars",
            _arg_int(
                args,
                "input_dim",
                len(getattr(args, "map_vars", [])) or 1,
            ),
        )
        self.phys_dim = _arg_int(
            args,
            "phys_dim",
            len(getattr(args, "phys_feature_names", [])),
        )
        self.d_model = _arg_int(args, "baseline_d_model", 128)
        self.dropout_p = _arg_float(
            args, "baseline_dropout", _arg_float(args, "dropout", 0.1)
        )
        self.hidden_channels = max(
            8, _arg_int(args, "convlstm_hidden_channels", 48)
        )
        self.num_layers = max(1, _arg_int(args, "convlstm_layers", 2))
        self.pool_lat = max(2, _arg_int(args, "convlstm_pool_lat", 16))
        self.pool_lon = max(2, _arg_int(args, "convlstm_pool_lon", 32))
        kernel_size = _arg_int(args, "convlstm_kernel_size", 3)

        groups = min(8, self.hidden_channels)
        while self.hidden_channels % groups != 0 and groups > 1:
            groups -= 1
        self.input_projection = nn.Sequential(
            nn.Conv2d(
                self.input_dim,
                self.hidden_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(groups, self.hidden_channels),
            nn.GELU(),
        )
        self.cells = nn.ModuleList(
            [
                ConvLSTMCell(
                    self.hidden_channels,
                    self.hidden_channels,
                    kernel_size=kernel_size,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.inter_layer_dropout = nn.ModuleList(
            [nn.Dropout2d(self.dropout_p) for _ in range(self.num_layers - 1)]
        )
        self.map_state_projection = nn.Sequential(
            nn.Linear(2 * self.hidden_channels, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.GELU(),
            nn.Dropout(self.dropout_p),
        )
        self.phys_encoder = OptionalPhysEncoder(
            self.phys_dim, self.d_model, self.dropout_p
        )
        self.state_projection = nn.Sequential(
            nn.LayerNorm(2 * self.d_model),
            nn.Linear(2 * self.d_model, self.d_model),
            nn.GELU(),
            nn.Dropout(self.dropout_p),
        )
        self.decoder = ProgressiveLeadDecoder(
            self.d_model,
            self.output_len,
            self.target_dim,
            dropout=self.dropout_p,
        )
        self.loss_diagnostics: Dict[str, torch.Tensor] = {}

    def forward(
        self,
        x_map: torch.Tensor,
        x_phys: Optional[torch.Tensor] = None,
        init_month: Optional[torch.Tensor] = None,
        target_years: Optional[torch.Tensor] = None,
        return_diagnostics: bool = False,
    ):
        del target_years
        batch_size, time_steps, channels, height, width = x_map.shape
        device = x_map.device
        dtype = x_map.dtype
        if init_month is None:
            init_month = torch.zeros(
                batch_size, device=device, dtype=torch.long
            )
        init_month = init_month.to(device=device).long()

        pooled = F.adaptive_avg_pool2d(
            x_map.reshape(
                batch_size * time_steps, channels, height, width
            ),
            (self.pool_lat, self.pool_lon),
        )
        encoded = self.input_projection(pooled).reshape(
            batch_size,
            time_steps,
            self.hidden_channels,
            self.pool_lat,
            self.pool_lon,
        )
        recurrent_dtype = encoded.dtype

        states = [
            (
                torch.zeros(
                    batch_size,
                    self.hidden_channels,
                    self.pool_lat,
                    self.pool_lon,
                    device=device,
                    dtype=recurrent_dtype,
                ),
                torch.zeros(
                    batch_size,
                    self.hidden_channels,
                    self.pool_lat,
                    self.pool_lon,
                    device=device,
                    dtype=recurrent_dtype,
                ),
            )
            for _ in range(self.num_layers)
        ]
        for time_index in range(time_steps):
            layer_input = encoded[:, time_index]
            next_states = []
            for layer_index, cell in enumerate(self.cells):
                hidden, memory = cell(layer_input, states[layer_index])
                next_states.append((hidden, memory))
                layer_input = hidden
                if layer_index < self.num_layers - 1:
                    layer_input = self.inter_layer_dropout[layer_index](
                        layer_input
                    )
            states = next_states

        final_hidden = states[-1][0]
        average_state = final_hidden.mean(dim=(2, 3))
        maximum_state = F.adaptive_max_pool2d(final_hidden, 1).flatten(1)
        map_state = self.map_state_projection(
            torch.cat([average_state, maximum_state], dim=-1)
        )
        phys_state = self.phys_encoder(
            x_phys, batch_size, device, dtype
        )
        state = self.state_projection(
            torch.cat([map_state, phys_state], dim=-1)
        )

        pred, logvar = self.decoder(state, init_month)
        self.loss_diagnostics = {"final_logvar": logvar}
        if return_diagnostics:
            return pred, {
                "logvar": logvar.detach(),
                "state": state.detach(),
                "map_state": map_state.detach(),
                "phys_state": phys_state.detach(),
            }
        return pred, None


class ENSOCNN(nn.Module):
    """CNN + temporal GRU ENSO baseline.

    Each monthly climate map is encoded by a shared 2-D CNN.  The sequence of
    monthly spatial embeddings is then summarized by a GRU and decoded by the
    same lead/month-aware forecast head used elsewhere in the project.
    """

    def __init__(self, args):
        super().__init__()
        self.input_len = _arg_int(args, "input_len", 12)
        self.output_len = _arg_int(args, "output_len", 24)
        self.target_dim = _arg_int(args, "target_dim", 2)
        self.input_dim = _arg_int(args, "n_vars", _arg_int(args, "input_dim", len(getattr(args, "map_vars", [])) or 1))
        self.phys_dim = _arg_int(args, "phys_dim", len(getattr(args, "phys_feature_names", [])))
        self.d_model = _arg_int(args, "baseline_d_model", _arg_int(args, "ctm_d_model", 128))
        self.dropout_p = _arg_float(args, "baseline_dropout", _arg_float(args, "dropout", 0.1))
        width = _arg_int(args, "cnn_width", 48)
        n_layers = max(1, _arg_int(args, "cnn_layers", 3))
        temporal_layers = max(1, _arg_int(args, "cnn_temporal_layers", 2))

        blocks = []
        in_ch = self.input_dim
        out_ch = width
        for i in range(n_layers):
            out_ch = width * min(2 ** i, 4)
            blocks.append(ConvBlock(in_ch, out_ch, kernel_size=5 if i == 0 else 3,
                                    dropout=self.dropout_p))
            if i < n_layers - 1:
                blocks.append(nn.AvgPool2d(kernel_size=2, stride=2))
            in_ch = out_ch
        self.map_encoder = nn.Sequential(*blocks)
        self.spatial_pool = nn.AdaptiveAvgPool2d(1)
        self.spatial_proj = nn.Sequential(
            nn.Linear(out_ch, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.GELU(),
            nn.Dropout(self.dropout_p),
        )
        self.time_emb = nn.Embedding(max(self.input_len, 1), self.d_model)
        self.hist_month_proj = nn.Sequential(
            nn.Linear(2, self.d_model),
            nn.GELU(),
            nn.LayerNorm(self.d_model),
        )
        self.temporal_gru = nn.GRU(
            input_size=self.d_model,
            hidden_size=self.d_model,
            num_layers=temporal_layers,
            batch_first=True,
            dropout=self.dropout_p if temporal_layers > 1 else 0.0,
        )
        self.phys_encoder = OptionalPhysEncoder(self.phys_dim, self.d_model, self.dropout_p)
        self.state_proj = nn.Sequential(
            nn.LayerNorm(self.d_model * 3),
            nn.Linear(self.d_model * 3, self.d_model),
            nn.GELU(),
            nn.Dropout(self.dropout_p),
        )
        self.decoder = ProgressiveLeadDecoder(
            self.d_model, self.output_len, self.target_dim, dropout=self.dropout_p
        )
        self.loss_diagnostics: Dict[str, torch.Tensor] = {}

    def forward(
        self,
        x_map: torch.Tensor,
        x_phys: Optional[torch.Tensor] = None,
        init_month: Optional[torch.Tensor] = None,
        target_years: Optional[torch.Tensor] = None,
        return_diagnostics: bool = False,
    ):
        del target_years
        B, T, C, H, W = x_map.shape
        device = x_map.device
        dtype = x_map.dtype
        if init_month is None:
            init_month = torch.zeros(B, device=device, dtype=torch.long)
        init_month = init_month.to(device=device).long()

        feat = self.map_encoder(x_map.reshape(B * T, C, H, W))
        feat = self.spatial_pool(feat).flatten(1)
        feat = self.spatial_proj(feat).reshape(B, T, self.d_model)

        time_ids = torch.arange(T, device=device).clamp(max=self.input_len - 1)
        feat = feat + self.time_emb(time_ids).unsqueeze(0)
        feat = feat + self.hist_month_proj(_month_history_features(init_month, T))

        temporal, h = self.temporal_gru(feat)
        final_state = h[-1]
        mean_state = temporal.mean(dim=1)
        phys_state = self.phys_encoder(x_phys, B, device, dtype)
        state = self.state_proj(torch.cat([final_state, mean_state, phys_state], dim=-1))

        pred, logvar = self.decoder(state, init_month)
        self.loss_diagnostics = {"final_logvar": logvar}
        if return_diagnostics:
            return pred, {
                "logvar": logvar.detach(),
                "state": state.detach(),
                "temporal_state": temporal.detach(),
            }
        return pred, None


class ENSOGeoformer(nn.Module):
    """Geoformer-style spatiotemporal Transformer baseline.

    This is a compact 3-D Geoformer-style implementation for the project data:
    monthly maps are first pooled to a fixed latitude-longitude grid, split into
    spatial patches for each month, encoded by a Transformer, and queried by
    lead/month-conditioned decoder tokens.
    """

    def __init__(self, args):
        super().__init__()
        self.input_len = _arg_int(args, "input_len", 12)
        self.output_len = _arg_int(args, "output_len", 24)
        self.target_dim = _arg_int(args, "target_dim", 2)
        self.input_dim = _arg_int(args, "n_vars", _arg_int(args, "input_dim", len(getattr(args, "map_vars", [])) or 1))
        self.phys_dim = _arg_int(args, "phys_dim", len(getattr(args, "phys_feature_names", [])))
        self.d_model = _arg_int(args, "baseline_d_model", _arg_int(args, "ctm_d_model", 128))
        self.dropout_p = _arg_float(args, "baseline_dropout", _arg_float(args, "dropout", 0.1))
        requested_heads = _arg_int(args, "baseline_n_heads", _arg_int(args, "ctm_n_heads", 4))
        n_heads = _choose_heads(self.d_model, requested_heads)
        depth = max(1, _arg_int(args, "baseline_depth", 3))
        decoder_depth = max(1, _arg_int(args, "geoformer_decoder_depth", 2))
        ffn_mult = max(2, _arg_int(args, "baseline_ffn_mult", 4))
        self.pool_lat = max(2, _arg_int(args, "geoformer_pool_lat", _arg_int(args, "ctm_pool_lat", 8)))
        self.pool_lon = max(2, _arg_int(args, "geoformer_pool_lon", _arg_int(args, "ctm_pool_lon", 16)))
        patch = max(1, _arg_int(args, "geoformer_patch_size", 2))
        self.patch_size = min(patch, self.pool_lat, self.pool_lon)
        self.grid_lat = max(1, (self.pool_lat - self.patch_size) // self.patch_size + 1)
        self.grid_lon = max(1, (self.pool_lon - self.patch_size) // self.patch_size + 1)

        self.patch_embed = nn.Conv2d(
            self.input_dim, self.d_model,
            kernel_size=self.patch_size, stride=self.patch_size,
        )
        self.time_emb = nn.Embedding(max(self.input_len, 1), self.d_model)
        self.lat_emb = nn.Embedding(self.grid_lat, self.d_model)
        self.lon_emb = nn.Embedding(self.grid_lon, self.d_model)
        self.token_norm = nn.LayerNorm(self.d_model)
        self.token_drop = nn.Dropout(self.dropout_p)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=n_heads,
            dim_feedforward=self.d_model * ffn_mult,
            dropout=self.dropout_p,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=depth)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=self.d_model,
            nhead=n_heads,
            dim_feedforward=self.d_model * ffn_mult,
            dropout=self.dropout_p,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=decoder_depth)

        self.phys_encoder = OptionalPhysEncoder(self.phys_dim, self.d_model, self.dropout_p)
        self.lead_emb = nn.Embedding(self.output_len, self.d_model)
        self.future_month_proj = nn.Sequential(
            nn.Linear(4, self.d_model),
            nn.GELU(),
            nn.LayerNorm(self.d_model),
        )
        self.query_norm = nn.LayerNorm(self.d_model)
        self.head = nn.Sequential(
            nn.LayerNorm(self.d_model * 3),
            nn.Linear(self.d_model * 3, self.d_model),
            nn.GELU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(self.d_model, self.target_dim),
        )
        self.logvar_head = nn.Sequential(
            nn.LayerNorm(self.d_model * 3),
            nn.Linear(self.d_model * 3, self.d_model // 2),
            nn.GELU(),
            nn.Linear(self.d_model // 2, self.target_dim),
        )
        self.loss_diagnostics: Dict[str, torch.Tensor] = {}

    def _future_month_features(self, init_month: torch.Tensor, B: int, device) -> torch.Tensor:
        leads = torch.arange(1, self.output_len + 1, device=device).view(1, -1)
        target_month = (init_month.view(B, 1).long() + leads) % 12
        init_m = init_month.view(B, 1).float().expand(B, self.output_len)
        tm = target_month.float()
        return torch.stack(
            [
                torch.sin(2 * torch.pi * tm / 12.0),
                torch.cos(2 * torch.pi * tm / 12.0),
                torch.sin(2 * torch.pi * init_m / 12.0),
                torch.cos(2 * torch.pi * init_m / 12.0),
            ],
            dim=-1,
        )

    def _patch_tokens(self, x_map: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x_map.shape
        pooled = F.adaptive_avg_pool2d(
            x_map.reshape(B * T, C, H, W), (self.pool_lat, self.pool_lon)
        )
        patches = self.patch_embed(pooled)
        _, D, Gh, Gw = patches.shape
        patches = patches.reshape(B, T, D, Gh, Gw).permute(0, 1, 3, 4, 2)
        tokens = patches.reshape(B, T * Gh * Gw, D)

        device = x_map.device
        t_ids = torch.arange(T, device=device).clamp(max=self.input_len - 1)
        t_ids = t_ids.view(T, 1, 1).expand(T, Gh, Gw).reshape(-1)
        lat_ids = torch.arange(Gh, device=device).view(1, Gh, 1).expand(T, Gh, Gw).reshape(-1)
        lon_ids = torch.arange(Gw, device=device).view(1, 1, Gw).expand(T, Gh, Gw).reshape(-1)
        tokens = tokens + self.time_emb(t_ids).unsqueeze(0)
        tokens = tokens + self.lat_emb(lat_ids).unsqueeze(0)
        tokens = tokens + self.lon_emb(lon_ids).unsqueeze(0)
        return self.token_drop(self.token_norm(tokens))

    def forward(
        self,
        x_map: torch.Tensor,
        x_phys: Optional[torch.Tensor] = None,
        init_month: Optional[torch.Tensor] = None,
        target_years: Optional[torch.Tensor] = None,
        return_diagnostics: bool = False,
    ):
        del target_years
        B = x_map.shape[0]
        device = x_map.device
        dtype = x_map.dtype
        if init_month is None:
            init_month = torch.zeros(B, device=device, dtype=torch.long)
        init_month = init_month.to(device=device).long()

        src = self._patch_tokens(x_map)
        memory = self.encoder(src)
        memory_mean = memory.mean(dim=1)
        phys_state = self.phys_encoder(x_phys, B, device, dtype)

        lead_ids = torch.arange(self.output_len, device=device).unsqueeze(0).expand(B, -1)
        lead_query = self.lead_emb(lead_ids)
        month_query = self.future_month_proj(self._future_month_features(init_month, B, device))
        query = self.query_norm(lead_query + month_query + phys_state.unsqueeze(1))

        decoded = self.decoder(query, memory)
        context = memory_mean.unsqueeze(1).expand(B, self.output_len, self.d_model)
        h = torch.cat([decoded, query, context], dim=-1)
        pred = self.head(h)
        logvar = self.logvar_head(h).clamp(min=-6.0, max=4.0)

        self.loss_diagnostics = {"final_logvar": logvar}
        if return_diagnostics:
            return pred, {
                "logvar": logvar.detach(),
                "memory": memory.detach(),
                "query": query.detach(),
                "decoded": decoded.detach(),
            }
        return pred, None


class ENSOGLGeoformer(nn.Module):
    """Lightweight global-local Geoformer for ENSO upper-bound experiments.

    The encoder deliberately separates the two operations needed by the
    scientific question:

    1. local temporal attention learns the evolution at every pooled grid cell;
    2. global spatial attention exchanges information among ocean basins after
       each cell's history has been summarized.

    Lead/month queries then cross-attend to the spatial memory.  Optional
    mechanism-index histories enter through a small GRU and do not increase the
    map-token count.
    """

    def __init__(self, args):
        super().__init__()
        self.input_len = _arg_int(args, "input_len", 12)
        self.output_len = _arg_int(args, "output_len", 24)
        self.target_dim = _arg_int(args, "target_dim", 2)
        self.input_dim = _arg_int(
            args,
            "n_vars",
            _arg_int(args, "input_dim", len(getattr(args, "map_vars", [])) or 1),
        )
        self.phys_dim = _arg_int(
            args, "phys_dim", len(getattr(args, "phys_feature_names", []))
        )
        self.d_model = _arg_int(args, "baseline_d_model", 96)
        self.dropout_p = _arg_float(
            args, "baseline_dropout", _arg_float(args, "dropout", 0.10)
        )
        requested_heads = _arg_int(args, "baseline_n_heads", 4)
        n_heads = _choose_heads(self.d_model, requested_heads)
        ffn_mult = max(2, _arg_int(args, "baseline_ffn_mult", 3))
        temporal_depth = max(1, _arg_int(args, "gl_temporal_depth", 2))
        spatial_depth = max(1, _arg_int(args, "gl_spatial_depth", 2))
        decoder_depth = max(1, _arg_int(args, "gl_decoder_depth", 1))
        self.pool_lat = max(2, _arg_int(args, "gl_pool_lat", 10))
        self.pool_lon = max(2, _arg_int(args, "gl_pool_lon", 20))
        self.n_cells = self.pool_lat * self.pool_lon

        self.map_projection = nn.Sequential(
            nn.Linear(self.input_dim, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.GELU(),
        )
        self.time_emb = nn.Embedding(max(self.input_len, 1), self.d_model)
        self.history_month_proj = nn.Sequential(
            nn.Linear(2, self.d_model),
            nn.GELU(),
            nn.LayerNorm(self.d_model),
        )
        self.lat_emb = nn.Embedding(self.pool_lat, self.d_model)
        self.lon_emb = nn.Embedding(self.pool_lon, self.d_model)
        self.input_norm = nn.LayerNorm(self.d_model)
        self.input_drop = nn.Dropout(self.dropout_p)

        temporal_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=n_heads,
            dim_feedforward=self.d_model * ffn_mult,
            dropout=self.dropout_p,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            temporal_layer, num_layers=temporal_depth
        )
        self.temporal_pool = nn.Linear(self.d_model, 1)
        self.local_norm = nn.LayerNorm(self.d_model)

        spatial_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=n_heads,
            dim_feedforward=self.d_model * ffn_mult,
            dropout=self.dropout_p,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.spatial_encoder = nn.TransformerEncoder(
            spatial_layer, num_layers=spatial_depth
        )
        self.spatial_norm = nn.LayerNorm(self.d_model)

        self.phys_encoder = OptionalPhysEncoder(
            self.phys_dim, self.d_model, self.dropout_p
        )
        self.lead_emb = nn.Embedding(self.output_len, self.d_model)
        self.future_month_proj = nn.Sequential(
            nn.Linear(4, self.d_model),
            nn.GELU(),
            nn.LayerNorm(self.d_model),
        )
        self.query_norm = nn.LayerNorm(self.d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.d_model,
            nhead=n_heads,
            dim_feedforward=self.d_model * ffn_mult,
            dropout=self.dropout_p,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=decoder_depth
        )
        self.head = nn.Sequential(
            nn.LayerNorm(self.d_model * 3),
            nn.Linear(self.d_model * 3, self.d_model),
            nn.GELU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(self.d_model, self.target_dim),
        )
        self.logvar_head = nn.Sequential(
            nn.LayerNorm(self.d_model * 3),
            nn.Linear(self.d_model * 3, max(16, self.d_model // 2)),
            nn.GELU(),
            nn.Linear(max(16, self.d_model // 2), self.target_dim),
        )
        self.loss_diagnostics: Dict[str, torch.Tensor] = {}

    def _spatial_positions(self, device: torch.device) -> torch.Tensor:
        lat_ids = torch.arange(self.pool_lat, device=device)
        lon_ids = torch.arange(self.pool_lon, device=device)
        lat_grid = lat_ids[:, None].expand(self.pool_lat, self.pool_lon).reshape(-1)
        lon_grid = lon_ids[None, :].expand(self.pool_lat, self.pool_lon).reshape(-1)
        return self.lat_emb(lat_grid) + self.lon_emb(lon_grid)

    def _future_month_features(
        self, init_month: torch.Tensor, batch: int, device: torch.device
    ) -> torch.Tensor:
        leads = torch.arange(1, self.output_len + 1, device=device).view(1, -1)
        target_month = (init_month.view(batch, 1).long() + leads) % 12
        init_values = init_month.view(batch, 1).float().expand(batch, self.output_len)
        target_values = target_month.float()
        return torch.stack(
            [
                torch.sin(2 * torch.pi * target_values / 12.0),
                torch.cos(2 * torch.pi * target_values / 12.0),
                torch.sin(2 * torch.pi * init_values / 12.0),
                torch.cos(2 * torch.pi * init_values / 12.0),
            ],
            dim=-1,
        )

    def _encode_maps(
        self, x_map: torch.Tensor, init_month: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, history, channels, height, width = x_map.shape
        pooled = F.adaptive_avg_pool2d(
            x_map.reshape(batch * history, channels, height, width),
            (self.pool_lat, self.pool_lon),
        )
        pooled = pooled.reshape(
            batch, history, channels, self.pool_lat, self.pool_lon
        )
        series = pooled.permute(0, 3, 4, 1, 2).reshape(
            batch * self.n_cells, history, channels
        )
        tokens = self.map_projection(series)

        time_ids = torch.arange(history, device=x_map.device).clamp(
            max=self.input_len - 1
        )
        month_features = self.history_month_proj(
            _month_history_features(init_month, history)
        )
        month_features = month_features[:, None].expand(
            batch, self.n_cells, history, self.d_model
        ).reshape(batch * self.n_cells, history, self.d_model)
        positions = self._spatial_positions(x_map.device)
        positions = positions[None, :, None, :].expand(
            batch, self.n_cells, history, self.d_model
        ).reshape(batch * self.n_cells, history, self.d_model)
        tokens = tokens + self.time_emb(time_ids)[None] + month_features + positions
        tokens = self.input_drop(self.input_norm(tokens))

        temporal = self.temporal_encoder(tokens)
        temporal_weights = torch.softmax(self.temporal_pool(temporal), dim=1)
        local = (temporal_weights * temporal).sum(dim=1)
        local = self.local_norm(local).reshape(batch, self.n_cells, self.d_model)
        memory = self.spatial_norm(self.spatial_encoder(local))
        return memory, temporal_weights.reshape(batch, self.n_cells, history)

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

        memory, temporal_weights = self._encode_maps(x_map, init_month)
        memory_mean = memory.mean(dim=1)
        phys_state = self.phys_encoder(x_phys, batch, device, dtype)

        lead_ids = torch.arange(self.output_len, device=device)[None].expand(batch, -1)
        query = self.lead_emb(lead_ids)
        query = query + self.future_month_proj(
            self._future_month_features(init_month, batch, device)
        )
        query = self.query_norm(query + phys_state[:, None, :])
        decoded = self.decoder(query, memory)
        context = memory_mean[:, None, :].expand(batch, self.output_len, self.d_model)
        joint = torch.cat([decoded, query, context], dim=-1)
        prediction = self.head(joint)
        logvar = self.logvar_head(joint).clamp(min=-6.0, max=4.0)
        self.loss_diagnostics = {"final_logvar": logvar}

        if return_diagnostics:
            return prediction, {
                "logvar": logvar.detach(),
                "spatial_memory": memory.detach(),
                "temporal_weights": temporal_weights.detach(),
            }
        return prediction, None
