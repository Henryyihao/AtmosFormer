"""Shared lead-aware forecast decoder for the neural baselines."""
from __future__ import annotations

import torch
import torch.nn as nn


class ProgressiveLeadDecoder(nn.Module):
    """Decode one encoded state into a lead-by-lead forecast trajectory."""

    def __init__(
        self,
        d_model: int,
        output_len: int,
        target_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.output_len = output_len
        self.target_dim = target_dim

        self.lead_emb = nn.Embedding(output_len, d_model)
        self.month_proj = nn.Sequential(
            nn.Linear(4, d_model // 2),
            nn.GELU(),
            nn.LayerNorm(d_model // 2),
        )
        self.temporal_gru = nn.GRU(
            input_size=d_model + d_model // 2,
            hidden_size=d_model,
            num_layers=2,
            batch_first=True,
            dropout=dropout if dropout > 0 else 0.0,
        )

        head_in = d_model * 2 + d_model // 2
        self.pred_head = nn.Sequential(
            nn.Linear(head_in, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, target_dim),
        )
        self.logvar_head = nn.Sequential(
            nn.Linear(head_in, d_model // 2),
            nn.GELU(),
            nn.LayerNorm(d_model // 2),
            nn.Linear(d_model // 2, target_dim),
        )

    def _target_month_features(
        self,
        init_month: torch.Tensor,
        batch_size: int,
        device,
    ) -> torch.Tensor:
        leads = torch.arange(1, self.output_len + 1, device=device).view(1, -1)
        target_month = (init_month.view(batch_size, 1).long() + leads) % 12
        init_m = init_month.view(batch_size, 1).float().expand(
            batch_size, self.output_len
        )
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

    def forward(
        self,
        sync_state: torch.Tensor,
        init_month: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = sync_state.shape[0]
        device = sync_state.device
        state = sync_state.unsqueeze(1).expand(
            batch_size, self.output_len, self.d_model
        )

        lead_ids = torch.arange(self.output_len, device=device).unsqueeze(0)
        lead_ids = lead_ids.expand(batch_size, -1)
        lead_feat = self.lead_emb(lead_ids)
        month_feat = self.month_proj(
            self._target_month_features(init_month, batch_size, device)
        )

        gru_input = torch.cat([lead_feat, month_feat], dim=-1)
        h0 = sync_state.unsqueeze(0).expand(2, batch_size, self.d_model).contiguous()
        gru_out, _ = self.temporal_gru(gru_input, h0)

        features = torch.cat([state, gru_out, month_feat], dim=-1)
        pred = self.pred_head(features)
        logvar = self.logvar_head(features).clamp(min=-6.0, max=4.0)
        return pred, logvar
