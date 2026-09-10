"""Report trainable parameter counts for the four core architectures.

Run this on the training environment because it imports PyTorch.  The defaults
match ``train_spb_core_claim_one.sh`` and report the fixed four-channel model
capacity used by every configuration.  HC300 remains loaded to hold capacity
constant but is masked out of every core-claim input coalition.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train_v2 import build_model, build_parser


def make_args(model_name: str, map_vars: str):
    parser = build_parser()
    args = parser.parse_args(
        [
            "--stage", "train",
            "--model_name", model_name,
            "--map_vars", map_vars,
            "--phys_vars", "nino34,wwv,thermocline_tilt",
            "--target_vars", "nino34,thermocline_tilt",
            "--input_len", "12",
            "--output_len", "24",
        ]
    )
    args.map_vars = [item.strip() for item in map_vars.split(",") if item.strip()]
    args.input_dim = args.n_vars = len(args.map_vars)
    args.phys_dim = 7
    args.base_phys_dim = 3
    args.phys_feature_names = [
        "nino34", "wwv", "thermocline_tilt",
        "input_month_sin", "input_month_cos",
        "init_month_sin", "init_month_cos",
    ]
    args.target_dim = 2
    args.img_height = 121
    args.img_width = 180
    args.atmos_d_model = 96
    args.atmos_n_heads = 4
    args.atmos_pool_lat = 10
    args.atmos_pool_lon = 20
    args.atmos_temporal_depth = 2
    args.atmos_spatial_depth = 2
    args.atmos_group_depth = 2
    args.atmos_decoder_depth = 1
    args.atmos_ffn_mult = 3
    args.atmos_dropout = 0.12
    args.atmos_coalition_probability = 1.0
    args.atmos_mask_strategy = "balanced"
    args.atmos_full_mask_probability = 0.55
    args.atmos_leave_basin_probability = 0.20
    args.atmos_leave_variable_probability = 0.15
    args.atmos_group_drop = 0.04
    args.atmos_basin_drop = 0.02
    args.atmos_variable_drop = 0.02
    args.atmos_spring_scale = 0.12
    args.atmos_phase_summary_scale = 0.20
    args.atmos_router_prior_strength = 0.85
    args.atmos_multiscale_phase = True
    args.atmos_phase_film = True
    args.atmos_phase_film_scale = 0.18
    args.atmos_phase_film_lead_power = 1.0
    args.atmos_target_calendar_decay = 0.65
    args.atmos_anomaly_calibration = model_name == "AtmosFormerSPB"
    args.atmos_anomaly_gain_scale = 0.40
    args.atmos_anomaly_offset_scale = 0.25
    args.atmos_regime_moe = model_name == "AtmosFormerSPB"
    args.atmos_horizon_router = model_name == "AtmosFormerSPB"
    args.baseline_d_model = 128
    args.baseline_n_heads = 4
    args.baseline_depth = 3
    args.baseline_ffn_mult = 4
    args.baseline_dropout = 0.15
    args.dropout = 0.12
    args.cnn_width = 48
    args.cnn_layers = 3
    args.cnn_temporal_layers = 2
    args.convlstm_hidden_channels = 48
    args.convlstm_layers = 2
    args.convlstm_pool_lat = 16
    args.convlstm_pool_lon = 32
    args.convlstm_kernel_size = 3
    args.geoformer_pool_lat = 8
    args.geoformer_pool_lon = 16
    args.geoformer_patch_size = 2
    args.geoformer_decoder_depth = 2
    return args


def count(model_name: str, map_vars: str) -> dict:
    model = build_model(make_args(model_name, map_vars))
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    return {
        "model_name": model_name,
        "map_vars": map_vars,
        "trainable_parameters": int(trainable),
        "total_parameters": int(total),
        "trainable_millions": trainable / 1e6,
        "parameter_fp32_mib": trainable * 4 / (1024 ** 2),
        "adamw_fp32_state_mib": trainable * 16 / (1024 ** 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--map_vars", default="slp,sst,hc300,tauu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = [
        count(model, args.map_vars)
        for model in (
            "AtmosFormerSPB",
            "ENSOCNN",
            "ENSOConvLSTM",
            "ENSOGeoformer",
        )
    ]
    print(json.dumps(rows, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
