"""Run reproduatmosle event hindcasts for the SPB core-claim figures.

The default experiment evaluates several input-space masks on the same
AtmosFormer checkpoint.  This keeps the event curves paired: only the retained
input coalition changes, while weights, samples and the forecast protocol stay
fixed.  The default event list is registered before plotting; ``--event_years
auto`` is available for exploratory screening and selects the strongest warm
and cold April-initialized events from the observed test trajectory.

The evaluator writes real dates from the NetCDF time coordinate.  Do not infer
event dates from the integer sample id in the training CSV files.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVALUATOR = PROJECT_ROOT / "scripts" / "evaluate_atmos_spb_evidence.py"
CHECKPOINT_PREFIX = "SPBCore_atmosformer_"
DEFAULT_EVENT_YEARS = (1982, 1987, 1997, 2009, 2015, 2020, 2023)
DEFAULT_MASKS = (
    "full",
    "state_only",
    "variable_pair_slp_sst",
    "basin_tp",
    "basin_pair_tp_atlantic",
    "basin_triplet_tp_indian_atlantic",
)


def parse_ints(value: str) -> List[int]:
    values = [item.strip() for item in str(value).split(",") if item.strip()]
    if not values:
        raise ValueError("Expected a non-empty comma-separated integer list")
    return [int(item) for item in values]


def parse_years(value: str) -> tuple[List[int], bool]:
    value = str(value).strip().lower()
    if value == "auto":
        return [], True
    if ":" in value:
        start, stop = (int(item) for item in value.split(":", 1))
        if stop < start:
            raise ValueError("Event year range must be ascending")
        return list(range(start, stop + 1)), False
    return parse_ints(value), False


def select_best_seed(training_results_dir: Path, config: str, metric: str) -> tuple[int, pd.DataFrame]:
    """Select one seed within a fixed configuration, never across configs."""
    rows = []
    for summary_path in sorted(training_results_dir.glob("**/spb_summary.json")):
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("model_key") != "atmosformer" or payload.get("configuration") != config:
            continue
        try:
            rows.append(
                {
                    "seed": int(payload["seed"]),
                    "acc_6_18": float(payload.get("nino34_mean_acc_6_18", np.nan)),
                    "effective_lead": float(payload.get("effective_lead_3ma_acc_05", np.nan)),
                    "rmse_6_18": float(payload.get("nino34_mean_rmse_6_18", np.nan)),
                    "summary_path": str(summary_path),
                    "_mtime": summary_path.stat().st_mtime,
                }
            )
        except (KeyError, TypeError, ValueError):
            continue
    table = pd.DataFrame(rows).sort_values("_mtime").drop_duplicates("seed", keep="last").drop(columns="_mtime")
    if table.empty:
        raise FileNotFoundError(
            f"No AtmosFormer summaries for config={config} below {training_results_dir}"
        )
    metric_column = {
        "nino34_mean_acc_6_18": "acc_6_18",
        "effective_lead_3ma_acc_05": "effective_lead",
        "nino34_mean_rmse_6_18": "rmse_6_18",
    }[metric]
    table = table[np.isfinite(table[metric_column])].copy()
    if table.empty:
        raise ValueError(f"Metric {metric} is unavailable for config={config}")
    ascending = metric == "nino34_mean_rmse_6_18"
    table = table.sort_values(
        [metric_column, "effective_lead", "rmse_6_18", "seed"],
        ascending=[ascending, False, True, True],
    ).reset_index(drop=True)
    return int(table.iloc[0].seed), table


def resolve_checkpoint(root: Path, config: str, seed: int) -> Path:
    exact = root / f"{CHECKPOINT_PREFIX}{config}_seed{seed}.pth"
    if exact.is_file():
        return exact.resolve()
    candidates = [
        path
        for path in root.glob(f"{CHECKPOINT_PREFIX}{config}_seed{seed}*.pth")
    ]
    candidates.sort(key=lambda path: path.stat().st_mtime)
    if candidates:
        return candidates[-1].resolve()
    raise FileNotFoundError(
        f"No validation-best checkpoint for config={config}, seed={seed} below {root}"
    )


def _event_file(path: Path) -> Path | None:
    candidate = path / "event_hindcasts_full.csv"
    return candidate if candidate.is_file() and candidate.stat().st_size > 0 else None


def run_evaluator(
    checkpoint: Path,
    output_dir: Path,
    source_label: str,
    masks: Sequence[str],
    event_years: Sequence[int],
    obs_path: str,
    manifest: Path,
    batch_size: int,
    device: str,
    force: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    expected = [_event_file(output_dir)]
    metadata_ok = False
    metadata_path = output_dir / "metadata.json"
    if metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata_ok = set(int(value) for value in metadata.get("event_years", [])) == set(int(value) for value in event_years)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            metadata_ok = False
    complete = (
        not force
        and expected[0] is not None
        and metadata_ok
        and all((output_dir / f"event_hindcasts_{name}.csv").is_file() for name in masks)
    )
    if complete:
        print(f"[Skip] event backtest exists: {output_dir}")
        return
    command = [
        sys.executable,
        str(EVALUATOR),
        "--checkpoint",
        str(checkpoint),
        "--source_label",
        source_label,
        "--output_dir",
        str(output_dir),
        "--manifest",
        str(manifest),
        "--mask_names",
        ",".join(masks),
        "--batch_size",
        str(int(batch_size)),
        "--bootstrap_samples",
        "0",
        "--device",
        device,
        "--event_years",
        ",".join(str(year) for year in event_years),
    ]
    if obs_path:
        command.extend(["--obs_path", obs_path])
    print("[Run]", " ".join(command))
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def load_event_truth(path: Path) -> pd.DataFrame:
    frames = []
    for file in sorted(path.glob("event_hindcasts_full.csv")):
        frame = pd.read_csv(file)
        if not frame.empty:
            frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No event_hindcasts_full.csv below {path}")
    frame = pd.concat(frames, ignore_index=True)
    required = {"event_year", "lead", "true_nino34"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Event table is missing columns: {sorted(missing)}")
    return frame


def select_events(
    frame: pd.DataFrame,
    warm_count: int,
    cold_count: int,
    threshold: float,
) -> pd.DataFrame:
    rows = []
    for year, group in frame.groupby("event_year", sort=True):
        truth = group.sort_values("lead").drop_duplicates("lead")
        values = truth.true_nino34.to_numpy(dtype=float)
        if values.size == 0 or not np.isfinite(values).any():
            continue
        valid = np.isfinite(values)
        peak_index = int(np.nanargmax(np.abs(values[valid])))
        valid_leads = truth.loc[valid, "lead"].to_numpy(dtype=int)
        valid_values = values[valid]
        peak_value = float(valid_values[peak_index])
        if abs(peak_value) < float(threshold):
            continue
        phase = "warm" if peak_value > 0 else "cold"
        rows.append(
            {
                "event_year": int(year),
                "phase": phase,
                "peak_nino34": peak_value,
                "peak_abs_nino34": abs(peak_value),
                "peak_lead": int(valid_leads[peak_index]),
            }
        )
    selected = pd.DataFrame(rows)
    if selected.empty:
        raise RuntimeError("Automatic event screening found no event above the threshold")
    warm = selected.query("phase == 'warm'").sort_values("peak_abs_nino34", ascending=False).head(int(warm_count))
    cold = selected.query("phase == 'cold'").sort_values("peak_abs_nino34", ascending=False).head(int(cold_count))
    selected = pd.concat([warm, cold], ignore_index=True).sort_values("event_year")
    selected.insert(0, "selection", "auto_strength")
    return selected.reset_index(drop=True)


def aggregate_outputs(output_root: Path, run_dirs: Sequence[Path], selected: pd.DataFrame) -> None:
    frames = []
    for run_dir in run_dirs:
        match = re.match(r"atmosformer_(.+)_seed(\d+)$", run_dir.name)
        if not match:
            continue
        config, seed_text = match.groups()
        seed = int(seed_text)
        for path in sorted(run_dir.glob("event_hindcasts_*.csv")):
            frame = pd.read_csv(path)
            if frame.empty:
                continue
            frame.insert(0, "seed", seed)
            frame.insert(0, "configuration", config)
            frame.insert(0, "model_key", "atmosformer")
            frames.append(frame)
    if frames:
        merged = pd.concat(frames, ignore_index=True)
        merged.to_csv(output_root / "event_hindcasts_all.csv", index=False)
    selected.to_csv(output_root / "selected_events.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint_root", type=Path, default=PROJECT_ROOT / "checkpoints_spb_core_claim")
    parser.add_argument("--output_dir", type=Path, default=PROJECT_ROOT / "results_spb_core_claim" / "event_backtest")
    parser.add_argument("--config", default="global_slp_sst_tauu", help="AtmosFormer checkpoint configuration")
    parser.add_argument("--seeds", default="best", help="best, all, or a comma-separated seed list")
    parser.add_argument("--training_results_dir", type=Path, default=PROJECT_ROOT / "results_spb_core_claim_training")
    parser.add_argument(
        "--best_seed_metric",
        choices=("nino34_mean_acc_6_18", "effective_lead_3ma_acc_05", "nino34_mean_rmse_6_18"),
        default="nino34_mean_acc_6_18",
    )
    parser.add_argument("--mask_names", default=",".join(DEFAULT_MASKS))
    parser.add_argument("--event_years", default=",".join(str(year) for year in DEFAULT_EVENT_YEARS))
    parser.add_argument("--warm_count", type=int, default=4)
    parser.add_argument("--cold_count", type=int, default=3)
    parser.add_argument("--event_threshold", type=float, default=0.5)
    parser.add_argument("--obs_path", default=os.environ.get("OBS_PATH", ""))
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "configs" / "atmos_spb_evidence_manifest.csv")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", default=os.environ.get("DEVICE", "auto"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    seed_selection = str(args.seeds).strip().lower()
    if seed_selection == "best":
        best_seed, seed_table = select_best_seed(args.training_results_dir, args.config, args.best_seed_metric)
        seeds = [best_seed]
        seed_selection_payload = {
            "mode": "best",
            "configuration": args.config,
            "metric": args.best_seed_metric,
            "selected_seed": best_seed,
            "candidates": seed_table.to_dict(orient="records"),
        }
        print(f"[SPB events] selected best seed={best_seed} by {args.best_seed_metric} for {args.config}")
    elif seed_selection == "all":
        seeds = [2025, 2026, 2027, 2028, 2029]
        seed_selection_payload = {"mode": "all", "configuration": args.config, "metric": None, "selected_seed": None}
    else:
        seeds = parse_ints(args.seeds)
        seed_selection_payload = {"mode": "manual", "configuration": args.config, "metric": None, "selected_seed": None, "seeds": seeds}
    masks = [item.strip() for item in args.mask_names.split(",") if item.strip()]
    years, auto = parse_years(args.event_years)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    selected = pd.DataFrame()
    if auto:
        selector_seed = seeds[0]
        selector_checkpoint = resolve_checkpoint(args.checkpoint_root, args.config, selector_seed)
        selector_dir = args.output_dir / f"_screen_all_{args.config}_seed{selector_seed}"
        all_years = list(range(1980, 2026))
        run_evaluator(
            selector_checkpoint,
            selector_dir,
            f"event_screen_{args.config}_seed{selector_seed}",
            ["full"],
            all_years,
            args.obs_path,
            args.manifest,
            args.batch_size,
            args.device,
            args.force,
        )
        selected = select_events(load_event_truth(selector_dir), args.warm_count, args.cold_count, args.event_threshold)
        years = selected.event_year.astype(int).tolist()
    else:
        phase_by_year = {
            1982: "warm",
            1997: "warm",
            2015: "warm",
            2023: "warm",
            1987: "transition",
            2009: "transition",
            2020: "cold",
        }
        selected = pd.DataFrame(
            {
                "selection": "pre_registered",
                "event_year": years,
                "phase": [phase_by_year.get(year, "event") for year in years],
            }
        )

    run_dirs = []
    for seed in seeds:
        checkpoint = resolve_checkpoint(args.checkpoint_root, args.config, seed)
        run_dir = args.output_dir / f"atmosformer_{args.config}_seed{seed}"
        run_dirs.append(run_dir)
        run_evaluator(
            checkpoint,
            run_dir,
            f"event_{args.config}_seed{seed}",
            masks,
            years,
            args.obs_path,
            args.manifest,
            args.batch_size,
            args.device,
            args.force,
        )
        metadata = {
            "model_key": "atmosformer",
            "configuration": args.config,
            "seed": int(seed),
            "checkpoint": str(checkpoint),
            "checkpoint_kind": "validation_best",
            "event_years": years,
            "event_selection": "auto_strength" if auto else "pre_registered",
            "mask_names": masks,
        }
        (run_dir / "backtest_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    if auto:
        selected["selection"] = "auto_strength"
    aggregate_outputs(args.output_dir, run_dirs, selected)
    (args.output_dir / "seed_selection.json").write_text(
        json.dumps(seed_selection_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[SPB events] wrote {args.output_dir / 'event_hindcasts_all.csv'}")
    print(f"[SPB events] selected years: {','.join(str(year) for year in years)}")


if __name__ == "__main__":
    main()
