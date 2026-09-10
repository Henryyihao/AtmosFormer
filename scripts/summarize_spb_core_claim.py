"""Summarize the frozen SPB core-claim training matrix.

The reported effects are paired input-space forecast utilities.  They are
descriptive evidence about information used by the predictor, not dynamical
causal estimates.  Positive values always mean a better forecast: ACC is used
directly and RMSE is sign-reversed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd


BASIN_FORMULAS = {
    "indian": ("basin_p_i", "basin_p"),
    "atlantic": ("basin_p_a", "basin_p"),
    "interaction": ("basin_p_i_a", "basin_p_i", "basin_p_a", "basin_p"),
    "total_remote": ("basin_p_i_a", "basin_p"),
    "global_context": ("global_slp_sst_tauu", "basin_p_i_a"),
}

VARIABLE_FORMULAS = {
    "slp_gain_over_state": ("global_slp", "state_only"),
    "sst_gain_over_state": ("global_sst", "state_only"),
    "slp_sst_complementarity": (
        "global_slp_sst", "global_slp", "global_sst", "state_only"
    ),
    "tauu_increment": ("global_slp_sst_tauu", "global_slp_sst"),
    "total_map_gain": ("global_slp_sst_tauu", "state_only"),
}


def parse_seeds(value: str) -> List[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def discover(root: Path) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for path in sorted(root.glob("**/spb_summary.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        required = {"model_key", "configuration", "seed"}
        if not required.issubset(payload):
            continue
        payload = dict(payload)
        payload["summary_path"] = str(path.resolve())
        payload["seed"] = int(payload["seed"])
        rows.append(payload)
    if not rows:
        raise FileNotFoundError(f"No spb_summary.json found below {root}")
    frame = pd.DataFrame(rows)
    frame["_mtime"] = frame.summary_path.map(lambda value: Path(value).stat().st_mtime)
    frame = (
        frame.sort_values("_mtime")
        .drop_duplicates(["model_key", "configuration", "seed"], keep="last")
        .drop(columns="_mtime")
    )
    return frame.sort_values(["model_key", "configuration", "seed"]).reset_index(drop=True)


def expected_inventory(manifest: Path, atmos_seeds: Sequence[int], multi_seeds: Sequence[int]) -> pd.DataFrame:
    rows = []
    table = pd.read_csv(manifest, dtype=str).fillna("")
    for _, config in table.iterrows():
        if config.atmosformer == "1":
            rows.extend(
                {"model_key": "atmosformer", "configuration": config.config_id, "seed": seed}
                for seed in atmos_seeds
            )
        if config.multi_arch == "1":
            for model_key in ("cnn", "convlstm", "geoformer"):
                rows.extend(
                    {"model_key": model_key, "configuration": config.config_id, "seed": seed}
                    for seed in multi_seeds
                )
    return pd.DataFrame(rows)


def load_lead_metrics(inventory: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for _, row in inventory.iterrows():
        summary_path = Path(str(row.summary_path))
        path = summary_path.parent / "spb_lead_metrics.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        frame.insert(0, "model_key", row.model_key)
        frame.insert(1, "configuration", row.configuration)
        frame.insert(2, "seed", int(row.seed))
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_stratified(inventory: pd.DataFrame, filename: str, level: str) -> pd.DataFrame:
    frames = []
    for _, row in inventory.iterrows():
        summary_path = Path(str(row.summary_path))
        path = summary_path.parent / filename
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        frame.insert(0, "model_key", row.model_key)
        frame.insert(1, "configuration", row.configuration)
        frame.insert(2, "seed", int(row.seed))
        frame.insert(3, "level", level)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def utility_from_summary(row: Mapping[str, object], metric: str) -> float:
    if metric == "acc_6_18":
        return float(row.get("nino34_mean_acc_6_18", np.nan))
    if metric == "negative_rmse_6_18":
        return -float(row.get("nino34_mean_rmse_6_18", np.nan))
    raise KeyError(metric)


def formula_value(values: Mapping[str, float], specification: Sequence[str]) -> float:
    if not all(name in values and np.isfinite(values[name]) for name in specification):
        return float("nan")
    if len(specification) == 2:
        return values[specification[0]] - values[specification[1]]
    if len(specification) == 4:
        return values[specification[0]] - values[specification[1]] - values[specification[2]] + values[specification[3]]
    raise ValueError(specification)


def build_effects(inventory: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    def emit(model_key: str, seed: int, level: str, coordinates: Mapping[str, object], metric: str, values: Mapping[str, float]) -> None:
        for name, spec in list(BASIN_FORMULAS.items()) + list(VARIABLE_FORMULAS.items()):
            value = formula_value(values, spec)
            if np.isfinite(value):
                rows.append({
                    "model_key": model_key,
                    "seed": int(seed),
                    "level": level,
                    **coordinates,
                    "metric": metric,
                    "component": name,
                    "value": float(value),
                    "utility_definition": "ACC" if metric.startswith("acc") else "negative RMSE",
                })
        single_names = ("global_slp", "global_sst", "global_slp_sst")
        if all(name in values and np.isfinite(values[name]) for name in single_names):
            rows.append({
                "model_key": model_key,
                "seed": int(seed),
                "level": level,
                **coordinates,
                "metric": metric,
                "component": "slp_sst_gain_over_best_single",
                "value": float(
                    values["global_slp_sst"]
                    - max(values["global_slp"], values["global_sst"])
                ),
                "utility_definition": "ACC" if metric.startswith("acc") else "negative RMSE",
            })

    for (model_key, seed), group in inventory.groupby(["model_key", "seed"], sort=False):
        values_by_config = {
            str(row.configuration): {
                metric: utility_from_summary(row, metric)
                for metric in ("acc_6_18", "negative_rmse_6_18")
            }
            for _, row in group.iterrows()
        }
        for metric in ("acc_6_18", "negative_rmse_6_18"):
            emit(model_key, int(seed), "overall", {}, metric, {
                config: vals[metric] for config, vals in values_by_config.items()
            })

    lead = load_lead_metrics(inventory)
    if not lead.empty:
        for (model_key, seed, lead_number), group in lead.groupby(["model_key", "seed", "lead"], sort=False):
            for metric, column in (("acc_3ma", "acc_3ma"), ("negative_rmse", "rmse")):
                values = {
                    str(row.configuration): (float(row[column]) if metric == "acc_3ma" else -float(row[column]))
                    for _, row in group.iterrows()
                }
                emit(model_key, int(seed), "lead", {"lead": int(lead_number)}, metric, values)

    for filename, level, coordinate in (
        ("spb_init_month_lead.csv", "init_month", "init_month"),
        ("spb_target_month_lead.csv", "target_month", "target_month"),
    ):
        table = load_stratified(inventory, filename, level)
        if table.empty:
            continue
        for (model_key, seed, coordinate_value, lead_number), group in table.groupby(
            ["model_key", "seed", coordinate, "lead"], sort=False
        ):
            for metric, column in (("acc", "acc"), ("negative_rmse", "rmse")):
                values = {
                    str(row.configuration): (float(row[column]) if metric == "acc" else -float(row[column]))
                    for _, row in group.iterrows()
                }
                emit(
                    model_key,
                    int(seed),
                    level,
                    {coordinate: int(coordinate_value), "lead": int(lead_number)},
                    metric,
                    values,
                )
    return pd.DataFrame(rows)


def write_summary(inventory: pd.DataFrame, effects: pd.DataFrame, expected: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    inventory.to_csv(output_dir / "run_inventory.csv", index=False)
    missing = expected.merge(
        inventory[["model_key", "configuration", "seed"]],
        on=["model_key", "configuration", "seed"],
        how="left",
        indicator=True,
    )
    missing = missing[missing["_merge"] == "left_only"].drop(columns="_merge")
    missing.to_csv(output_dir / "missing_runs.csv", index=False)
    if not effects.empty:
        effects.to_csv(output_dir / "core_effects_by_seed.csv", index=False)
        aggregate = effects.groupby(
            ["model_key", "level", "metric", "component"]
            + [column for column in ("lead", "init_month", "target_month") if column in effects.columns],
            dropna=False,
            as_index=False,
        ).agg(
            seed_count=("seed", "nunique"),
            mean=("value", "mean"),
            std=("value", "std"),
            positive_fraction=("value", lambda values: float((values > 0).mean())),
        )
        aggregate.to_csv(output_dir / "core_effects_summary.csv", index=False)
    protocol = {
        "claim": "Across the SPB, useful forecast information shifts from local Pacific state memory toward distributed cross-basin surface SLP/SST/TAUU precursors.",
        "training_tasks_expected": int(len(expected)),
        "completed_tasks": int(len(inventory)),
        "missing_tasks": int(len(missing)),
        "effect_scope": "paired input-space forecast utility; not dynamical causality",
        "utility": {"ACC": "direct", "RMSE": "sign reversed"},
        "basin_formulas": {
            "Indian": "U(P+I)-U(P)",
            "Atlantic": "U(P+A)-U(P)",
            "interaction": "U(P+I+A)-U(P+I)-U(P+A)+U(P)",
            "total_remote": "U(P+I+A)-U(P)",
            "global_context": "U(Global)-U(P+I+A)",
        },
        "variable_formulas": {
            "SLP": "U(SLP)-U(state)",
            "SST": "U(SST)-U(state)",
            "SLP_SST_complementarity": "U(SLP+SST)-U(SLP)-U(SST)+U(state)",
            "SLP_SST_gain_over_best_single": "U(SLP+SST)-max(U(SLP),U(SST))",
            "TAUU_increment": "U(SLP+SST+TAUU)-U(SLP+SST)",
        },
    }
    (output_dir / "core_claim_protocol.json").write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(protocol, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training_results_dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--atmos_seeds", default="2025,2026,2027,2028,2029")
    parser.add_argument("--multi_seeds", default="2025,2026,2027,2028,2029")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    inventory = discover(args.training_results_dir)
    expected = expected_inventory(args.manifest, parse_seeds(args.atmos_seeds), parse_seeds(args.multi_seeds))
    effects = build_effects(inventory)
    write_summary(inventory, effects, expected, args.output_dir)
    missing_count = len(expected.merge(inventory[["model_key", "configuration", "seed"]], on=["model_key", "configuration", "seed"], how="left", indicator=True).query("_merge == 'left_only'"))
    if args.strict and missing_count:
        raise SystemExit(f"Missing {missing_count} expected completed runs")


if __name__ == "__main__":
    main()
