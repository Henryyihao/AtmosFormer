"""Export title-free Nature-style evidence figures for the SPB core claim.

The figures are quantitative panels, not presentation slides: panel letters,
axis labels, legends and event labels are retained, while ``set_title`` and
``suptitle`` are intentionally never used.  All plotting and visual export is
performed with Python/matplotlib so that the script runs in the existing ENSO
environment without an R dependency.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


PROJECT_ROOT = Path(__file__).resolve().parents[1]

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.65,
        "axes.labelsize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.frameon": False,
        "legend.fontsize": 5.6,
        "lines.solid_capstyle": "round",
        "lines.dash_capstyle": "round",
    }
)

COLORS = {
    "state_only": "#7C858C",
    "global_slp": "#35658F",
    "global_sst": "#C9854A",
    "global_slp_sst": "#3F8F8B",
    "global_slp_sst_tauu": "#80638C",
    "basin_p": "#7C858C",
    "basin_p_i": "#B48B3C",
    "basin_p_a": "#C06B45",
    "basin_p_i_a": "#35658F",
    "full": "#35658F",
    "variable_pair_slp_sst": "#3F8F8B",
    "basin_tp": "#B48B3C",
    "basin_pair_tp_atlantic": "#C06B45",
    "basin_triplet_tp_indian_atlantic": "#80638C",
}

LABELS = {
    "state_only": "Index-only baseline",
    "global_slp": "SLP",
    "global_sst": "SST",
    "global_slp_sst": "SLP + SST",
    "global_slp_sst_tauu": "SLP + SST + TAUU",
    "basin_p": "Pacific",
    "basin_p_i": "Pacific + Indian",
    "basin_p_a": "Pacific + Atlantic",
    "basin_p_i_a": "Pacific + Indian + Atlantic",
    "full": "All retained",
    "variable_pair_slp_sst": "SLP + SST",
    "basin_tp": "Tropical Pacific",
    "basin_pair_tp_atlantic": "Pacific + Atlantic",
    "basin_triplet_tp_indian_atlantic": "Pacific + Indian + Atlantic",
}

MODEL_LABELS = {
    "atmosformer": "AtmosFormer",
    "cnn": "CNN",
    "convlstm": "ConvLSTM",
    "geoformer": "Geoformer",
}

MONTH_LABELS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")

SPB_PANEL_SPECS = (
    ("state_only", LABELS["state_only"], "Maps masked; Nino3.4, WWV and thermocline-tilt histories retained"),
    ("global_slp", "Global SLP", "SLP over the complete input grid"),
    ("global_sst", "Global SST", "SST over the complete input grid"),
    ("global_slp_sst", "Global SLP + SST", "SLP and SST over the complete input grid"),
    ("basin_p", "Pacific coalition", "Pacific SLP, SST and TAUU"),
    ("basin_p_i", "Pacific + Indian", "Pacific and Indian SLP, SST and TAUU"),
    ("basin_p_a", "Pacific + Atlantic", "Pacific and Atlantic SLP, SST and TAUU"),
    ("basin_p_i_a", "Five-basin coalition", "Pacific, Indian and Atlantic basin-union SLP, SST and TAUU"),
    ("global_slp_sst_tauu", "Global surface triple", "SLP, SST and TAUU over the complete input grid"),
)


def save_publication(fig: plt.Figure, output_base: Path, dpi: int = 600) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(output_base.with_suffix(".tiff"), dpi=int(dpi), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.16, 1.04, f"({label})", transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom")


def configure_axis(ax: plt.Axes) -> None:
    ax.tick_params(length=2.5, width=0.55, pad=2)
    ax.grid(axis="y", color="#D9DEE2", lw=0.35, alpha=0.65)
    ax.axhline(0, color="#555B60", lw=0.55, zorder=0)


def write_source(frame: pd.DataFrame, output_dir: Path, name: str) -> None:
    if not frame.empty:
        frame.to_csv(output_dir / f"source_{name}.csv", index=False)


def discover_runs(training_root: Path) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    for summary_path in sorted(training_root.glob("**/spb_summary.json")):
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        required = {"model_key", "configuration", "seed"}
        if not required.issubset(payload):
            continue
        records.append(
            {
                "model_key": str(payload["model_key"]),
                "configuration": str(payload["configuration"]),
                "seed": int(payload["seed"]),
                "result_dir": summary_path.parent,
                "summary": payload,
            }
        )
    newest: Dict[Tuple[str, str, int], Dict[str, object]] = {}
    for record in records:
        key = (str(record["model_key"]), str(record["configuration"]), int(record["seed"]))
        path = Path(record["result_dir"]) / "spb_summary.json"
        if key not in newest or path.stat().st_mtime > Path(newest[key]["result_dir"]).stat().st_mtime:
            newest[key] = record
    return sorted(newest.values(), key=lambda row: (str(row["model_key"]), str(row["configuration"]), int(row["seed"])))


def load_run_tables(records: Sequence[Mapping[str, object]], filename: str) -> pd.DataFrame:
    frames = []
    for record in records:
        path = Path(record["result_dir"]) / filename
        if not path.is_file():
            continue
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        frame.insert(0, "model_key", str(record["model_key"]))
        frame.insert(1, "configuration", str(record["configuration"]))
        frame.insert(2, "seed", int(record["seed"]))
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def effect_table(summary_dir: Path) -> pd.DataFrame:
    path = summary_dir / "core_effects_by_seed.csv"
    if not path.is_file():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    if "value" not in frame:
        return pd.DataFrame()
    frame = frame[np.isfinite(frame.value)]
    return frame


def mean_curve(
    lead: pd.DataFrame,
    model_key: str,
    configuration: str,
    value: str = "acc_3ma",
) -> pd.DataFrame:
    frame = lead.query("model_key == @model_key and configuration == @configuration").copy()
    if frame.empty or value not in frame:
        return pd.DataFrame(columns=["lead", "mean", "std", "n"])
    return frame.groupby("lead", as_index=False)[value].agg(mean="mean", std="std", n="count")


def plot_curve(ax: plt.Axes, curve: pd.DataFrame, label: str, color: str, ls: str = "-") -> None:
    if curve.empty:
        return
    x = curve.lead.to_numpy(float)
    y = curve["mean"].to_numpy(float)
    s = curve["std"].fillna(0.0).to_numpy(float)
    ax.plot(x, y, color=color, lw=1.35, ls=ls, label=label)
    ax.fill_between(x, y - s, y + s, color=color, alpha=0.12, linewidth=0)


def effect_curve(effects: pd.DataFrame, model_key: str, metric: str, component: str) -> pd.DataFrame:
    frame = effects.query(
        "model_key == @model_key and level == 'lead' and metric == @metric and component == @component"
    ).copy()
    if frame.empty:
        return pd.DataFrame(columns=["lead", "mean", "std", "n"])
    return frame.groupby("lead", as_index=False).value.agg(mean="mean", std="std", n="count")


def overall_effect(effects: pd.DataFrame, model_key: str, metric: str, components: Sequence[str]) -> pd.DataFrame:
    frame = effects.query(
        "model_key == @model_key and level == 'overall' and metric == @metric and component in @components"
    ).copy()
    if frame.empty:
        return pd.DataFrame(columns=["component", "mean", "std", "n"])
    return frame.groupby("component", as_index=False).value.agg(mean="mean", std="std", n="count")


def draw_effect_lines(
    ax: plt.Axes,
    effects: pd.DataFrame,
    components: Sequence[str],
    labels: Mapping[str, str],
    colors: Mapping[str, str],
) -> None:
    for component in components:
        curve = effect_curve(effects, "atmosformer", "acc_3ma", component)
        plot_curve(ax, curve, labels.get(component, component), colors.get(component, "#555B60"))
    ax.set_xlim(1, 24)
    ax.set_xlabel("Lead (months)")
    ax.set_ylabel("Delta ACC3MA")
    configure_axis(ax)


def figure1(lead: pd.DataFrame, runs: Sequence[Mapping[str, object]], output_dir: Path, dpi: int) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(183 / 25.4, 72 / 25.4), constrained_layout=True)
    ax = axes[0]
    variable_configs = ["state_only", "global_slp", "global_sst", "global_slp_sst"]
    for config in variable_configs:
        plot_curve(ax, mean_curve(lead, "atmosformer", config), LABELS[config], COLORS[config])
    ax.axhline(0.5, color="#555B60", lw=0.55, ls=":")
    ax.set_xlim(1, 24)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("Lead (months)")
    ax.set_ylabel("ACC3MA")
    configure_axis(ax)
    ax.legend(loc="lower left", fontsize=5.2)
    panel_label(ax, "a")

    ax = axes[1]
    basin_configs = ["basin_p", "basin_p_i", "basin_p_a", "basin_p_i_a"]
    for config in basin_configs:
        plot_curve(ax, mean_curve(lead, "atmosformer", config), LABELS[config], COLORS[config])
    ax.axhline(0.5, color="#555B60", lw=0.55, ls=":")
    ax.set_xlim(1, 24)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("Lead (months)")
    ax.set_ylabel("ACC3MA")
    configure_axis(ax)
    ax.legend(loc="lower left", fontsize=5.0)
    panel_label(ax, "b")

    ax = axes[2]
    selected = ["state_only", "global_slp_sst", "basin_p_a", "basin_p_i_a"]
    rows = []
    for config in selected:
        values = [
            float(row["summary"].get("effective_lead_3ma_acc_05", np.nan))
            for row in runs
            if row["model_key"] == "atmosformer" and row["configuration"] == config
        ]
        values = np.asarray(values, dtype=float)
        if np.isfinite(values).any():
            rows.append((config, np.nanmean(values), np.nanstd(values, ddof=1) if np.sum(np.isfinite(values)) > 1 else 0.0))
    if rows:
        x = np.arange(len(rows))
        ax.bar(x, [row[1] for row in rows], yerr=[row[2] for row in rows], color=[COLORS[row[0]] for row in rows], width=0.68, capsize=2, alpha=0.9)
        ax.set_xticks(x, [LABELS[row[0]].replace(" + ", "\n+").replace(" baseline", "\nbaseline") for row in rows], rotation=0)
        ax.set_ylim(0, 24.5)
        ax.set_ylabel("Effective lead (months)")
    else:
        ax.text(0.5, 0.5, "data unavailable", ha="center", va="center", transform=ax.transAxes)
    configure_axis(ax)
    panel_label(ax, "c")
    save_publication(fig, output_dir / "fig1_long_lead_skill", dpi)


def figure2(effects: pd.DataFrame, output_dir: Path, dpi: int) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(183 / 25.4, 112 / 25.4), constrained_layout=True)
    variable_components = ["slp_gain_over_state", "sst_gain_over_state", "slp_sst_gain_over_best_single", "tauu_increment"]
    variable_labels = {
        "slp_gain_over_state": "SLP",
        "sst_gain_over_state": "SST",
        "slp_sst_gain_over_best_single": "SLP + SST over best single",
        "tauu_increment": "Zonal wind-stress increment",
    }
    variable_tick_labels = {
        "slp_gain_over_state": "SLP",
        "sst_gain_over_state": "SST",
        "slp_sst_gain_over_best_single": "SLP + SST\nbest single",
        "tauu_increment": "Zonal wind stress\nincrement",
    }
    variable_colors = {
        "slp_gain_over_state": COLORS["global_slp"],
        "sst_gain_over_state": COLORS["global_sst"],
        "slp_sst_gain_over_best_single": COLORS["global_slp_sst"],
        "tauu_increment": COLORS["global_slp_sst_tauu"],
    }

    ax = axes[0, 0]
    draw_effect_lines(ax, effects, variable_components, variable_labels, variable_colors)
    ax.set_ylim(-0.18, 0.22)
    ax.legend(loc="upper right", fontsize=5.0)
    panel_label(ax, "a")

    ax = axes[0, 1]
    rows = []
    for component in variable_components:
        curve = effect_curve(effects, "atmosformer", "acc_3ma", component)
        if curve.empty:
            continue
        rows.append((component, curve.set_index("lead")["mean"].reindex(range(1, 25)).to_numpy(float)))
    if rows:
        data = np.vstack([row[1] for row in rows])
        finite = np.abs(data[np.isfinite(data)])
        limit = max(float(np.nanquantile(finite, 0.98)) if finite.size else 0.05, 0.05)
        norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
        image = ax.imshow(data, aspect="auto", cmap="RdBu_r", norm=norm, interpolation="nearest", extent=(0.5, 24.5, len(rows) + 0.5, 0.5))
        ax.set_yticks(np.arange(1, len(rows) + 1), [variable_labels[row[0]] for row in rows])
        ax.set_xticks([1, 6, 12, 18, 24])
        ax.set_xlabel("Lead (months)")
        cb = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("Delta ACC3MA")
    else:
        ax.text(0.5, 0.5, "data unavailable", ha="center", va="center", transform=ax.transAxes)
    ax.grid(False)
    panel_label(ax, "b")

    ax = axes[1, 0]
    overall = overall_effect(effects, "atmosformer", "acc_6_18", variable_components)
    if not overall.empty:
        order = [component for component in variable_components if component in set(overall.component)]
        values = overall.set_index("component").reindex(order)
        x = np.arange(len(order))
        ax.bar(x, values["mean"], yerr=values["std"].fillna(0), color=[variable_colors[item] for item in order], capsize=2, width=0.67, alpha=0.9)
        ax.set_xticks(x, [variable_tick_labels[item] for item in order], rotation=0)
        ax.set_ylabel("Delta ACC (6-18 months)")
    else:
        ax.text(0.5, 0.5, "data unavailable", ha="center", va="center", transform=ax.transAxes)
    configure_axis(ax)
    panel_label(ax, "c")

    ax = axes[1, 1]
    seed_frame = effects.query("model_key == 'atmosformer' and level == 'overall' and metric == 'acc_6_18' and component in @variable_components").copy()
    if not seed_frame.empty:
        order = [component for component in variable_components if component in set(seed_frame.component)]
        x = np.arange(len(order))
        for index, component in enumerate(order):
            values = seed_frame.loc[seed_frame.component == component, "value"].to_numpy(float)
            values = values[np.isfinite(values)]
            jitter = np.linspace(-0.08, 0.08, max(len(values), 1))
            ax.scatter(np.full(len(values), index) + jitter[: len(values)], values, s=12, color=variable_colors[component], edgecolor="white", linewidth=0.25, zorder=2)
            if values.size:
                ax.errorbar(index, values.mean(), yerr=values.std(ddof=1) if values.size > 1 else 0.0, color="#33383C", fmt="o", ms=3.2, capsize=2, zorder=3)
        ax.set_xticks(x, [variable_tick_labels[item] for item in order])
        ax.set_ylabel("Delta ACC (6-18 months)")
    else:
        ax.text(0.5, 0.5, "data unavailable", ha="center", va="center", transform=ax.transAxes)
    configure_axis(ax)
    panel_label(ax, "d")
    write_source(seed_frame, output_dir, "variable_effects")
    save_publication(fig, output_dir / "fig2_variable_coalition", dpi)


def figure3(effects: pd.DataFrame, output_dir: Path, dpi: int) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(183 / 25.4, 112 / 25.4), constrained_layout=True)
    basin_components = ["indian", "atlantic", "total_remote", "interaction"]
    basin_labels = {
        "indian": "Indian increment",
        "atlantic": "Atlantic increment",
        "total_remote": "Total remote-basin increment",
        "interaction": "Indian-Atlantic interaction",
    }
    basin_colors = {"indian": "#B48B3C", "atlantic": "#C06B45", "total_remote": "#35658F", "interaction": "#80638C"}

    ax = axes[0, 0]
    draw_effect_lines(ax, effects, basin_components, basin_labels, basin_colors)
    ax.set_ylim(-0.14, 0.18)
    ax.legend(loc="upper right", fontsize=5.2)
    panel_label(ax, "a")

    ax = axes[0, 1]
    rows = []
    for component in basin_components:
        curve = effect_curve(effects, "atmosformer", "acc_3ma", component)
        if curve.empty:
            continue
        rows.append((component, curve.set_index("lead")["mean"].reindex(range(1, 25)).to_numpy(float)))
    if rows:
        data = np.vstack([row[1] for row in rows])
        finite = np.abs(data[np.isfinite(data)])
        limit = max(float(np.nanquantile(finite, 0.98)) if finite.size else 0.05, 0.05)
        image = ax.imshow(data, aspect="auto", cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit), interpolation="nearest", extent=(0.5, 24.5, len(rows) + 0.5, 0.5))
        ax.set_yticks(np.arange(1, len(rows) + 1), [basin_labels[row[0]] for row in rows])
        ax.set_xticks([1, 6, 12, 18, 24])
        ax.set_xlabel("Lead (months)")
        cb = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("Delta ACC3MA")
    else:
        ax.text(0.5, 0.5, "data unavailable", ha="center", va="center", transform=ax.transAxes)
    ax.grid(False)
    panel_label(ax, "b")

    ax = axes[1, 0]
    overall = overall_effect(effects, "atmosformer", "acc_6_18", basin_components)
    if not overall.empty:
        order = [component for component in basin_components if component in set(overall.component)]
        values = overall.set_index("component").reindex(order)
        x = np.arange(len(order))
        ax.bar(x, values["mean"], yerr=values["std"].fillna(0), color=[basin_colors[item] for item in order], capsize=2, width=0.67, alpha=0.9)
        ax.set_xticks(x, [basin_labels[item].replace(" increment", "\ninc.").replace(" over P", "\nover P") for item in order], rotation=25, ha="right")
        ax.set_ylabel("Delta ACC (6-18 months)")
    else:
        ax.text(0.5, 0.5, "data unavailable", ha="center", va="center", transform=ax.transAxes)
    configure_axis(ax)
    panel_label(ax, "c")

    ax = axes[1, 1]
    rows = []
    for component in ["atlantic", "total_remote"]:
        values = effects.query("model_key == 'atmosformer' and level == 'overall' and metric == 'acc_6_18' and component == @component").value.to_numpy(float)
        values = values[np.isfinite(values)]
        if values.size:
            rows.append((component, values.mean(), values.std(ddof=1) if values.size > 1 else 0.0))
    if rows:
        x = np.arange(len(rows))
        ax.bar(x, [row[1] for row in rows], yerr=[row[2] for row in rows], color=[basin_colors[row[0]] for row in rows], width=0.6, capsize=2)
        ax.set_xticks(x, [basin_labels[row[0]].replace(" increment", "\nincrement") for row in rows])
        ax.set_ylabel("Delta ACC (6-18 months)")
    else:
        ax.text(0.5, 0.5, "data unavailable", ha="center", va="center", transform=ax.transAxes)
    configure_axis(ax)
    panel_label(ax, "d")
    write_source(overall, output_dir, "basin_effects")
    save_publication(fig, output_dir / "fig3_basin_coalition", dpi)


def architecture_effects(effects: pd.DataFrame, component: str) -> pd.DataFrame:
    frame = effects.query("level == 'overall' and metric == 'acc_6_18' and component == @component").copy()
    if frame.empty:
        return pd.DataFrame(columns=["model_key", "mean", "std", "n"])
    return frame.groupby("model_key", as_index=False).value.agg(mean="mean", std="std", n="count")


def figure4(effects: pd.DataFrame, output_dir: Path, dpi: int) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(183 / 25.4, 72 / 25.4), constrained_layout=True)
    panels = [
        ("atlantic", "Atlantic increment"),
        ("total_remote", "Total remote-basin increment"),
        ("slp_sst_gain_over_best_single", "SLP + SST over best single"),
    ]
    model_order = ["atmosformer", "cnn", "convlstm", "geoformer"]
    colors = {
        "atmosformer": "#35658F",
        "cnn": "#C9854A",
        "convlstm": "#80638C",
        "geoformer": "#3F8F8B",
    }
    for ax, (component, ylabel) in zip(axes, panels):
        frame = architecture_effects(effects, component)
        if not frame.empty:
            frame["model_key"] = pd.Categorical(
                frame.model_key, model_order, ordered=True
            )
            frame = frame.sort_values("model_key")
            x = np.arange(len(frame))
            ax.bar(x, frame["mean"], yerr=frame["std"].fillna(0), color=[colors.get(str(value), "#7C858C") for value in frame.model_key], width=0.62, capsize=2)
            ax.set_xticks(x, [MODEL_LABELS.get(str(value), str(value)) for value in frame.model_key])
            ax.set_ylabel("Delta ACC (6-18 months)")
        else:
            ax.text(0.5, 0.5, "data unavailable", ha="center", va="center", transform=ax.transAxes)
        configure_axis(ax)
    panel_label(axes[0], "a")
    panel_label(axes[1], "b")
    panel_label(axes[2], "c")
    save_publication(fig, output_dir / "fig4_cross_architecture", dpi)


def figure5(crossing: pd.DataFrame, runs: Sequence[Mapping[str, object]], output_dir: Path, dpi: int) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(183 / 25.4, 72 / 25.4), constrained_layout=True)
    configs = ["state_only", "global_slp_sst", "basin_p_a", "basin_p_i_a"]
    ax = axes[0]
    for config in configs:
        frame = crossing.query("model_key == 'atmosformer' and configuration == @config and lead <= 9").copy()
        frame = frame[(frame.n_not_crossed >= 3) & (frame.n_crossed >= 3)]
        if frame.empty:
            continue
        curve = frame.groupby("lead", as_index=False).acc_penalty_crossed.agg(mean="mean", std="std")
        plot_curve(ax, curve, LABELS[config], COLORS[config])
    ax.set_xlabel("Lead (months)")
    ax.set_ylabel("Raw-ACC penalty\n(not crossed - crossed)")
    ax.set_xlim(1, 9)
    configure_axis(ax)
    ax.legend(fontsize=5.0, loc="lower right", bbox_to_anchor=(1.0, 0.06))
    panel_label(ax, "a")

    ax = axes[1]
    for config in ["state_only", "global_slp_sst", "basin_p_a"]:
        frame = crossing.query("model_key == 'atmosformer' and configuration == @config and lead <= 9").copy()
        frame = frame[(frame.n_not_crossed >= 3) & (frame.n_crossed >= 3)]
        if frame.empty:
            continue
        for column, color, label, ls in [
            ("acc_not_crossed", "#33383C", "not crossed", "-"),
            ("acc_crossed", COLORS[config], "crossed", "--"),
        ]:
            curve = frame.groupby("lead", as_index=False)[column].agg(mean="mean", std="std")
            plot_curve(ax, curve, f"{LABELS[config]}: {label}", color, ls=ls)
    ax.set_xlabel("Lead (months)")
    ax.set_ylabel("ACC")
    ax.set_xlim(1, 9)
    ax.set_ylim(0.4, 1.02)
    configure_axis(ax)
    ax.legend(fontsize=4.7, ncol=1, loc="lower left")
    panel_label(ax, "b")

    ax = axes[2]
    rows = []
    for config in configs:
        values = [
            float(record["summary"].get("spb_mean_acc_penalty", np.nan))
            for record in runs
            if record["model_key"] == "atmosformer" and record["configuration"] == config
        ]
        values = np.asarray(values, float)
        if np.isfinite(values).any():
            rows.append((config, np.nanmean(values), np.nanstd(values, ddof=1) if np.sum(np.isfinite(values)) > 1 else 0.0))
    if rows:
        x = np.arange(len(rows))
        ax.bar(x, [row[1] for row in rows], yerr=[row[2] for row in rows], color=[COLORS[row[0]] for row in rows], capsize=2, width=0.65)
        ax.set_xticks(x, [LABELS[row[0]].replace(" + ", "\n+").replace(" baseline", "\nbaseline") for row in rows])
        ax.set_ylabel("Mean raw-ACC penalty\n(not crossed - crossed)")
    else:
        ax.text(0.5, 0.5, "data unavailable", ha="center", va="center", transform=ax.transAxes)
    configure_axis(ax)
    panel_label(ax, "c")
    save_publication(fig, output_dir / "fig5_spb_crossing", dpi)


def event_tables(events_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    merged = events_dir / "event_hindcasts_all.csv"
    if merged.is_file():
        events = pd.read_csv(merged)
    else:
        frames = []
        for path in sorted(events_dir.glob("atmosformer_*/event_hindcasts_*.csv")):
            frame = pd.read_csv(path)
            if frame.empty:
                continue
            directory = path.parent.name
            if "seed" in directory:
                seed = int(directory.rsplit("seed", 1)[-1])
            else:
                seed = -1
            if "seed" not in frame.columns:
                frame.insert(0, "seed", seed)
            frames.append(frame)
        events = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    selected_path = events_dir / "selected_events.csv"
    selected = pd.read_csv(selected_path) if selected_path.is_file() else pd.DataFrame()
    additional_path = events_dir / "event_hindcasts_additional.csv"
    additional_selected_path = events_dir / "selected_additional_events.csv"
    if additional_path.is_file():
        if not additional_selected_path.is_file():
            raise FileNotFoundError("Additional hindcasts require selection metadata")
        additional = pd.read_csv(additional_path)
        if set(additional.event_year) & set(events.event_year):
            raise ValueError("Additional event years duplicate the original event list")
        events = pd.concat([events, additional], ignore_index=True)
        selected = pd.concat([selected, pd.read_csv(additional_selected_path)], ignore_index=True)
        selected = selected.sort_values("event_year").reset_index(drop=True)
    return events, selected


def figure6(events_dir: Path, output_dir: Path, dpi: int, max_events: int = 8) -> None:
    events, selected = event_tables(events_dir)
    if events.empty:
        return
    years = sorted(events.event_year.dropna().astype(int).unique())
    if not selected.empty and "event_year" in selected:
        selected = selected.drop_duplicates("event_year")
        order = [int(value) for value in selected.event_year if int(value) in years]
        years = order + [year for year in years if year not in order]
    years = years[: int(max_events)]
    write_source(events[events.event_year.isin(years)], output_dir, "fig6_event_hindcasts")
    if not selected.empty:
        write_source(selected[selected.event_year.isin(years)], output_dir, "fig6_selected_events")
    ncols = min(4, max(1, len(years)))
    nrows = int(math.ceil(len(years) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(183 / 25.4, (54 * nrows) / 25.4), squeeze=False, sharex=True)
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.11, top=0.85, wspace=0.34, hspace=0.30)
    masks = ["state_only", "variable_pair_slp_sst", "basin_tp", "basin_pair_tp_atlantic", "basin_triplet_tp_indian_atlantic", "full"]
    labels = dict(LABELS)
    labels.update(
        {
            "state_only": LABELS["state_only"],
            "variable_pair_slp_sst": "SLP + SST mask",
            "basin_tp": "Tropical Pacific mask",
            "basin_pair_tp_atlantic": "Pacific + Atlantic mask",
            "basin_triplet_tp_indian_atlantic": "Pacific + Indian + Atlantic mask",
            "full": "All active surface fields",
        }
    )
    for index, year in enumerate(years):
        ax = axes.flat[index]
        subset = events[events.event_year.astype(int) == int(year)]
        truth = subset.sort_values("lead").drop_duplicates("lead")
        if not truth.empty:
            ax.plot(truth.lead, truth.true_nino34, color="#202427", lw=1.25, label="Observed")
        for mask in masks:
            item = subset[subset.mask_name == mask]
            if item.empty:
                continue
            curve = item.groupby("lead", as_index=False).pred_nino34.agg(mean="mean", std="std")
            color = COLORS.get(mask, "#7C858C")
            ax.plot(curve.lead, curve["mean"], color=color, lw=1.05, label=labels.get(mask, mask))
            spread = curve["std"].fillna(0).to_numpy(float)
            ax.fill_between(curve.lead, curve["mean"] - spread, curve["mean"] + spread, color=color, alpha=0.08, linewidth=0)
        phase = ""
        if not selected.empty and "phase" in selected:
            row = selected[selected.event_year.astype(int) == int(year)]
            if not row.empty:
                phase = str(row.iloc[0].phase)
        if phase == "cold":
            label_position = (0.96, 0.96, "right", "top")
        else:
            label_position = (0.04, 0.04, "left", "bottom")
        x, y, ha, va = label_position
        ax.text(x, y, f"{int(year)} {phase}".strip(), transform=ax.transAxes, fontsize=7, fontweight="bold", ha=ha, va=va, bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.0})
        ax.axhline(0, color="#555B60", lw=0.45)
        ax.axvline(18, color="#7C858C", lw=0.45, ls=":")
        ax.set_xlim(1, 24)
        ax.margins(y=0.16 if phase == "cold" else 0.10)
        ax.set_xticks([1, 6, 12, 18, 24])
        if index % ncols == 0:
            ax.set_ylabel("Nino3.4 anomaly (°C)")
        if index // ncols == nrows - 1:
            ax.set_xlabel("Lead (months)")
        configure_axis(ax)
        panel_label(ax, chr(ord("a") + index))
    for index in range(len(years), nrows * ncols):
        axes.flat[index].set_visible(False)
    handles, legend_labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, legend_labels, loc="upper center", bbox_to_anchor=(0.5, 0.985), ncol=min(4, len(legend_labels)), fontsize=5.3, frameon=False)
    save_publication(fig, output_dir / "fig6_event_hindcasts", dpi)


def figure7(
    init_month: pd.DataFrame,
    output_dir: Path,
    dpi: int,
    max_lead: int = 24,
    seed: int | None = None,
) -> None:
    """Plot the 12 initialization-month by lead ENSO skill surfaces."""
    selected_configs = [row[0] for row in SPB_PANEL_SPECS]
    frame = init_month.query(
        "model_key == 'atmosformer' and configuration in @selected_configs and lead <= @max_lead"
    ).copy()
    if seed is not None:
        frame = frame[frame.seed == int(seed)].copy()
    if frame.empty:
        suffix = f" for seed {seed}" if seed is not None else ""
        raise ValueError(f"No AtmosFormer initialization-month data found{suffix}")

    if seed is None:
        aggregate = (
            frame.groupby(["configuration", "init_month", "lead"], as_index=False)
            .agg(
                acc_mean=("acc", "mean"),
                acc_std=("acc", "std"),
                seed_count=("acc", "count"),
                sample_count=("n", "sum"),
            )
        )
        source_name = "source_spb_init_month_lead_mean.csv"
        output_name = "fig7_spb_init_month_lead"
    else:
        aggregate = frame[
            ["seed", "configuration", "init_month", "lead", "acc", "rmse", "n"]
        ].rename(columns={"acc": "acc_mean", "n": "sample_count"})
        aggregate.insert(5, "seed_count", 1)
        source_name = f"source_spb_init_month_lead_seed{seed}.csv"
        output_name = f"fig7_spb_init_month_lead_seed{seed}"
    aggregate = aggregate.sort_values(["configuration", "init_month", "lead"]).reset_index(drop=True)
    grid_keys = ["configuration", "init_month", "lead"]
    expected_grid = pd.MultiIndex.from_product(
        [selected_configs, range(1, 13), range(1, int(max_lead) + 1)],
        names=grid_keys,
    )
    actual_grid = pd.MultiIndex.from_frame(aggregate[grid_keys])
    missing_cells = expected_grid.difference(actual_grid)
    unexpected_cells = actual_grid.difference(expected_grid)
    duplicate_cells = aggregate.duplicated(grid_keys, keep=False)
    if missing_cells.size or unexpected_cells.size or duplicate_cells.any():
        raise ValueError(
            "Invalid SPB grid: "
            f"missing={missing_cells.size}, unexpected={unexpected_cells.size}, "
            f"duplicate_rows={int(duplicate_cells.sum())}"
        )
    if not np.isfinite(aggregate["acc_mean"]).all():
        raise ValueError("The SPB grid contains non-finite ACC values")
    aggregate.to_csv(output_dir / source_name, index=False)
    panel_source_name = (
        f"source_spb_panel_definitions_seed{seed}.csv"
        if seed is not None
        else "source_spb_panel_definitions.csv"
    )
    pd.DataFrame(
        [
            {
                "panel": chr(ord("a") + index),
                "configuration": configuration,
                "display_label": label,
                "retained_input": description,
                "common_state_branch": "Nino3.4, WWV and thermocline-tilt histories",
            }
            for index, (configuration, label, description) in enumerate(SPB_PANEL_SPECS)
        ]
    ).to_csv(output_dir / panel_source_name, index=False)

    fig, axes = plt.subplots(
        3,
        3,
        figsize=(183 / 25.4, 145 / 25.4),
        squeeze=False,
    )
    fig.subplots_adjust(left=0.072, right=0.91, bottom=0.075, top=0.965, wspace=0.32, hspace=0.39)
    levels = np.linspace(0.0, 1.0, 11)
    norm = TwoSlopeNorm(vmin=0.0, vcenter=0.5, vmax=1.0)
    cmap = mpl.colormaps["RdBu_r"].copy()
    cmap.set_under("#2166AC")
    cmap.set_over("#8E1230")
    x = np.arange(1, int(max_lead) + 1)
    y = np.arange(1, 13)
    filled = None

    for index, (configuration, label, _) in enumerate(SPB_PANEL_SPECS):
        ax = axes.flat[index]
        panel = aggregate[aggregate.configuration == configuration]
        matrix = (
            panel.pivot(index="init_month", columns="lead", values="acc_mean")
            .reindex(index=y, columns=x)
            .to_numpy(dtype=float)
        )
        filled = ax.contourf(x, y, matrix, levels=levels, cmap=cmap, norm=norm, extend="both")
        contour = ax.contour(
            x,
            y,
            matrix,
            levels=[0.5, 0.7, 0.9],
            colors="#25292C",
            linewidths=[0.75, 0.45, 0.45],
        )
        ax.clabel(contour, fmt="%.1f", fontsize=4.5, inline=True, inline_spacing=1)
        ax.set_xlim(1, int(max_lead))
        ax.set_ylim(1, 12)
        ax.set_xticks([value for value in (3, 6, 9, 12, 15, 18, 21, 24) if value <= int(max_lead)])
        ax.set_yticks(y, MONTH_LABELS)
        ax.set_xlabel("Lead (months)")
        if index % 3 == 0:
            ax.set_ylabel("Initialization month")
        ax.tick_params(direction="in", top=False, right=False, length=2.2, width=0.55, pad=1.8)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.65)
            spine.set_color("#25292C")
        ax.text(-0.14, 1.035, f"({chr(ord('a') + index)})", transform=ax.transAxes, fontsize=8.2, fontweight="bold", va="bottom")
        ax.text(0.02, 1.035, label, transform=ax.transAxes, fontsize=6.4, fontweight="bold", va="bottom")

    if filled is not None:
        color_axis = fig.add_axes([0.935, 0.18, 0.014, 0.67])
        colorbar = fig.colorbar(filled, cax=color_axis, ticks=np.arange(0.0, 1.01, 0.1), extend="both")
        colorbar.set_label("Nino3.4 ACC", fontsize=7)
        colorbar.ax.tick_params(labelsize=5.6, length=2.0, width=0.5)
    save_publication(fig, output_dir / output_name, dpi)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary_dir", type=Path, default=PROJECT_ROOT / "results_spb_core_claim" / "summary")
    parser.add_argument("--training_results_dir", type=Path, default=PROJECT_ROOT / "results_spb_core_claim_training")
    parser.add_argument("--events_dir", type=Path, default=PROJECT_ROOT / "results_spb_core_claim" / "event_backtest")
    parser.add_argument("--output_dir", type=Path, default=PROJECT_ROOT / "results_spb_core_claim" / "figures_nature")
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--max_events", type=int, default=8)
    parser.add_argument("--figures", type=int, nargs="+", choices=range(1, 8), default=list(range(1, 8)), help="Figure numbers to export")
    parser.add_argument(
        "--spb_seed",
        type=int,
        default=None,
        help="Use one AtmosFormer seed for Fig. 7 instead of averaging all available seeds",
    )
    parser.add_argument(
        "--only_spb",
        action="store_true",
        help="Export only the initialization-month by lead SPB figure",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    runs = discover_runs(args.training_results_dir)
    if not runs:
        raise SystemExit(f"No spb_summary.json found below {args.training_results_dir}")
    init_month = load_run_tables(runs, "spb_init_month_lead.csv")
    if args.only_spb:
        figure7(init_month, args.output_dir, args.dpi, seed=args.spb_seed)
        print(f"[SPB figures] wrote the title-free SPB figure to {args.output_dir}")
        return

    lead = load_run_tables(runs, "spb_lead_metrics.csv")
    crossing = load_run_tables(runs, "spb_crossing_contrast.csv")
    effects = effect_table(args.summary_dir)
    if lead.empty:
        raise SystemExit("No spb_lead_metrics.csv was found in the training results")
    write_source(lead, args.output_dir, "lead_metrics")
    write_source(crossing, args.output_dir, "crossing_metrics")
    write_source(init_month, args.output_dir, "spb_init_month_lead")
    write_source(effects, args.output_dir, "core_effects")
    for filename in ("core_effects_by_seed.csv", "core_effects_summary.csv", "run_inventory.csv", "core_claim_protocol.json"):
        path = args.summary_dir / filename
        if path.is_file():
            shutil.copy2(path, args.output_dir / filename)

    if 1 in args.figures:
        figure1(lead, runs, args.output_dir, args.dpi)
    if 2 in args.figures:
        figure2(effects, args.output_dir, args.dpi)
    if 3 in args.figures:
        figure3(effects, args.output_dir, args.dpi)
    if 4 in args.figures:
        figure4(effects, args.output_dir, args.dpi)
    if 5 in args.figures:
        figure5(crossing, runs, args.output_dir, args.dpi)
    if 6 in args.figures:
        figure6(args.events_dir, args.output_dir, args.dpi, args.max_events)
    if 7 in args.figures:
        figure7(init_month, args.output_dir, args.dpi, seed=args.spb_seed)
    print(f"[SPB figures] wrote title-free Nature-style figures to {args.output_dir}")


if __name__ == "__main__":
    main()
