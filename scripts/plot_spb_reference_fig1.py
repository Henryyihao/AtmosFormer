"""Create a title-free reference-style Figure 1 for SPB forecast skill.

The figure compares the frozen global SLP+SST+TAUU configuration of the three
deep-learning architectures with the supplied NMME member ACC curves. Deep
learning curves are solid and show five-seed mean +/- sample SD; NMME members
are dashed and retain their individual curves. The comparison windows are
kept explicit because the supplied NMME file ends at lead 11.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_ORDER = ("atmosformer", "cnn", "geoformer")
MODEL_LABELS = {
    "atmosformer": "AtmosFormer",
    "cnn": "CNN",
    "geoformer": "Geoformer",
}
MODEL_COLORS = {
    "atmosformer": "#D1495B",
    "cnn": "#35658F",
    "geoformer": "#2A8C82",
}
NMME_COLORS = (
    "#6C757D",
    "#8C6D5A",
    "#7967A8",
    "#A66A8F",
    "#4C8C8A",
    "#7D8FB2",
    "#B07A4C",
    "#5E7895",
)
MONTH_LABELS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
DL_PERIOD = "1980-2025"
NMME_PERIOD = "1981-2021"
CAPTION_ZH = (
    "图1 | 不同预测系统的ENSO指数预测技巧与春季可预报性。"
    "(a) Niño3.4指数随预测提前期变化的ACC。实线表示深度学习模型，"
    "其结果为五个随机种子的平均值，阴影表示种子间的样本标准差；"
    "虚线表示NMME各成员，AtmosFormer使用醒目的红色突出显示。"
    "三种深度学习模型均采用全球SLP+SST+TAUU输入配置。"
    "深度学习模型的测试时段为1980-2025，NMME成员的资料时段为1981-2021。"
    "所提供的NMME曲线覆盖1-11个月预测提前期。"
    "(b) 在两类资料均有覆盖的5-11个月提前期内，对ACC曲线进行放大比较；"
    "NMME灰色带表示成员范围，黑色虚线表示成员平均值。"
    "(c) AtmosFormer seed2025在不同初始化月份下的ACC填色等值线图，"
    "黑色等值线分别表示ACC=0.5、0.7和0.9。"
    "(d) ACC达到0.5所对应的有效提前期；深度学习模型显示五个种子分布，"
    "NMME显示各成员的有效提前期。"
)

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
        "legend.frameon": True,
        "legend.framealpha": 0.92,
        "legend.edgecolor": "#D6DADD",
        "legend.fontsize": 5.0,
        "lines.solid_capstyle": "round",
        "lines.dash_capstyle": "round",
    }
)


def save_publication(fig: plt.Figure, output_base: Path, dpi: int = 600) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(output_base.with_suffix(".tiff"), dpi=int(dpi), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.15,
        1.04,
        f"({label})",
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        va="bottom",
    )


def configure_axis(ax: plt.Axes, acc_reference: bool = True) -> None:
    ax.tick_params(length=2.5, width=0.55, pad=2)
    ax.grid(axis="y", color="#D9DEE2", lw=0.35, alpha=0.65)
    if acc_reference:
        ax.axhline(0.5, color="#4A5055", lw=0.55, ls=(0, (3, 2)), zorder=0)


def discover_records(training_root: Path, configuration: str) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    for summary_path in sorted(training_root.glob("**/spb_summary.json")):
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        model_key = str(payload.get("model_key", ""))
        if model_key not in MODEL_ORDER or str(payload.get("configuration", "")) != configuration:
            continue
        if "seed" not in payload:
            continue
        records.append(
            {
                "model_key": model_key,
                "seed": int(payload["seed"]),
                "result_dir": summary_path.parent,
                "summary": payload,
                "mtime": summary_path.stat().st_mtime,
            }
        )

    newest: Dict[Tuple[str, int], Dict[str, object]] = {}
    for record in records:
        key = (str(record["model_key"]), int(record["seed"]))
        if key not in newest or float(record["mtime"]) > float(newest[key]["mtime"]):
            newest[key] = record
    return sorted(
        newest.values(),
        key=lambda row: (MODEL_ORDER.index(str(row["model_key"])), int(row["seed"])),
    )


def load_deep_leads(records: Sequence[Mapping[str, object]]) -> pd.DataFrame:
    frames = []
    for record in records:
        path = Path(str(record["result_dir"])) / "spb_lead_metrics.csv"
        if not path.is_file():
            continue
        frame = pd.read_csv(path)
        if not {"lead", "acc"}.issubset(frame.columns):
            continue
        frame = frame[["lead", "acc"]].copy()
        frame.insert(0, "model_key", str(record["model_key"]))
        frame.insert(1, "seed", int(record["seed"]))
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["model_key", "seed", "lead", "acc"])
    result = pd.concat(frames, ignore_index=True)
    result["lead"] = pd.to_numeric(result["lead"], errors="coerce")
    result["acc"] = pd.to_numeric(result["acc"], errors="coerce")
    return result.dropna(subset=["lead", "acc"]).sort_values(["model_key", "seed", "lead"])


def load_deep_init(records: Sequence[Mapping[str, object]]) -> pd.DataFrame:
    frames = []
    for record in records:
        path = Path(str(record["result_dir"])) / "spb_init_month_lead.csv"
        if not path.is_file():
            continue
        frame = pd.read_csv(path)
        if not {"init_month", "lead", "acc"}.issubset(frame.columns):
            continue
        frame = frame[["init_month", "lead", "acc"]].copy()
        frame.insert(0, "model_key", str(record["model_key"]))
        frame.insert(1, "seed", int(record["seed"]))
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["model_key", "seed", "init_month", "lead", "acc"])
    result = pd.concat(frames, ignore_index=True)
    for column in ("init_month", "lead", "acc"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    return result.dropna(subset=["init_month", "lead", "acc"])


def load_nmme(path: Path) -> Tuple[pd.DataFrame, List[str]]:
    wide = pd.read_csv(path)
    if not {"lead_month", "lead_index"}.issubset(wide.columns):
        raise ValueError("NMME CSV must contain lead_index and lead_month columns")
    members = [column for column in wide.columns if column not in {"lead_index", "lead_month"}]
    if not members:
        raise ValueError("NMME CSV contains no member columns")
    long = wide.melt(
        id_vars=["lead_index", "lead_month"],
        value_vars=members,
        var_name="member",
        value_name="acc",
    )
    long["lead"] = pd.to_numeric(long["lead_month"], errors="coerce")
    long["acc"] = pd.to_numeric(long["acc"], errors="coerce")
    return long.dropna(subset=["lead", "acc"]), members


def aggregate_deep(deep: pd.DataFrame) -> pd.DataFrame:
    if deep.empty:
        return pd.DataFrame(columns=["model_key", "lead", "mean", "std", "n"])
    return (
        deep.groupby(["model_key", "lead"], as_index=False)["acc"]
        .agg(mean="mean", std="std", n="count")
        .sort_values(["model_key", "lead"])
    )


def effective_lead(values: pd.DataFrame, threshold: float = 0.5) -> float:
    """Return the last consecutive finite lead with ACC >= threshold."""
    if values.empty:
        return float("nan")
    values = values.sort_values("lead")
    for row in values.itertuples(index=False):
        if float(row.acc) < threshold:
            return max(float(row.lead) - 1.0, 0.0)
    return float(values.lead.max())


def build_effective_table(deep: pd.DataFrame, nmme: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model_key, seed), group in deep.groupby(["model_key", "seed"], sort=False):
        rows.append(
            {
                "system_type": "deep_learning",
                "system": model_key,
                "member": f"seed{int(seed)}",
                "period": DL_PERIOD,
                "effective_lead": effective_lead(group),
            }
        )
    for member, group in nmme.groupby("member", sort=False):
        rows.append(
            {
                "system_type": "NMME",
                "system": "nmme",
                "member": str(member),
                "period": NMME_PERIOD,
                "effective_lead": effective_lead(group),
            }
        )
    return pd.DataFrame(rows)


def plot_deep_lines(ax: plt.Axes, aggregate: pd.DataFrame, x_min: float, x_max: float) -> None:
    for model_key in MODEL_ORDER:
        curve = aggregate.query("model_key == @model_key and lead >= @x_min and lead <= @x_max")
        if curve.empty:
            continue
        x = curve.lead.to_numpy(float)
        y = curve["mean"].to_numpy(float)
        sd = curve["std"].fillna(0.0).to_numpy(float)
        color = MODEL_COLORS[model_key]
        lw = 2.25 if model_key == "atmosformer" else 1.45
        alpha = 0.18 if model_key == "atmosformer" else 0.10
        ax.fill_between(x, y - sd, y + sd, color=color, alpha=alpha, lw=0, zorder=2)
        ax.plot(x, y, color=color, lw=lw, ls="-", zorder=4, label=MODEL_LABELS[model_key])


def plot_nmme_lines(ax: plt.Axes, nmme: pd.DataFrame, x_min: float, x_max: float) -> None:
    members = list(nmme.member.drop_duplicates())
    for index, member in enumerate(members):
        curve = nmme.query("member == @member and lead >= @x_min and lead <= @x_max")
        if curve.empty:
            continue
        ax.plot(
            curve.lead.to_numpy(float),
            curve.acc.to_numpy(float),
            color=NMME_COLORS[index % len(NMME_COLORS)],
            lw=0.85,
            ls="--",
            alpha=0.78,
            zorder=1,
        )


def panel_a(ax: plt.Axes, aggregate: pd.DataFrame, nmme: pd.DataFrame, members: Sequence[str]) -> None:
    plot_nmme_lines(ax, nmme, 1, 24)
    plot_deep_lines(ax, aggregate, 1, 24)
    ax.set_xlim(1, 24)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("Prediction lead (months)")
    ax.set_ylabel("ACC")
    configure_axis(ax)
    ax.set_xticks([1, 3, 6, 9, 12, 15, 18, 21, 24])
    dl_handles = [
        Line2D(
            [0],
            [0],
            color=MODEL_COLORS[key],
            lw=2.3 if key == "atmosformer" else 1.5,
            ls="-",
            label=f"{MODEL_LABELS[key]} ({DL_PERIOD})",
        )
        for key in MODEL_ORDER
    ]
    nmme_handles = [
        Line2D(
            [0],
            [0],
            color=NMME_COLORS[index % len(NMME_COLORS)],
            lw=0.9,
            ls="--",
            label=f"{member} ({NMME_PERIOD})",
        )
        for index, member in enumerate(members)
    ]
    dl_legend = ax.legend(handles=dl_handles, loc="upper right", fontsize=5.0, handlelength=2.0)
    ax.add_artist(dl_legend)
    ax.legend(
        handles=nmme_handles,
        loc="lower left",
        ncol=2,
        fontsize=4.45,
        handlelength=1.8,
        columnspacing=0.75,
        borderpad=0.4,
    )
    panel_label(ax, "a")


def panel_b(ax: plt.Axes, aggregate: pd.DataFrame, nmme: pd.DataFrame) -> None:
    common = nmme.query("lead >= 5 and lead <= 11")
    if not common.empty:
        grouped = common.groupby("lead")["acc"]
        minimum = grouped.min()
        maximum = grouped.max()
        mean = grouped.mean()
        x = mean.index.to_numpy(float)
        ax.fill_between(x, minimum.to_numpy(float), maximum.to_numpy(float), color="#AAB2B8", alpha=0.25, lw=0)
        ax.plot(x, mean.to_numpy(float), color="#535B61", lw=1.0, ls="--", zorder=2)
        plot_nmme_lines(ax, common, 5, 11)
    plot_deep_lines(ax, aggregate, 5, 11)
    ax.set_xlim(5, 11)
    ax.set_ylim(0.35, 0.95)
    ax.set_xlabel("Prediction lead (months)")
    ax.set_ylabel("ACC")
    configure_axis(ax)
    ax.set_xticks([5, 6, 7, 8, 9, 10, 11])
    panel_label(ax, "b")


def panel_c(ax: plt.Axes, init: pd.DataFrame) -> None:
    atmos = init.query("model_key == 'atmosformer' and seed == 2025")
    if atmos.empty:
        ax.text(0.5, 0.5, "data unavailable", ha="center", va="center", transform=ax.transAxes)
        panel_label(ax, "c")
        return
    if atmos.duplicated(["init_month", "lead"]).any():
        raise ValueError("Duplicate AtmosFormer seed2025 initialization-month cells")
    table = atmos.pivot(index="init_month", columns="lead", values="acc")
    table = table.reindex(index=range(1, 13), columns=range(1, 25))
    values = table.to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError("AtmosFormer seed2025 SPB grid contains non-finite values")
    x = np.arange(1, 25)
    y = np.arange(1, 13)
    levels = np.linspace(0.0, 1.0, 11)
    norm = TwoSlopeNorm(vmin=0.0, vcenter=0.5, vmax=1.0)
    cmap = mpl.colormaps["RdBu_r"].copy()
    cmap.set_under("#2166AC")
    cmap.set_over("#8E1230")
    filled = ax.contourf(
        x,
        y,
        values,
        levels=levels,
        cmap=cmap,
        norm=norm,
        extend="both",
    )
    contour = ax.contour(
        x,
        y,
        values,
        levels=[0.5, 0.7, 0.9],
        colors="#25292C",
        linewidths=[0.75, 0.50, 0.50],
    )
    ax.clabel(contour, fmt="%.1f", fontsize=4.7, inline=True, inline_spacing=1)
    ax.set_xlim(1, 24)
    ax.set_ylim(1, 12)
    ax.set_xlabel("Lead (months)")
    ax.set_ylabel("Initialization month")
    ax.set_xticks([3, 6, 9, 12, 15, 18, 21, 24])
    ax.set_yticks(y, MONTH_LABELS)
    ax.tick_params(direction="in", top=False, right=False, length=2.2, width=0.55, pad=1.8, labelsize=5.3)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.65)
        spine.set_color("#25292C")
    colorbar = ax.figure.colorbar(filled, ax=ax, fraction=0.046, pad=0.035, ticks=np.arange(0.0, 1.01, 0.2))
    colorbar.set_label("ACC", fontsize=6.5)
    colorbar.ax.tick_params(labelsize=5.3, length=2)
    panel_label(ax, "c")


def panel_d(ax: plt.Axes, effective: pd.DataFrame) -> None:
    model_values = [
        effective.query("system == @model_key").effective_lead.to_numpy(float)
        for model_key in MODEL_ORDER
    ]
    positions = np.arange(1, 5)
    box = ax.boxplot(
        model_values,
        positions=positions[:3],
        widths=0.48,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#202428", "lw": 0.8},
        whiskerprops={"color": "#4A5055", "lw": 0.7},
        capprops={"color": "#4A5055", "lw": 0.7},
        boxprops={"lw": 0.7},
    )
    for patch, model_key in zip(box["boxes"], MODEL_ORDER):
        patch.set_facecolor(MODEL_COLORS[model_key])
        patch.set_alpha(0.78 if model_key == "atmosformer" else 0.55)
        patch.set_edgecolor(MODEL_COLORS[model_key])

    rng = np.random.default_rng(2025)
    for position, values, model_key in zip(positions[:3], model_values, MODEL_ORDER):
        jitter = rng.uniform(-0.12, 0.12, size=len(values))
        ax.scatter(
            np.full(len(values), position) + jitter,
            values,
            s=11 if model_key == "atmosformer" else 9,
            color=MODEL_COLORS[model_key],
            edgecolor="white",
            linewidth=0.35,
            zorder=4,
        )

    nmme_values = effective.query("system == 'nmme'").effective_lead.to_numpy(float)
    jitter = rng.uniform(-0.12, 0.12, size=len(nmme_values))
    ax.scatter(
        np.full(len(nmme_values), 4.0) + jitter,
        nmme_values,
        s=12,
        color="#68727A",
        edgecolor="white",
        linewidth=0.35,
        zorder=4,
    )
    if len(nmme_values):
        ax.hlines(np.nanmedian(nmme_values), 3.72, 4.28, color="#343A40", lw=0.8, zorder=3)
    ax.axhline(18, color="#7A838A", lw=0.55, ls=(0, (2, 2)), zorder=0)
    ax.text(4.28, 18.25, "18-month target", ha="right", va="bottom", fontsize=5.2, color="#626A70")
    ax.set_xlim(0.45, 4.55)
    ax.set_ylim(0, 24.5)
    ax.set_ylabel("Effective lead (ACC >= 0.5; months)")
    ax.set_xticks(positions, [
        f"AtmosFormer\n{DL_PERIOD}",
        f"CNN\n{DL_PERIOD}",
        f"Geoformer\n{DL_PERIOD}",
        f"NMME members\n{NMME_PERIOD}",
    ])
    ax.tick_params(axis="x", labelsize=5.2)
    configure_axis(ax, acc_reference=False)
    panel_label(ax, "d")


def write_sources(
    output_dir: Path,
    deep: pd.DataFrame,
    nmme: pd.DataFrame,
    init: pd.DataFrame,
    effective: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    deep.to_csv(output_dir / "source_fig1_deep_learning_acc.csv", index=False)
    nmme.to_csv(output_dir / "source_fig1_nmme_acc.csv", index=False)
    init.query("model_key == 'atmosformer' and seed == 2025").to_csv(
        output_dir / "source_fig1_atmosformer_init_month_acc.csv", index=False
    )
    effective.to_csv(output_dir / "source_fig1_effective_lead.csv", index=False)
    (output_dir / "fig1_reference_skill_caption_zh.txt").write_text(CAPTION_ZH + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training_results_dir", type=Path, default=PROJECT_ROOT / "results_spb_core_claim_training")
    parser.add_argument("--nmme_csv", type=Path, required=True)
    parser.add_argument("--configuration", default="global_slp_sst_tauu")
    parser.add_argument("--output_dir", type=Path, default=PROJECT_ROOT / "results_spb_core_claim" / "figures_nature")
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()

    records = discover_records(args.training_results_dir, args.configuration)
    expected = {(model, seed) for model in MODEL_ORDER for seed in range(2025, 2030)}
    present = {(str(row["model_key"]), int(row["seed"])) for row in records}
    missing = sorted(expected - present)
    if missing:
        raise SystemExit(f"Missing deep-learning runs for configuration {args.configuration}: {missing}")

    deep = load_deep_leads(records)
    init = load_deep_init(records)
    nmme, members = load_nmme(args.nmme_csv)
    aggregate = aggregate_deep(deep)
    effective = build_effective_table(deep, nmme)
    if aggregate.empty or nmme.empty or effective.empty:
        raise SystemExit("Insufficient data to build the reference-style figure")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_sources(args.output_dir, deep, nmme, init, effective)

    fig = plt.figure(figsize=(183 / 25.4, 118 / 25.4))
    grid = fig.add_gridspec(2, 2, left=0.065, right=0.965, bottom=0.09, top=0.965, wspace=0.34, hspace=0.42)
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])]
    panel_a(axes[0], aggregate, nmme, members)
    panel_b(axes[1], aggregate, nmme)
    panel_c(axes[2], init)
    panel_d(axes[3], effective)
    save_publication(fig, args.output_dir / "fig1_reference_skill", args.dpi)
    print(f"[Figure 1] wrote {args.output_dir / 'fig1_reference_skill'}")
    print(f"[Figure 1] deep-learning runs={len(records)}, NMME members={len(members)}, NMME max lead={int(nmme.lead.max())}")


if __name__ == "__main__":
    main()
