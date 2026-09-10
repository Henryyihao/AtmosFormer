import torch
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.dates as mdates
from scipy.ndimage import gaussian_filter
from scipy.signal import welch
from scipy.interpolate import RectBivariateSpline
from matplotlib.ticker import MaxNLocator

INIT_MONTH = {}


class EarlyStopping:
    def __init__(self, patience=7, verbose=True, delta=0, path='checkpoint.pth'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model, args, stats=None):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, args, stats)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'  [EarlyStopping] No improvement. Counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, args, stats)
            self.counter = 0
            if self.verbose:
                print(f'  [EarlyStopping] Validation loss decreased. Counter reset to 0/{self.patience}')

    def save_checkpoint(self, val_loss, model, args, stats):
        torch.save({
            'args': args,
            'model_state_dict': model.state_dict(),
            'stats': stats
        }, self.path)
        self.val_loss_min = val_loss


def denorm_nino34(arr, stats):
    """将归一化的 Nino3.4 标量反归一化为物理量 (°C)"""
    mean = stats.get('nino34_mean', 0.0)
    std  = stats.get('nino34_std', 1.0)
    return arr * std + mean


def smooth_ts_3ma(series: np.ndarray) -> np.ndarray:
    """
    对 (N, T) 时间序列矩阵沿 lead 维（axis=1）做 3 个月中心滑动平均。


    边界处理：
      t=0   : mean(t=0, t=1)        只用右侧邻居
      t=T-1 : mean(t=T-2, t=T-1)   只用左侧邻居
      其余   : mean(t-1, t, t+1)    标准三点中心平均

    Parameters
    ----------
    series : (N, T)  物理量矩阵，每行是一个样本的 T 步预测/观测序列

    Returns
    -------
    smoothed : (N, T)  同形状，沿 lead 维平滑后的序列
    """
    N, T = series.shape
    smoothed = np.empty_like(series)

    smoothed[:, 0] = np.nanmean(series[:, 0:2], axis=1)
    smoothed[:, T - 1] = np.nanmean(series[:, T - 2:T], axis=1)
    for t in range(1, T - 1):
        smoothed[:, t] = np.nanmean(series[:, t - 1:t + 2], axis=1)

    return smoothed


def _corr_per_lead(preds: np.ndarray, trues: np.ndarray) -> np.ndarray:
    """内部工具：逐 lead 计算 Pearson 相关，返回 (T,) 数组。"""
    T = preds.shape[1]
    acc = np.full(T, np.nan)
    for t in range(T):
        p  = preds[:, t]
        tr = trues[:, t]
        valid = ~np.isnan(p) & ~np.isnan(tr)
        if (np.sum(valid) > 1
                and np.std(p[valid]) > 1e-6
                and np.std(tr[valid]) > 1e-6):
            acc[t] = float(np.corrcoef(p[valid], tr[valid])[0, 1])
    return acc


def compute_acc_per_lead(preds: np.ndarray, trues: np.ndarray) -> np.ndarray:
    """
    逐 lead 计算原始 Pearson ACC（不做任何平滑）。

    Parameters
    ----------
    preds : (N, T_out)  物理量（已反归一化）
    trues : (N, T_out)  物理量（已反归一化）

    Returns
    -------
    acc : (T_out,)  每个 lead 的 ACC，无效位置填 np.nan
    """
    return _corr_per_lead(preds, trues)


def compute_acc_3ma(preds: np.ndarray, trues: np.ndarray) -> np.ndarray:
    """
    先对 Nino3.4 时间序列做 3 个月滑动平均，再逐 lead 计算 ACC。
    与 CTEFNet 论文定义完全一致。

    ⚠ 与"平滑 ACC 数组"的区别：
      - 平滑 ACC 数组：ACC 曲线本身已很光滑，变化极小（几乎看不出效果）
      - 先平滑时间序列：去掉每个样本序列里的月际高频噪声，再算相关；
        长 lead 处噪声占比大，去噪后 ACC 可提升 0.05~0.15

    Parameters
    ----------
    preds : (N, T_out)  物理量（已反归一化）
    trues : (N, T_out)  物理量（已反归一化）

    Returns
    -------
    acc_3ma : (T_out,)  3MA 时间序列对应的逐 lead ACC
    """
    preds_sm = smooth_ts_3ma(preds)
    trues_sm = smooth_ts_3ma(trues)
    return _corr_per_lead(preds_sm, trues_sm)


def compute_effective_lead(acc_3ma: np.ndarray, threshold: float = 0.5) -> int:
    """
    计算有效预测 lead 月数：ACC_3MA 首次跌破 threshold 之前的月数。

    采用"首次跌破"策略，与多数 ENSO 论文（包括 CTEFNet）一致：
      有效 lead = 首个 acc_3ma[i] < threshold 的索引 i（对应 lead = i+1 月）

    Parameters
    ----------
    acc_3ma   : (T,)  compute_acc_3ma 返回的数组
    threshold : float 默认 0.5

    Returns
    -------
    eff_lead : int  有效 lead 月数（0 表示第 1 个月就失效，T 表示全程有效）
    """
    for i, v in enumerate(acc_3ma):
        if not np.isnan(v) and v < threshold:
            return i
    return len(acc_3ma)


def evaluate_nino34_skill_decay(all_preds, all_trues, stats, save_dir, args):
    """
    all_preds: (N, T_out) 归一化
    all_trues: (N, T_out) 归一化

    同时绘制：
      - 原始 ACC（细线 + 圆点）：对原始时间序列逐 lead 计算
      - 3MA-ACC（粗虚线）：先对时间序列做 3 个月滑动平均，再逐 lead 计算
        ⚠ 这才是 CTEFNet 论文的正确计算方式，两条线应有明显差距
      - 有效 lead 竖线标注（基于 3MA-ACC 首次 < 0.5 的位置）
    CSV 增加 ACC_3MA 列。
    返回 (accs_raw, accs_3ma, rmses) 供调用方使用。
    """
    print("\n--- Generating Nino3.4 Skill Decay Plots ---")
    os.makedirs(save_dir, exist_ok=True)

    preds = denorm_nino34(all_preds, stats)
    trues = denorm_nino34(all_trues, stats)

    T      = preds.shape[1]
    x_axis = np.arange(1, T + 1)

    accs_raw = compute_acc_per_lead(preds, trues)

    accs_3ma = compute_acc_3ma(preds, trues)
    eff_lead = compute_effective_lead(accs_3ma, threshold=0.5)

    rmses = []
    for t in range(T):
        p  = preds[:, t]
        tr = trues[:, t]
        valid = ~np.isnan(p) & ~np.isnan(tr)
        rmses.append(
            float(np.sqrt(np.nanmean((p[valid] - tr[valid]) ** 2)))
            if np.sum(valid) > 0 else np.nan
        )
    rmses = np.array(rmses)

    print(f"  Raw  ACC (mean leads 1-{T}): {np.nanmean(accs_raw):.4f}")
    print(f"  3MA  ACC (mean leads 1-{T}): {np.nanmean(accs_3ma):.4f}")
    print(f"  Effective lead (3MA-ACC >= 0.5): {eff_lead} months")

    df = pd.DataFrame({
        'Lead_Time': x_axis,
        'ACC':       np.round(accs_raw, 4),
        'ACC_3MA':   np.round(accs_3ma, 4),
        'RMSE':      np.round(rmses, 4),
    })
    df.to_csv(os.path.join(save_dir, 'nino34_skill_decay.csv'), index=False)

    fig, ax1 = plt.subplots(figsize=(9, 5), dpi=150)
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.set_xlim(x_axis[0] - 0.5, x_axis[-1] + 0.5)

    ax1.plot(x_axis, accs_raw,
             color='tab:red', marker='o', linewidth=1.2, markersize=4,
             alpha=0.5, label='ACC (raw)')

    ax1.plot(x_axis, accs_3ma,
             color='tab:red', linewidth=2.5, linestyle='--',
             label='ACC (3-month MA)')

    if 0 < eff_lead <= T:
        ax1.axvline(eff_lead, color='darkred', linestyle=':', linewidth=1.5,
                    label=f'Eff. lead = {eff_lead} months (3MA)')

    ax1.axhline(0.5, color='gray', linestyle=':', linewidth=1.5)
    ax1.set_xlabel('Forecast Lead (Months)', fontsize=11)
    ax1.set_ylabel('Nino3.4 Correlation (ACC)', color='tab:red', fontsize=11)
    ax1.tick_params(axis='y', labelcolor='tab:red')
    ax1.set_ylim(0.0, 1.05)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc='upper right', fontsize=9)

    ax2 = ax1.twinx()
    ax2.plot(x_axis, rmses,
             color='tab:blue', marker='s', linestyle='--',
             linewidth=1.5, markersize=4, label='RMSE')
    ax2.set_ylabel('Nino3.4 RMSE (°C)', color='tab:blue', fontsize=11)
    ax2.tick_params(axis='y', labelcolor='tab:blue')
    ax2.legend(loc='upper left', fontsize=9)

    plt.title(
        f'Nino3.4 Prediction Skill Decay  '
        f'[Eff. lead = {eff_lead} months @ 3MA-ACC ≥ 0.5]',
        fontsize=11)
    plt.savefig(os.path.join(save_dir, 'nino34_skill_decay.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved nino34_skill_decay.png")

    return accs_raw, accs_3ma, rmses


def save_nino34_to_csv_and_plot(all_preds, all_trues, stats, save_dir,
                                 output_len, test_times, args):
    """
    all_preds, all_trues: (N, T_out) 归一化
    test_times: DatetimeIndex 或 list, 长度 >= N
    """
    print("\n--- Saving Nino3.4 CSV and Time Series Plot ---")
    os.makedirs(save_dir, exist_ok=True)

    preds = denorm_nino34(all_preds, stats)
    trues = denorm_nino34(all_trues, stats)
    N, T = preds.shape

    input_len = getattr(args, 'input_len', 12)

    for lead in range(T):
        rows = []
        for i in range(N):
            if test_times is not None and i < len(test_times):
                base = pd.Timestamp(test_times[i])
                target_date = base + pd.DateOffset(months=input_len + lead)
                date_str = target_date.strftime('%Y-%m')
            else:
                date_str = f"sample_{i}"
            rows.append({
                'date': date_str,
                'lead_month': lead + 1,
                'nino34_pred': preds[i, lead],
                'nino34_true': trues[i, lead],
            })
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(save_dir, f'nino34_lead{lead+1}.csv'), index=False)

    key_leads = [1, 3, 6, 12, 18, 24]
    key_leads = [l for l in key_leads if l <= T]

    for lead in key_leads:
        lead_idx = lead - 1
        p = preds[:, lead_idx]
        t = trues[:, lead_idx]

        if test_times is not None and len(test_times) >= N:
            dates = [pd.Timestamp(test_times[i]) + pd.DateOffset(months=input_len + lead_idx)
                     for i in range(N)]
        else:
            dates = list(range(N))

        fig, ax = plt.subplots(figsize=(12, 4), dpi=150)
        ax.plot(dates, t, 'k-', linewidth=1.5, label='Observed', alpha=0.8)
        ax.plot(dates, p, 'r-', linewidth=1.2, label=f'Predicted (Lead {lead}m)', alpha=0.8)
        ax.fill_between(dates, 0.5, ax.get_ylim()[1] if ax.get_ylim()[1] > 0.5 else 2.0,
                         alpha=0.05, color='red')
        ax.fill_between(dates, -0.5, ax.get_ylim()[0] if ax.get_ylim()[0] < -0.5 else -2.0,
                         alpha=0.05, color='blue')
        ax.axhline(0.5, color='red', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.axhline(-0.5, color='blue', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.axhline(0, color='gray', linewidth=0.5)
        ax.set_ylabel('Nino3.4 Index (deg C)', fontsize=10)
        ax.set_title(f'Nino3.4 Prediction vs Observation (Lead = {lead} months)', fontsize=11)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.4)
        if isinstance(dates[0], pd.Timestamp):
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            fig.autofmt_xdate(rotation=30)
        plt.savefig(os.path.join(save_dir, f'nino34_timeseries_lead{lead}.png'),
                    bbox_inches='tight', dpi=150)
        plt.close()

    print(f"  Saved Nino3.4 CSV and time series plots for leads {key_leads}")


def plot_nino34_lead_correlation(all_preds_dict, all_trues, stats,
                                  save_dir, args, test_times=None):
    """
    all_preds_dict: {model_name: (N, T_out)} 归一化
    all_trues: (N, T_out) 归一化

    [修改]
      - 每个模型同时绘制原始 ACC（细实线）和 3MA-ACC（粗虚线）
      - 图例中标注各模型的有效预测月数
      - 两条线共享同一颜色，3MA 线加粗以示区分
    """
    print("\n--- Plotting Nino3.4 Lead Correlation ---")
    os.makedirs(save_dir, exist_ok=True)
    trues = denorm_nino34(all_trues, stats)
    T = trues.shape[1]

    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    colors = ['#C0392B', '#2980B9', '#27AE60', '#8E44AD', '#E67E22']

    summary_lines = []
    for idx, (name, preds_raw) in enumerate(all_preds_dict.items()):
        preds    = denorm_nino34(preds_raw, stats)
        color    = colors[idx % len(colors)]

        accs_raw = compute_acc_per_lead(preds, trues)
        accs_3ma = compute_acc_3ma(preds, trues)
        eff_lead = compute_effective_lead(accs_3ma, threshold=0.5)

        summary_lines.append(f"  {name}: eff_lead={eff_lead}m  "
                              f"mean_raw={np.nanmean(accs_raw):.4f}  "
                              f"mean_3ma={np.nanmean(accs_3ma):.4f}")

        ax.plot(range(1, T + 1), accs_raw,
                color=color, linewidth=1.2, alpha=0.45,
                marker='o', markersize=3)
        ax.plot(range(1, T + 1), accs_3ma,
                color=color, linewidth=2.5, linestyle='--',
                label=f'{name}  [eff={eff_lead}m]')

    print('\n'.join(summary_lines))

    ax.axhline(0.5, color='gray', linestyle=':', linewidth=1.5, label='ACC = 0.5')
    ax.set_xlabel('Forecast Lead (Months)', fontsize=11)
    ax.set_ylabel('Correlation (ACC)', fontsize=11)
    ax.set_title('Nino3.4 Correlation vs Lead Time\n'
                 '(dashed = 3-month MA, solid = raw)', fontsize=11)
    ax.set_ylim(0.0, 1.05)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'nino34_lead_correlation.png'),
                bbox_inches='tight', dpi=150)
    plt.close()
    print("  Saved nino34_lead_correlation.png")


def plot_seasonal_lead_heatmap(all_preds, all_trues, stats, save_dir,
                                args, test_times):
    """
    SPB 季节性热力图:
      X = Prediction Lead
      Y = Target Calendar Month
      Color = ACC
    """
    print("\n--- Generating Seasonal Lead Heatmap (Smooth Contour Style) ---")
    if test_times is None or len(test_times) == 0:
        print("  [Warning] test_times missing, skipping.")
        return

    os.makedirs(save_dir, exist_ok=True)

    preds = denorm_nino34(all_preds, stats)
    trues = denorm_nino34(all_trues, stats)

    N, T = preds.shape
    input_len = getattr(args, 'input_len', 12)

    acc_matrix = np.full((12, T), np.nan)
    sample_cnt = np.zeros((12, T), dtype=int)

    for lead_idx in range(T):
        month_preds = {m: [] for m in range(1, 13)}
        month_trues = {m: [] for m in range(1, 13)}

        for b in range(N):
            if b >= len(test_times):
                break

            try:
                base_ts = pd.Timestamp(test_times[b])
            except Exception:
                continue

            target_date = base_ts + pd.DateOffset(months=input_len + lead_idx)
            target_month = target_date.month

            month_preds[target_month].append(preds[b, lead_idx])
            month_trues[target_month].append(trues[b, lead_idx])

        for m in range(1, 13):
            n = len(month_preds[m])
            sample_cnt[m - 1, lead_idx] = n

            if n >= 2:
                p_arr = np.asarray(month_preds[m], dtype=np.float32)
                t_arr = np.asarray(month_trues[m], dtype=np.float32)

                valid = ~np.isnan(p_arr) & ~np.isnan(t_arr)
                if (
                    np.sum(valid) >= 2
                    and np.std(p_arr[valid]) > 1e-6
                    and np.std(t_arr[valid]) > 1e-6
                ):
                    acc_matrix[m - 1, lead_idx] = np.corrcoef(
                        p_arr[valid], t_arr[valid]
                    )[0, 1]

    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    acc_df = pd.DataFrame(
        acc_matrix,
        index=month_names,
        columns=[f'Lead_{i+1}' for i in range(T)]
    )
    acc_df.to_csv(os.path.join(save_dir, 'seasonal_lead_heatmap_acc_matrix.csv'))

    cnt_df = pd.DataFrame(
        sample_cnt,
        index=month_names,
        columns=[f'Lead_{i+1}' for i in range(T)]
    )
    cnt_df.to_csv(os.path.join(save_dir, 'seasonal_lead_heatmap_sample_count.csv'))

    smooth = acc_matrix.copy()

    valid_mask = ~np.isnan(smooth)
    if np.any(valid_mask):
        fill_val = np.nanmean(smooth)
        smooth[~valid_mask] = fill_val

        smooth = gaussian_filter(smooth, sigma=0.8)

    else:
        print("  [Warning] No valid ACC values for SPB heatmap.")
        return

    x = np.arange(1, T + 1)
    y = np.arange(1, 13)
    X, Y = np.meshgrid(x, y)

    fig, ax = plt.subplots(figsize=(10, 4.6), dpi=150)

    fill_levels = np.linspace(0.0, 1.0, 21)

    line_levels = [0.5, 0.6, 0.7, 0.8, 0.9]

    cf = ax.contourf(
        X, Y, smooth,
        levels=fill_levels,
        cmap='coolwarm',
        vmin=0.0, vmax=1.0,
        extend='both'
    )

    cs = ax.contour(
        X, Y, smooth,
        levels=line_levels,
        colors='k',
        linewidths=1.0,
        alpha=0.7
    )

    ax.clabel(cs, fmt='%.1f', inline=True, fontsize=8)

    ax.set_xlim(1, T)
    ax.set_ylim(1, 12)

    if T <= 12:
        xticks = np.arange(1, T + 1, 1)
    elif T <= 24:
        xticks = np.arange(1, T + 1, 2)
    else:
        xticks = np.arange(1, T + 1, max(1, T // 10))
    ax.set_xticks(xticks)

    ax.set_yticks(np.arange(1, 13))
    ax.set_yticklabels(month_names, fontsize=11)

    ax.set_xlabel('Prediction lead (months)', fontsize=12)
    ax.set_ylabel('Month', fontsize=12)

    ax.set_title('Seasonality and Lead-time Performance (SPB)',
                 fontsize=13, pad=6)

    for spine in ax.spines.values():
        spine.set_linewidth(1.0)

    ax.tick_params(axis='both', which='major', labelsize=11, direction='in', length=3)

    cbar = fig.colorbar(cf, ax=ax, pad=0.012, fraction=0.035)
    cbar.set_label('Correlation Skill (ACC)', fontsize=11)
    cbar.ax.tick_params(labelsize=10)

    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, 'seasonal_lead_heatmap.png'),
        bbox_inches='tight',
        dpi=180
    )
    plt.close()

    print("  Saved seasonal_lead_heatmap.png")
    print("  Saved seasonal_lead_heatmap_acc_matrix.csv")
    print("  Saved seasonal_lead_heatmap_sample_count.csv")

def plot_init_month_lead_heatmap(all_preds, all_trues, stats, save_dir,
                                  args, test_times):
    """
    """
    print("\n--- Generating Init-month Lead Heatmap ---")

    if test_times is None or len(test_times) == 0:
        print("  [Warning] test_times missing, skipping.")
        return

    os.makedirs(save_dir, exist_ok=True)

    preds = denorm_nino34(all_preds, stats)
    trues = denorm_nino34(all_trues, stats)

    N, T = preds.shape
    input_len = getattr(args, 'input_len', 12)

    acc_matrix = np.full((12, T), np.nan)
    sample_cnt = np.zeros((12, T), dtype=int)

    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    print("  [Init-month Check]")
    for b in range(min(3, N)):
        base_ts = pd.Timestamp(test_times[b])
        init_ts = base_ts + pd.DateOffset(months=input_len - 1)
        lead1_ts = base_ts + pd.DateOffset(months=input_len)
        print(
            f"    sample {b:02d}: "
            f"input_start={base_ts.strftime('%Y-%m')} | "
            f"init={init_ts.strftime('%Y-%m')} | "
            f"lead1_target={lead1_ts.strftime('%Y-%m')}"
        )

    for lead_idx in range(T):
        month_preds = {m: [] for m in range(1, 13)}
        month_trues = {m: [] for m in range(1, 13)}

        for b in range(N):
            if b >= len(test_times):
                break

            try:
                base_ts = pd.Timestamp(test_times[b])
            except Exception:
                continue

            init_date = base_ts + pd.DateOffset(months=input_len - 1)
            init_month = init_date.month

            month_preds[init_month].append(preds[b, lead_idx])
            month_trues[init_month].append(trues[b, lead_idx])

        for m in range(1, 13):
            n = len(month_preds[m])
            sample_cnt[m - 1, lead_idx] = n

            if n >= 2:
                p_arr = np.asarray(month_preds[m], dtype=np.float32)
                t_arr = np.asarray(month_trues[m], dtype=np.float32)

                valid = ~np.isnan(p_arr) & ~np.isnan(t_arr)

                if (
                    np.sum(valid) >= 2
                    and np.std(p_arr[valid]) > 1e-6
                    and np.std(t_arr[valid]) > 1e-6
                ):
                    acc_matrix[m - 1, lead_idx] = np.corrcoef(
                        p_arr[valid], t_arr[valid]
                    )[0, 1]

    acc_df = pd.DataFrame(
        acc_matrix,
        index=month_names,
        columns=[f'Lead_{i+1}' for i in range(T)]
    )
    acc_df.to_csv(os.path.join(save_dir, 'init_month_lead_heatmap_acc_matrix.csv'))

    cnt_df = pd.DataFrame(
        sample_cnt,
        index=month_names,
        columns=[f'Lead_{i+1}' for i in range(T)]
    )
    cnt_df.to_csv(os.path.join(save_dir, 'init_month_lead_heatmap_sample_count.csv'))

    print(
        f"  [Init-SPB] sample count per cell: "
        f"min={sample_cnt.min()}, max={sample_cnt.max()}, "
        f"mean={sample_cnt.mean():.1f}"
    )

    valid_mask = ~np.isnan(acc_matrix)

    if np.sum(valid_mask) == 0:
        print("  [Warning] acc_matrix is all NaN, skipping plot.")
        return

    plot_field = acc_matrix.copy()
    fill_val = np.nanmean(plot_field)
    plot_field[np.isnan(plot_field)] = fill_val

    plot_field = gaussian_filter(plot_field, sigma=0.8)

    month_axis = np.arange(1, 13)
    lead_axis = np.arange(1, T + 1)

    kx = min(3, len(month_axis) - 1)
    ky = min(3, len(lead_axis) - 1)

    spline = RectBivariateSpline(
        month_axis,
        lead_axis,
        plot_field,
        kx=kx,
        ky=ky
    )

    month_fine = np.linspace(1, 12, 240)
    lead_fine = np.linspace(1, T, max(240, T * 20))

    smooth_fine = spline(month_fine, lead_fine)
    smooth_fine = np.clip(smooth_fine, 0.0, 1.0)

    fig, ax = plt.subplots(figsize=(10, 4.8), dpi=180)

    fill_levels = np.linspace(0.0, 1.0, 11)
    line_levels = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    cf = ax.contourf(
        lead_fine,
        month_fine,
        smooth_fine,
        levels=fill_levels,
        cmap='RdBu_r',
        extend='both'
    )

    cs = ax.contour(
        lead_fine,
        month_fine,
        smooth_fine,
        levels=line_levels,
        colors='0.25',
        linewidths=1.0
    )

    ax.clabel(cs, fmt='%.1f', fontsize=8, inline=True)

    cbar = fig.colorbar(cf, ax=ax, pad=0.012, fraction=0.035)
    cbar.set_label('Correlation Skill (ACC)', fontsize=11)
    cbar.set_ticks(np.linspace(0, 1, 6))
    cbar.ax.tick_params(labelsize=10)

    ax.set_xlim(1, T)
    ax.set_ylim(1, 12)

    ax.set_xlabel('Prediction lead (months)', fontsize=12)
    ax.set_ylabel('Initialization Month', fontsize=12)

    ax.set_xticks(np.arange(1, T + 1, max(1, T // 10)))
    ax.set_yticks(np.arange(1, 13))
    ax.set_yticklabels(month_names, fontsize=10)

    ax.tick_params(axis='x', labelsize=10, direction='in', length=3)
    ax.tick_params(axis='y', labelsize=10, direction='in', length=3)

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color('0.2')

    ax.set_title('Initialization Month and Lead-time Performance',
                 fontsize=14, pad=8)

    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, 'init_month_lead_heatmap.png'),
        dpi=180,
        bbox_inches='tight'
    )
    plt.close()

    print("  Saved init_month_lead_heatmap.png")
    print("  Saved init_month_lead_heatmap_acc_matrix.csv")
    print("  Saved init_month_lead_heatmap_sample_count.csv")

def plot_extreme_event_case_study(all_preds, all_trues, stats, save_dir,
                                   test_times, args):
    """找到最强 El Nino 事件，展示不同 lead 的预测"""
    print("\n--- Plotting Extreme Event Case Study ---")
    os.makedirs(save_dir, exist_ok=True)

    preds = denorm_nino34(all_preds, stats)
    trues = denorm_nino34(all_trues, stats)
    N, T = trues.shape
    input_len = getattr(args, 'input_len', 12)

    target_lead = T - 1
    max_b_idx   = np.argmax(trues[:, target_lead])

    seq_len = 24
    start_b = max(0, max_b_idx - seq_len // 2)
    end_b   = min(N, max_b_idx + seq_len // 2)

    real_traj      = trues[start_b:end_b, target_lead]
    pred_traj_long = preds[start_b:end_b, target_lead]
    mid_lead       = T // 2
    pred_traj_mid  = preds[start_b:end_b, mid_lead]

    dates = []
    for b in range(start_b, end_b):
        if test_times is not None and b < len(test_times):
            base = pd.Timestamp(test_times[b])
            dates.append(base + pd.DateOffset(months=input_len + target_lead))
        else:
            dates.append(b)

    fig, ax = plt.subplots(figsize=(9, 4), dpi=150)
    ax.plot(dates, real_traj, 'k-', linewidth=2.5, label='Observation')
    ax.plot(dates, pred_traj_mid, 'g--', linewidth=1.5, marker='s', markersize=4,
            label=f'Lead {mid_lead+1} Prediction')
    ax.plot(dates, pred_traj_long, 'r-.', linewidth=2.0, marker='o', markersize=4,
            label=f'Lead {T} Prediction')

    ax.axhline(0.5, color='gray', linestyle=':', alpha=0.7)
    ax.set_ylabel('Nino 3.4 Index (deg C)')
    ax.set_title(f'Extreme Event Tracking (Peak Sample ID: {max_b_idx})', fontsize=12)
    if isinstance(dates[0], pd.Timestamp):
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        fig.autofmt_xdate()
    ax.legend(loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(os.path.join(save_dir, 'extreme_event_case_study.png'), bbox_inches='tight')
    plt.close()
    print("  Saved extreme_event_case_study.png")


def plot_power_spectrum(all_preds, all_trues, stats, save_dir):
    """Nino3.4 功率谱对比"""
    print("\n--- Plotting Power Spectrum ---")
    os.makedirs(save_dir, exist_ok=True)

    preds = denorm_nino34(all_preds, stats)
    trues = denorm_nino34(all_trues, stats)

    for lead_idx, lead_name in [(0, 'Lead1'), (5, 'Lead6'), (11, 'Lead12')]:
        if lead_idx >= preds.shape[1]:
            continue
        p_ts = preds[:, lead_idx]
        t_ts = trues[:, lead_idx]

        valid = ~np.isnan(p_ts) & ~np.isnan(t_ts)
        if np.sum(valid) < 24:
            continue

        nperseg = min(64, np.sum(valid) // 2)
        if nperseg < 8:
            continue

        f_p, pxx_p = welch(p_ts[valid], fs=12, nperseg=nperseg)
        f_t, pxx_t = welch(t_ts[valid], fs=12, nperseg=nperseg)

        fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
        ax.semilogy(1/f_t[1:], pxx_t[1:], 'k-', linewidth=1.5, label='Observed')
        ax.semilogy(1/f_p[1:], pxx_p[1:], 'r--', linewidth=1.5, label='Predicted')
        ax.set_xlabel('Period (years)', fontsize=10)
        ax.set_ylabel('Power Spectral Density', fontsize=10)
        ax.set_title(f'Power Spectrum ({lead_name})', fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.set_xlim(0, 10)
        plt.savefig(os.path.join(save_dir, f'power_spectrum_{lead_name}.png'),
                    bbox_inches='tight', dpi=150)
        plt.close()

    print("  Saved power_spectrum plots")
