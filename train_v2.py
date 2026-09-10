from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Sampler
from tqdm import tqdm

from dataset import (
    CALENDAR_FEATURE_NAMES,
    ENSOGlobalMechanismDataset,
    load_mechanism_file,
    resolve_map_vars,
    resolve_phys_vars,
    resolve_target_vars,
    split_segments,
)
from loss_v2 import ENSOCTMLossV2, ctm_thought_loss_v2
from loss_pathtrace import ENSOPathTraceLoss
from utils import (
    EarlyStopping,
    compute_acc_3ma,
    compute_acc_per_lead,
    compute_effective_lead,
    evaluate_nino34_skill_decay,
    plot_init_month_lead_heatmap,
    plot_seasonal_lead_heatmap,
    save_nino34_to_csv_and_plot,
)

from models import get_model_dict

CTM_MODELS = {}
PATH_TRACE_MODELS = {}
MIST_MODELS = {}


class DomainIdDataset(Dataset):
    """Remap the final ``segment_id`` field to a global domain identifier.

    ``ENSOGlobalMechanismDataset`` numbers segments locally.  Consequently an
    OBS dataset with one segment and the first CMIP model both emit id 0 when
    they are joined with ``ConcatDataset``.  MIST needs ids that are stable
    across constituent datasets, so this transparent wrapper either adds an
    offset or assigns a constant id (used for every OBS source).

    Attribute access is forwarded to the wrapped dataset so it can still be
    used as the reference dataset in ``build_datasets`` during test-only runs.
    """

    def __init__(
        self,
        dataset: Dataset,
        domain_id: Optional[int] = None,
        offset: int = 0,
        domain_map: Optional[Dict[int, int]] = None,
    ):
        self.dataset = dataset
        self.domain_id = None if domain_id is None else int(domain_id)
        self.offset = int(offset)
        self.domain_map = None if domain_map is None else {
            int(local): int(global_id) for local, global_id in domain_map.items()
        }

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        if not isinstance(sample, (tuple, list)) or len(sample) < 1:
            raise TypeError("DomainIdDataset expects a tuple/list sample ending in segment_id")
        local_id = sample[-1]
        if self.domain_id is None:
            local_int = int(local_id.item()) if torch.is_tensor(local_id) else int(local_id)
            if self.domain_map is not None:
                if local_int not in self.domain_map:
                    raise KeyError(f"No global domain mapping for local segment {local_int}")
                mapped_int = self.domain_map[local_int]
            else:
                mapped_int = local_int + self.offset
            mapped_id = torch.tensor(mapped_int, dtype=torch.long)
        else:
            mapped_id = torch.tensor(self.domain_id, dtype=torch.long)
        return tuple(sample[:-1]) + (mapped_id,)

    def __getattr__(self, name):
        dataset = self.__dict__.get("dataset", None)
        if dataset is None:
            raise AttributeError(name)
        return getattr(dataset, name)


DomainOffsetDataset = DomainIdDataset


def _dataset_domain_ids(dataset: Dataset) -> np.ndarray:
    """Read domain ids without materializing the large map tensors."""
    if isinstance(dataset, DomainIdDataset):
        if dataset.domain_id is not None:
            return np.full(len(dataset), dataset.domain_id, dtype=np.int64)
        local = _dataset_domain_ids(dataset.dataset)
        if dataset.domain_map is not None:
            try:
                return np.asarray([dataset.domain_map[int(v)] for v in local], dtype=np.int64)
            except KeyError as exc:
                raise KeyError(f"Incomplete domain_map in DomainIdDataset: {exc}") from exc
        return local + dataset.offset
    if isinstance(dataset, ConcatDataset):
        return np.concatenate([_dataset_domain_ids(ds) for ds in dataset.datasets])
    local_ids = getattr(dataset, "_segment_ids", None)
    if local_ids is not None and len(local_ids) == len(dataset):
        return np.asarray(local_ids, dtype=np.int64)
    raise TypeError(
        "Cannot infer domain ids without loading samples from dataset type "
        f"{type(dataset).__name__}; wrap it in DomainIdDataset."
    )


class DomainBalancedBatchSampler(Sampler):
    """Uniformly cover available domains in each MIST training batch.

    This is optional because it deliberately changes the empirical sampling
    distribution.  It is useful for GroupDRO with many CMIP environments: a
    normal shuffled batch of size 32 frequently omits several of 17 domains.
    Sampling is with reshuffling/replacement within small domains, so every
    batch has near-equal domain counts while an epoch retains the original
    number of optimization steps.
    """

    def __init__(self, dataset: Dataset, batch_size: int, drop_last: bool, seed: int):
        if int(batch_size) <= 0:
            raise ValueError("batch_size must be positive")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.epoch = 0
        domain_ids = _dataset_domain_ids(dataset)
        self.pools = {
            int(domain): np.flatnonzero(domain_ids == domain).astype(np.int64)
            for domain in np.unique(domain_ids)
        }
        self.pools = {domain: pool for domain, pool in self.pools.items() if pool.size > 0}
        if not self.pools:
            raise ValueError("DomainBalancedBatchSampler found no non-empty domains")

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        return math.ceil(len(self.dataset) / self.batch_size)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        domains = np.asarray(sorted(self.pools), dtype=np.int64)
        permutations = {d: rng.permutation(self.pools[int(d)]) for d in domains}
        cursors = {d: 0 for d in domains}

        def draw(domain: int, count: int):
            pieces = []
            remaining = int(count)
            while remaining > 0:
                pool = permutations[domain]
                cursor = cursors[domain]
                available = len(pool) - cursor
                if available <= 0:
                    pool = rng.permutation(self.pools[domain])
                    permutations[domain] = pool
                    cursors[domain] = 0
                    cursor, available = 0, len(pool)
                take = min(remaining, available)
                pieces.extend(pool[cursor:cursor + take].tolist())
                cursors[domain] = cursor + take
                remaining -= take
            return pieces

        for batch_idx in range(len(self)):
            current_size = self.batch_size
            if not self.drop_last and batch_idx == len(self) - 1:
                current_size = len(self.dataset) - batch_idx * self.batch_size
            if current_size <= 0:
                break
            repeats, remainder = divmod(current_size, len(domains))
            assignments = np.repeat(domains, repeats).tolist()
            if remainder:
                assignments.extend(rng.permutation(domains)[:remainder].tolist())
            rng.shuffle(assignments)
            counts = {int(d): assignments.count(int(d)) for d in set(assignments)}
            batch = []
            for domain, count in counts.items():
                batch.extend(draw(domain, count))
            rng.shuffle(batch)
            yield batch


def fix_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def unpack_batch(batch):
    if len(batch) == 6:
        x_map, x_phys, y, init_month, target_years, segment_id = batch
    elif len(batch) == 5:
        x_map, y, init_month, target_years, segment_id = batch
        x_phys = None
    else:
        raise ValueError(f"Unexpected batch length: {len(batch)}")
    return x_map, x_phys, y, init_month, target_years, segment_id


def _is_unexpected_keyword(exc: TypeError, keyword: str) -> bool:
    message = str(exc)
    return keyword in message and (
        "unexpected keyword" in message or "invalid keyword" in message
    )


def model_forward(
    model,
    x_map,
    init_month,
    x_phys=None,
    target_years=None,
    domain_id=None,
    return_diagnostics=False,
    **model_kwargs,
):
    """Call new domain-aware models without breaking the legacy model API."""
    kwargs = {
        "x_phys": x_phys,
        "init_month": init_month,
        "target_years": target_years,
        "return_diagnostics": return_diagnostics,
    }
    if domain_id is not None:
        kwargs["domain_id"] = domain_id
    kwargs.update(model_kwargs)
    try:
        return model(x_map, **kwargs)
    except TypeError as exc:
        if domain_id is not None and _is_unexpected_keyword(exc, "domain_id"):
            kwargs.pop("domain_id", None)
            try:
                return model(x_map, **kwargs)
            except TypeError as legacy_exc:
                exc = legacy_exc
        if _is_unexpected_keyword(exc, "x_phys") or _is_unexpected_keyword(exc, "return_diagnostics"):
            return model(x_map, init_month, target_years=target_years)
        raise exc


def _split_model_output(output) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
    """Normalize tensor and ``(prediction, diagnostics)`` model outputs."""
    if torch.is_tensor(output):
        return output, None
    if isinstance(output, (tuple, list)) and len(output) >= 1:
        diag = output[1] if len(output) >= 2 and isinstance(output[1], dict) else None
        return output[0], diag
    raise TypeError(f"Unsupported model output type: {type(output)!r}")


def _loss_diagnostics(model, returned_diag, domain_id):
    """Obtain loss diagnostics and guarantee that the current domain ids exist."""
    base = getattr(model, "_orig_mod", model)
    model_diag = getattr(base, "loss_diagnostics", None)
    if not isinstance(model_diag, dict) and not isinstance(returned_diag, dict):
        return None
    diag = dict(model_diag) if isinstance(model_diag, dict) else {}
    if isinstance(returned_diag, dict):
        diag.update(returned_diag)
    if domain_id is not None:
        diag.setdefault("domain_id", domain_id)
    return diag


def build_model(args):
    model_dict = get_model_dict()
    if args.model_name not in model_dict:
        raise KeyError(f"Unknown model_name={args.model_name}. Available: {sorted(model_dict)}")
    return model_dict[args.model_name](args)


def build_criterion_v2(args):
    """Build the shared deterministic forecast loss."""
    if args.target_dim >= 2 or args.model_name in MIST_MODELS:
        criterion_class = ENSOPathTraceLoss if args.model_name in PATH_TRACE_MODELS else ENSOCTMLossV2
        pathtrace_kwargs = {}
        if args.model_name in PATH_TRACE_MODELS:
            pathtrace_kwargs = {
                "pathtrace_sparsity_weight": getattr(args, "pathtrace_sparsity_weight", 0.010),
                "pathtrace_coverage_weight": getattr(args, "pathtrace_coverage_weight", 0.020),
                "pathtrace_convergence_weight": getattr(args, "pathtrace_convergence_weight", 0.010),
                "pathtrace_update_weight": getattr(args, "pathtrace_update_weight", 0.001),
                "pathtrace_min_coverage": getattr(args, "pathtrace_min_coverage", 0.10),
                "pathtrace_probe_weight": getattr(args, "pathtrace_probe_weight", 0.0),
                "pathtrace_refinement_weight": getattr(args, "pathtrace_refinement_weight", 0.0),
                "pathtrace_contraction_weight": getattr(args, "pathtrace_contraction_weight", 0.0),
                "pathtrace_contraction_target": getattr(args, "pathtrace_contraction_target", 0.35),
                "pathtrace_message_weight": getattr(args, "pathtrace_message_weight", 0.0),
            }
        return criterion_class(
            aux_weight=args.aux_weight,
            aux_taper_start=args.aux_taper_start,
            aux_taper_end=args.aux_taper_end,
            aux_taper_min=getattr(args, "aux_taper_min", 0.05),
            spb_weight=args.spb_weight,
            spb_lead_start=getattr(args, "spb_lead_start", 2),
            spb_lead_end=getattr(args, "spb_lead_end", 16),
            spring_init_weight=getattr(args, "spring_init_weight", 0.0),
            spring_init_lead_start=getattr(args, "spring_init_lead_start", 12),
            spring_init_lead_end=getattr(args, "spring_init_lead_end", 18),
            spring_init_warmup_epochs=getattr(args, "spring_init_warmup_epochs", 8),
            ode_weight=args.ode_weight,
            ode_lead_start=getattr(args, "ode_lead_start", 8),
            smoothness_weight=args.smoothness_weight,
            trend_weight=getattr(args, "trend_weight", 0.03),
            adaptive_weight=args.adaptive_weight,
            adaptive_ema=getattr(args, "adaptive_ema", 0.95),
            max_adaptive_scale=getattr(args, "max_adaptive_scale", 1.20),
            nll_weight=getattr(args, "nll_weight", 0.10),
            nll_warmup_epochs=getattr(args, "nll_warmup_epochs", 10),
            amplitude_weight=getattr(args, "amplitude_weight", 0.08),
            amplitude_threshold=getattr(args, "amplitude_threshold", 1.0),
            corr_weight=getattr(args, "corr_weight", 0.05),
            corr_lead_start=getattr(args, "corr_lead_start", 6),
            batch_corr_weight=getattr(args, "batch_corr_weight", 0.0),
            batch_corr_lead_start=getattr(args, "batch_corr_lead_start", 6),
            batch_corr_lead_end=getattr(args, "batch_corr_lead_end", 18),
            batch_corr_warmup_epochs=getattr(args, "batch_corr_warmup_epochs", 8),
            long_lead_weight=getattr(args, "long_lead_weight", 0.0),
            long_lead_start=getattr(args, "long_lead_start", 12),
            long_lead_end=getattr(args, "long_lead_end", 18),
            scale_weight=getattr(args, "scale_weight", 0.0),
            scale_lead_start=getattr(args, "scale_lead_start", 12),
            scale_lead_end=getattr(args, "scale_lead_end", 18),
            scale_warmup_epochs=getattr(args, "scale_warmup_epochs", 8),
            phase_weight=getattr(args, "phase_weight", 0.0),
            phase_lead_start=getattr(args, "phase_lead_start", 12),
            phase_lead_end=getattr(args, "phase_lead_end", 18),
            phase_threshold=getattr(args, "phase_threshold", 0.5),
            phase_margin=getattr(args, "phase_margin", 0.25),
            phase_warmup_epochs=getattr(args, "phase_warmup_epochs", 8),
            phase_drift_weight=getattr(args, "phase_drift_weight", 0.0),
            phase_drift_lead_start=getattr(args, "phase_drift_lead_start", 9),
            phase_drift_lead_end=getattr(args, "phase_drift_lead_end", 18),
            phase_drift_warmup_epochs=getattr(args, "phase_drift_warmup_epochs", 8),
            seasonal_mean_weight=getattr(args, "seasonal_mean_weight", 0.0),
            seasonal_mean_lead_start=getattr(args, "seasonal_mean_lead_start", 12),
            seasonal_mean_lead_end=getattr(args, "seasonal_mean_lead_end", 18),
            seasonal_mean_warmup_epochs=getattr(args, "seasonal_mean_warmup_epochs", 8),
            selective_weight=getattr(args, "selective_weight", 0.0),
            selective_lead_start=getattr(args, "selective_lead_start", 12),
            selective_lead_end=getattr(args, "selective_lead_end", 18),
            selective_min_confidence=getattr(args, "selective_min_confidence", 0.35),
            selective_warmup_epochs=getattr(args, "selective_warmup_epochs", 8),
            scale_target_ratio=getattr(args, "scale_target_ratio", 1.0),
            scale_slope_target=getattr(args, "scale_slope_target", 1.0),
            ponder_weight=getattr(args, "ponder_weight", 0.01),
            halt_entropy_weight=getattr(args, "halt_entropy_weight", 0.05),
            overshoot_weight=getattr(args, "overshoot_weight", 0.05),
            pinn_weight=getattr(args, "pinn_weight", 0.0),
            pinn_energy_weight=getattr(args, "pinn_energy_weight", 0.0),
            pinn_warmup_epochs=getattr(args, "pinn_warmup_epochs", 8),
            pinn_prior_weight=getattr(args, "pinn_prior_weight", 0.0),
            latent_anchor_weight=getattr(args, "latent_anchor_weight", 0.0),
            cf_weight=getattr(args, "cf_weight", 0.0),
            cf_warmup_epochs=getattr(args, "cf_warmup_epochs", 6),
            gate_l1_weight=getattr(args, "gate_l1_weight", 0.0),
            mist_domain_residual_weight=getattr(args, "mist_domain_residual_weight", 0.0),
            mist_phase_condition_weight=getattr(args, "mist_phase_condition_weight", 0.0),
            mist_phase_condition_target=getattr(args, "mist_phase_condition_target", 10.0),
            mist_innovation_weight=getattr(args, "mist_innovation_weight", 0.0),
            mist_core_consistency_weight=getattr(args, "mist_core_consistency_weight", 0.0),
            mist_core_only_weight=getattr(args, "mist_core_only_weight", 0.0),
            mist_group_dro_weight=getattr(args, "mist_group_dro_weight", 0.0),
            mist_group_dro_temperature=getattr(args, "mist_group_dro_temperature", 0.10),
            mist_group_dro_mode=getattr(args, "mist_group_dro_mode", "logsumexp"),
            mist_loss_warmup_epochs=getattr(args, "mist_loss_warmup_epochs", 8),
            mist_history_observed_weight=getattr(
                args, "mist_history_observed_weight", 0.0
            ),
            mist_history_latent_weight=getattr(
                args, "mist_history_latent_weight", 0.0
            ),
            mist_history_core_weight=getattr(
                args, "mist_history_core_weight", 0.0
            ),
            mist_history_seasonal_dro_weight=getattr(
                args, "mist_history_seasonal_dro_weight", 0.0
            ),
            mist_history_seasonal_dro_temperature=getattr(
                args, "mist_history_seasonal_dro_temperature", 0.10
            ),
            mist_horizon_curriculum_epochs=getattr(
                args, "mist_horizon_curriculum_epochs", 0
            ),
            mist_horizon_start=getattr(args, "mist_horizon_start", 6),
            huber_beta=getattr(args, "huber_beta", 0.5),
            **pathtrace_kwargs,
        )
    return torch.nn.SmoothL1Loss(beta=0.5)


def criterion_forward_v2(criterion, pred, target, init_month=None, model_diag=None):
    if isinstance(criterion, ENSOCTMLossV2):
        return criterion(pred, target, init_month=init_month, model_diag=model_diag)
    return criterion(pred, target)


def evaluate_loader(model, loader, device, args, return_loss=False, criterion=None, epoch=0):
    model.eval()
    preds_all, trues_all = [], []
    total_loss, n_batches = 0.0, 0
    base = getattr(model, "_orig_mod", model)
    dev_type = "cuda" if getattr(device, "type", str(device)) == "cuda" else "cpu"
    use_amp = bool(getattr(args, "amp", False) and dev_type == "cuda")
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc="  Evaluate", leave=False)):
            max_eval_batches = int(getattr(args, "max_eval_batches", 0))
            if max_eval_batches > 0 and batch_idx >= max_eval_batches:
                break
            x_map, x_phys, y, init_month, target_years, domain_id = unpack_batch(batch)
            x_map = x_map.to(device, non_blocking=True)
            x_phys = None if x_phys is None else x_phys.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            init_month = init_month.to(device, non_blocking=True)
            target_years = None if target_years is None else target_years.to(device, non_blocking=True)
            domain_id = None if domain_id is None else domain_id.to(
                device=device, dtype=torch.long, non_blocking=True
            )
            with torch.autocast(device_type=dev_type, dtype=torch.bfloat16, enabled=use_amp):
                output = model_forward(
                    model, x_map, init_month, x_phys=x_phys,
                    target_years=target_years, domain_id=domain_id,
                )
                pred, returned_diag = _split_model_output(output)
            pred = pred.float()
            if not torch.isfinite(pred).all():
                finite = pred[torch.isfinite(pred)]
                finite_range = (
                    (float(finite.min().item()), float(finite.max().item()))
                    if finite.numel() else (float("nan"), float("nan"))
                )
                raise FloatingPointError(
                    "Non-finite prediction during evaluation at "
                    f"batch={batch_idx}; finite_range={finite_range}"
                )
            if not torch.isfinite(y).all():
                raise FloatingPointError(
                    f"Non-finite target during evaluation at batch={batch_idx}"
                )
            preds_all.append(pred.detach().cpu())
            trues_all.append(y.detach().cpu())
            if return_loss and criterion is not None:
                diag = _loss_diagnostics(base, returned_diag, domain_id)
                loss = criterion_forward_v2(criterion, pred, y, init_month=init_month, model_diag=diag)
                if args.model_name in CTM_MODELS and args.ctm_thought_loss_weight > 0:
                    eval_curriculum = 1.0
                    loss = loss + args.ctm_thought_loss_weight * ctm_thought_loss_v2(
                        diag, y, curriculum_tick_frac=eval_curriculum
                    )
                total_loss += float(loss.item())
                n_batches += 1
    if not preds_all:
        raise RuntimeError("Evaluation loader produced no batches")
    preds = torch.cat(preds_all, dim=0).numpy()
    trues = torch.cat(trues_all, dim=0).numpy()
    if return_loss:
        return preds, trues, total_loss / max(n_batches, 1)
    return preds, trues


def compute_nino_metrics(preds_state, trues_state, stats, eval_output_len=None):
    if eval_output_len and eval_output_len > 0:
        n = min(int(eval_output_len), preds_state.shape[1], trues_state.shape[1])
        preds_state = preds_state[:, :n]
        trues_state = trues_state[:, :n]
    pred_n = preds_state[..., 0]
    true_n = trues_state[..., 0]
    nino_mean = float(stats.get("nino34_mean", stats.get("target:nino34:mean", 0.0)))
    nino_std = float(stats.get("nino34_std", stats.get("target:nino34:std", 1.0)))
    pred_phys = pred_n * nino_std + nino_mean
    true_phys = true_n * nino_std + nino_mean
    acc_raw = compute_acc_per_lead(pred_phys, true_phys)
    acc_3ma = compute_acc_3ma(pred_phys, true_phys)
    rmse = np.sqrt(np.nanmean((pred_phys - true_phys) ** 2, axis=0))
    eff_lead = compute_effective_lead(acc_3ma)
    lead18_index = 17 if len(acc_3ma) > 17 else None
    long_start = 11 if len(acc_3ma) > 11 else None
    long_end = min(18, len(acc_3ma)) if long_start is not None else None
    return {
        "mean_acc_raw": float(np.nanmean(acc_raw)),
        "mean_acc_3ma": float(np.nanmean(acc_3ma)),
        "mean_rmse": float(np.nanmean(rmse)),
        "effective_lead": int(eff_lead),
        "lead18_acc_raw": float(acc_raw[lead18_index]) if lead18_index is not None else float("nan"),
        "lead18_acc_3ma": float(acc_3ma[lead18_index]) if lead18_index is not None else float("nan"),
        "lead18_rmse": float(rmse[lead18_index]) if lead18_index is not None else float("nan"),
        "mean_acc_3ma_12_18": (
            float(np.nanmean(acc_3ma[long_start:long_end]))
            if long_start is not None else float("nan")
        ),
        "mean_rmse_12_18": (
            float(np.nanmean(rmse[long_start:long_end]))
            if long_start is not None else float("nan")
        ),
        "acc_raw": acc_raw,
        "acc_3ma": acc_3ma,
        "rmse": rmse,
    }


def format_epoch_skill(tag, metrics, max_leads=24):
    eff = metrics["effective_lead"]
    return f"{tag} mean_ACC_3MA={metrics['mean_acc_3ma']:.4f}, eff_lead={eff}mo, mean_RMSE={metrics['mean_rmse']:.4f}"


def save_state_target_skill(preds, trues, stats, vis_dir):
    """Save per-target prediction skill info."""
    os.makedirs(vis_dir, exist_ok=True)


def _checkpoint_arg(preloaded_ckpt, name, default=None):
    if preloaded_ckpt is None:
        return default
    saved_args = preloaded_ckpt.get("args")
    if saved_args is None:
        return default
    if isinstance(saved_args, dict):
        return saved_args.get(name, default)
    return getattr(saved_args, name, default)


def _saved_base_phys_vars(preloaded_ckpt):
    saved = _checkpoint_arg(preloaded_ckpt, "loaded_phys_vars")
    if saved is not None:
        return resolve_phys_vars("none", saved) if isinstance(saved, str) else list(saved)
    feature_names = _checkpoint_arg(preloaded_ckpt, "phys_feature_names")
    if feature_names is None:
        return None
    calendar_names = set(CALENDAR_FEATURE_NAMES)
    return [name for name in feature_names if name not in calendar_names]


def _global_segment_domain_map(selected_segments, all_segments) -> Dict[int, int]:
    """Map a split dataset's local segment ids back to original CMIP ids."""
    mapping: Dict[int, int] = {}
    for local_id, (start, end) in enumerate(selected_segments):
        matches = [
            global_id
            for global_id, (full_start, full_end) in enumerate(all_segments)
            if int(full_start) <= int(start) and int(end) <= int(full_end)
        ]
        if len(matches) != 1:
            raise ValueError(
                "Could not uniquely map split segment to its original CMIP domain: "
                f"segment={(start, end)}, matches={matches}"
            )
        mapping[local_id] = matches[0]
    return mapping


def build_datasets(args, preloaded_ckpt=None):
    """Build train/val/test datasets with optional OBS mixed training.

    When obs_train_end_year > obs_start_year, OBS months before that year are
    added to training alongside CMIP6. Test uses only months from obs_test_start_year onward.
    """
    stats_override = None
    if preloaded_ckpt is not None:
        stats_override = preloaded_ckpt.get("stats", None)

    obs_domain_id = int(getattr(args, "mist_obs_domain_id", 16))
    if obs_domain_id < 0:
        raise ValueError("--mist_obs_domain_id must be non-negative")
    args.mist_obs_domain_id = obs_domain_id

    map_vars_csv = getattr(args, "map_vars_csv", None)
    phys_vars_csv = getattr(args, "phys_vars_csv", None)
    explicit_phys_selection = phys_vars_csv is not None or bool(args.no_phys)

    requested_map_vars = resolve_map_vars(args.var_group, map_vars_csv)
    saved_map_vars = _checkpoint_arg(preloaded_ckpt, "map_vars")
    if saved_map_vars is not None:
        saved_map_vars = (
            resolve_map_vars(args.var_group, saved_map_vars)
            if isinstance(saved_map_vars, str)
            else list(saved_map_vars)
        )
        if map_vars_csv is not None and requested_map_vars != saved_map_vars:
            raise ValueError(
                "--map_vars must match the checkpoint channel order. "
                f"checkpoint={saved_map_vars}, requested={requested_map_vars}"
            )
        args.map_vars = saved_map_vars
    else:
        args.map_vars = requested_map_vars

    requested_phys_vars = resolve_phys_vars(
        args.phys_group, phys_vars_csv, args.no_phys
    )
    saved_phys_vars = _saved_base_phys_vars(preloaded_ckpt)
    if saved_phys_vars is not None:
        if explicit_phys_selection and requested_phys_vars != saved_phys_vars:
            raise ValueError(
                "--phys_vars/--no_phys must match the checkpoint feature order. "
                f"checkpoint={saved_phys_vars}, requested={requested_phys_vars}"
            )
        args.phys_vars_requested = saved_phys_vars
    else:
        args.phys_vars_requested = requested_phys_vars

    saved_target_vars = _checkpoint_arg(preloaded_ckpt, "target_vars")
    if saved_target_vars is not None:
        args.target_vars = (
            resolve_target_vars(saved_target_vars)
            if isinstance(saved_target_vars, str)
            else list(saved_target_vars)
        )
    else:
        args.target_vars = (
            resolve_target_vars(args.target_vars)
            if isinstance(args.target_vars, str)
            else list(args.target_vars)
        )

    saved_no_calendar = _checkpoint_arg(preloaded_ckpt, "no_calendar_features")
    if saved_no_calendar is not None:
        if args.no_calendar_features and not bool(saved_no_calendar):
            raise ValueError(
                "--no_calendar_features does not match the checkpoint input features."
            )
        args.no_calendar_features = bool(saved_no_calendar)

    strict_phys = explicit_phys_selection or saved_phys_vars is not None
    print(f"[Config] map_vars ({len(args.map_vars)})={args.map_vars}")
    print(
        f"[Config] requested_phys_vars ({len(args.phys_vars_requested)})="
        f"{args.phys_vars_requested}"
    )

    def _parse_years(value):
        if not value:
            return None
        ys = [int(tok) for tok in str(value).replace(";", ",").split(",") if tok.strip()]
        return ys or None
    exclude_event_years = _parse_years(getattr(args, "exclude_event_years", ""))
    event_test_years = _parse_years(getattr(args, "event_test_years", ""))
    if exclude_event_years:
        print(f"[E1] Excluding strongest-event years from ALL training data: {exclude_event_years}")
    if event_test_years:
        print(f"[E1] Test set restricted to event years (OOD): {event_test_years}")

    cmip_data = None
    if args.stage == "train":
        print(f"[Data] CMIP6: {args.cmip_path}")
        cmip_data = load_mechanism_file(
            args.cmip_path, args.map_vars, args.phys_vars_requested,
            args.target_vars, args.lat_south, args.lat_north,
            strict_phys=strict_phys,
        )
        train_segments, val_segments = split_segments(
            cmip_data.segments,
            args.val_months_per_model,
            args.input_len + args.output_len,
        )
        num_cmip_domains = len(cmip_data.segments)
        if args.model_name in MIST_MODELS and obs_domain_id < num_cmip_domains:
            raise ValueError(
                f"OBS domain id {obs_domain_id} collides with {num_cmip_domains} CMIP domains; "
                f"set --mist_obs_domain_id >= {num_cmip_domains}."
            )
        train_domain_map = _global_segment_domain_map(train_segments, cmip_data.segments)
        val_domain_map = _global_segment_domain_map(val_segments, cmip_data.segments)
    else:
        train_segments, val_segments = None, None
        train_domain_map, val_domain_map = None, None
        num_cmip_domains = int(
            _checkpoint_arg(
                preloaded_ckpt,
                "mist_num_cmip_domains",
                max(obs_domain_id, int(getattr(args, "mist_num_domains", obs_domain_id + 1)) - 1),
            )
        )

    args.mist_num_cmip_domains = num_cmip_domains
    args.mist_num_domains = max(
        17,
        int(getattr(args, "mist_num_domains", 17)),
        num_cmip_domains,
        obs_domain_id + 1,
    )
    print(
        f"[Domain] CMIP ids=0..{max(num_cmip_domains - 1, 0)}, "
        f"OBS id={obs_domain_id}, total adapter slots={args.mist_num_domains}"
    )

    print(f"[Data] OBS: {args.obs_path}")
    obs_data = load_mechanism_file(
        args.obs_path, args.map_vars, args.phys_vars_requested,
        args.target_vars, args.lat_south, args.lat_north,
        strict_phys=strict_phys,
    )

    if cmip_data is not None:
        common_phys = [
            v for v in args.phys_vars_requested
            if v in cmip_data.physics and v in obs_data.physics
        ]
        dropped_phys = [v for v in args.phys_vars_requested if v not in common_phys]
        if dropped_phys:
            print(f"[Warning] physics indices dropped outside the CMIP/OBS intersection: {dropped_phys}")
        cmip_data.phys_vars = common_phys
        cmip_data.physics = {k: cmip_data.physics[k] for k in common_phys}
        obs_data.phys_vars = common_phys
        obs_data.physics = {k: obs_data.physics[k] for k in common_phys}
        print(f"[Config] common_phys_vars ({len(common_phys)})={common_phys[:10]}...")

    if args.stage == "train":
        cmip_train_ds = ENSOGlobalMechanismDataset(
            cmip_data, args.input_len, args.output_len, is_train=True, stats=None,
            segments=train_segments, start_month=args.cmip_start_month - 1,
            start_year=args.cmip_start_year, phys_scaler=args.phys_scaler,
            include_calendar_features=not args.no_calendar_features, map_noise=not args.no_map_noise,
            spatial_mask_spec=args.spatial_mask_spec,
        )
        stats = cmip_train_ds.get_stats()
        cmip_val_ds = ENSOGlobalMechanismDataset(
            cmip_data, args.input_len, args.output_len, is_train=False, stats=stats,
            segments=val_segments, start_month=args.cmip_start_month - 1,
            start_year=args.cmip_start_year, phys_scaler=args.phys_scaler,
            include_calendar_features=not args.no_calendar_features, map_noise=False,
            spatial_mask_spec=args.spatial_mask_spec,
        )
        train_ds = DomainIdDataset(cmip_train_ds, domain_map=train_domain_map)
        val_ds = DomainIdDataset(cmip_val_ds, domain_map=val_domain_map)

        obs_train_end_year = int(getattr(args, "obs_train_end_year", 0))
        obs_test_start_year = int(getattr(args, "obs_test_start_year", 0))
        if obs_test_start_year <= 0:
            obs_test_start_year = args.obs_start_year
        if obs_train_end_year > obs_test_start_year:
            raise ValueError(
                "--obs_train_end_year must not exceed --obs_test_start_year"
            )
        if obs_train_end_year > args.obs_start_year:
            obs_train_end_month = (obs_train_end_year - args.obs_start_year) * 12
            total_obs_months = next(iter(obs_data.maps.values())).shape[0]
            obs_train_end_month = min(obs_train_end_month, total_obs_months)
            obs_train_segments = [(0, obs_train_end_month)]

            obs_train_ds = ENSOGlobalMechanismDataset(
                obs_data, args.input_len, args.output_len, is_train=True, stats=stats,
                segments=obs_train_segments, start_month=args.obs_start_month - 1,
                start_year=args.obs_start_year, phys_scaler=args.phys_scaler,
                include_calendar_features=not args.no_calendar_features, map_noise=not args.no_map_noise,
                exclude_target_years=exclude_event_years,
                spatial_mask_spec=args.spatial_mask_spec,
            )
            obs_train_ds = DomainIdDataset(obs_train_ds, domain_id=obs_domain_id)
            n_cmip = len(train_ds)
            n_obs = len(obs_train_ds)

            obs_weight = getattr(args, "obs_train_weight", 3)
            obs_repeated = ConcatDataset([obs_train_ds] * int(obs_weight))
            train_ds = ConcatDataset([train_ds, obs_repeated])
            print(f"[Data] Mixed training: CMIP={n_cmip} + OBS_train={n_obs}×{obs_weight}={n_obs*int(obs_weight)} "
                  f"(OBS fraction={n_obs*int(obs_weight)/(n_cmip+n_obs*int(obs_weight)):.1%})")

            obs_test_start_month = (obs_test_start_year - args.obs_start_year) * 12
            obs_test_end_year = int(getattr(args, "obs_test_end_year", 0))
            if obs_test_end_year > obs_test_start_year:
                obs_test_end_month = min((obs_test_end_year - args.obs_start_year) * 12, total_obs_months)
            else:
                obs_test_end_month = total_obs_months
            obs_test_segments = [(obs_test_start_month, obs_test_end_month)]
        else:
            obs_test_start_year = args.obs_start_year
            obs_test_end_year = int(getattr(args, "obs_test_end_year", 0))
            total_obs_months = next(iter(obs_data.maps.values())).shape[0]
            if obs_test_end_year > obs_test_start_year:
                obs_test_end_month = min((obs_test_end_year - args.obs_start_year) * 12, total_obs_months)
                obs_test_segments = [(0, obs_test_end_month)]
            else:
                obs_test_segments = obs_data.segments

        obs_extra_path = getattr(args, "obs_extra_train_path", "")
        if obs_extra_path and os.path.exists(obs_extra_path):
            print(f"[Data] Extra OBS train: {obs_extra_path}")
            obs_extra_data = load_mechanism_file(
                obs_extra_path, args.map_vars, args.phys_vars_requested, args.target_vars,
                args.lat_south, args.lat_north, strict_phys=False
            )
            obs_extra_data.phys_vars = [v for v in obs_extra_data.phys_vars if v in common_phys]
            obs_extra_data.physics = {k: obs_extra_data.physics[k] for k in obs_extra_data.phys_vars}
            for v in common_phys:
                if v not in obs_extra_data.physics:
                    n_months = next(iter(obs_extra_data.maps.values())).shape[0]
                    obs_extra_data.physics[v] = np.zeros(n_months, dtype=np.float32)
            obs_extra_data.phys_vars = list(common_phys)
            obs_extra_data.physics = {k: obs_extra_data.physics[k] for k in common_phys}

            obs_extra_ds = ENSOGlobalMechanismDataset(
                obs_extra_data, args.input_len, args.output_len, is_train=True, stats=stats,
                segments=obs_extra_data.segments,
                start_month=int(getattr(args, "obs_extra_start_month", 1)) - 1,
                start_year=int(getattr(args, "obs_extra_start_year", 1958)),
                phys_scaler=args.phys_scaler,
                include_calendar_features=not args.no_calendar_features, map_noise=not args.no_map_noise,
                exclude_target_years=exclude_event_years,
                spatial_mask_spec=args.spatial_mask_spec,
            )
            obs_extra_ds = DomainIdDataset(obs_extra_ds, domain_id=obs_domain_id)
            n_extra = len(obs_extra_ds)
            obs_weight_extra = getattr(args, "obs_train_weight", 3)
            obs_extra_repeated = ConcatDataset([obs_extra_ds] * int(obs_weight_extra))
            train_ds = ConcatDataset([train_ds, obs_extra_repeated])
            print(f"[Data] Extra OBS: {n_extra}×{obs_weight_extra}={n_extra*int(obs_weight_extra)} samples added "
                  f"(total train={len(train_ds)})")

        test_ds = ENSOGlobalMechanismDataset(
            obs_data, args.input_len, args.output_len, is_train=False, stats=stats,
            segments=obs_test_segments, start_month=args.obs_start_month - 1,
            start_year=args.obs_start_year, phys_scaler=args.phys_scaler,
            include_calendar_features=not args.no_calendar_features, map_noise=False,
            keep_only_target_years=event_test_years,
            spatial_mask_spec=args.spatial_mask_spec,
        )
        test_ds = DomainIdDataset(test_ds, domain_id=obs_domain_id)
        test_end_label = obs_test_end_year if obs_test_end_year > obs_test_start_year else "end"
        print(f"[Data] Test: OBS {obs_test_start_year}-{test_end_label}, samples={len(test_ds)}")
    else:
        train_ds = val_ds = None
        stats = None
        obs_test_start_year = int(getattr(args, "obs_test_start_year", 0))
        obs_test_end_year = int(getattr(args, "obs_test_end_year", 0))
        if obs_test_start_year <= 0:
            obs_test_start_year = args.obs_start_year
        if obs_test_start_year > args.obs_start_year:
            total_obs_months = next(iter(obs_data.maps.values())).shape[0]
            obs_test_start_month = (obs_test_start_year - args.obs_start_year) * 12
            if obs_test_end_year > obs_test_start_year:
                obs_test_end_month = min((obs_test_end_year - args.obs_start_year) * 12, total_obs_months)
            else:
                obs_test_end_month = total_obs_months
            obs_test_segments = [(obs_test_start_month, obs_test_end_month)]
        else:
            total_obs_months = next(iter(obs_data.maps.values())).shape[0]
            if obs_test_end_year > args.obs_start_year:
                obs_test_end_month = min((obs_test_end_year - args.obs_start_year) * 12, total_obs_months)
                obs_test_segments = [(0, obs_test_end_month)]
            else:
                obs_test_segments = obs_data.segments
        test_ds = ENSOGlobalMechanismDataset(
            obs_data, args.input_len, args.output_len, is_train=False, stats=stats_override,
            segments=obs_test_segments, start_month=args.obs_start_month - 1,
            start_year=args.obs_start_year, phys_scaler=args.phys_scaler,
            include_calendar_features=not args.no_calendar_features, map_noise=False,
            keep_only_target_years=event_test_years,
            spatial_mask_spec=args.spatial_mask_spec,
        )
        test_ds = DomainIdDataset(test_ds, domain_id=obs_domain_id)

    ref_ds = val_ds if val_ds is not None else test_ds
    args.input_dim = len(args.map_vars)
    args.n_vars = len(args.map_vars)
    args.img_height, args.img_width = ref_ds.spatial_shape
    args.lat_coords = ref_ds.lat
    args.lon_coords = ref_ds.lon
    args.phys_feature_names = ref_ds.phys_feature_names
    args.phys_dim = ref_ds.phys_dim
    args.base_phys_dim = len(ref_ds.base_phys_vars)
    args.loaded_phys_vars = list(ref_ds.base_phys_vars)
    args.target_dim = ref_ds.target_dim
    args.state_scale_info = ref_ds.get_state_scale_info()
    target_phys_indices = []
    target_phys_scale = []
    target_phys_bias = []
    dataset_stats = ref_ds.get_stats()
    for target_name in args.target_vars:
        if target_name not in args.phys_feature_names:
            target_phys_indices.append(-1)
            target_phys_scale.append(1.0)
            target_phys_bias.append(0.0)
            continue
        target_phys_indices.append(args.phys_feature_names.index(target_name))
        phys_center = float(dataset_stats.get(f"phys:{target_name}:center", 0.0))
        phys_scale = float(dataset_stats.get(f"phys:{target_name}:scale", 1.0))
        target_mean = float(dataset_stats.get(f"target:{target_name}:mean", 0.0))
        target_std = max(
            float(dataset_stats.get(f"target:{target_name}:std", 1.0)), 1.0e-6
        )
        target_phys_scale.append(phys_scale / target_std)
        target_phys_bias.append((phys_center - target_mean) / target_std)
    args.target_phys_indices = target_phys_indices
    args.target_phys_scale = target_phys_scale
    args.target_phys_bias = target_phys_bias
    if args.model_name in {"MIST_ENSO_V2", "MIST-ENSO-V2", "MISTENSO_V2"}:
        missing_anchors = [
            name for name, index in zip(args.target_vars, target_phys_indices)
            if index < 0
        ]
        if missing_anchors:
            raise ValueError(
                "MIST V2 requires target histories among --phys_vars; missing "
                f"{missing_anchors}"
            )
        print(
            f"[MIST V2] observable anchors={args.target_vars}, "
            f"phys_indices={target_phys_indices}, scale={target_phys_scale}, "
            f"bias={target_phys_bias}"
        )
    saved_phys_feature_names = _checkpoint_arg(preloaded_ckpt, "phys_feature_names")
    if saved_phys_feature_names is not None:
        saved_phys_feature_names = list(saved_phys_feature_names)
        if args.phys_feature_names != saved_phys_feature_names:
            raise ValueError(
                "Loaded physical feature order does not match checkpoint. "
                f"checkpoint={saved_phys_feature_names}, loaded={args.phys_feature_names}"
            )
    return train_ds, val_ds, test_ds, stats


def build_scheduler(optimizer, args, steps_per_epoch: int):
    """Warmup + cosine annealing schedule."""
    warmup_steps = args.warmup_epochs * steps_per_epoch
    total_steps = args.epochs * steps_per_epoch
    min_lr = args.learning_rate * args.min_lr_ratio

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
        return max(min_lr / args.learning_rate, cosine)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _restore_mist_checkpoint_args(args, checkpoint) -> None:
    """Restore model-specific architecture and generic sequence shape for tests.

    Runtime options such as paths, batch size and output directories remain
    controlled by the current command.  Model identity and sequence lengths,
    however, must follow the checkpoint or an otherwise valid ``--stage test``
    command can instantiate the default CTM or build windows of the wrong size.
    """
    if checkpoint is None:
        return
    saved_args = checkpoint.get("args", None)
    if saved_args is None:
        return
    saved = saved_args if isinstance(saved_args, dict) else vars(saved_args)
    generic_model_args = {
        "model_name",
        "input_len",
        "output_len",
        "var_group",
        "map_vars_csv",
        "phys_group",
        "phys_vars_csv",
        "target_vars",
        "spatial_mask_spec",
    }
    for name, value in saved.items():
        if name.startswith(
            (
                "mist_",
                "pathtrace_",
                "baseline_",
                "geoformer_",
                "gl_",
                "cnn_",
                "convlstm_",
                "atmos_",
            )
        ) or name in generic_model_args:
            setattr(args, name, value)


def _load_model_state_flexible(model, state_dict) -> None:
    """Strictly load plain or ``torch.compile``-prefixed state dictionaries."""

    try:
        model.load_state_dict(state_dict, strict=True)
        return
    except RuntimeError as first_error:
        base_model = getattr(model, "_orig_mod", model)
        has_compile_prefix = any(key.startswith("_orig_mod.") for key in state_dict)
        if has_compile_prefix:
            stripped = {
                key[len("_orig_mod."):] if key.startswith("_orig_mod.") else key: value
                for key, value in state_dict.items()
            }
            base_model.load_state_dict(stripped, strict=True)
            return
        if base_model is not model:
            base_model.load_state_dict(state_dict, strict=True)
            return
        raise first_error


def main(args):
    fix_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = str(device)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.visual_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    exp_name = args.exp_tag or f"{args.model_name}_{args.var_group}_{args.phys_group}"
    ckpt_path = os.path.join(args.save_dir, f"{exp_name}.pth")

    print("=" * 80)
    print(f"Stage={args.stage} | Model={args.model_name} | Device={device}")
    print(f"Input={args.input_len} months -> Output={args.output_len} months")
    print("=" * 80)
    preloaded_ckpt = None
    if args.stage == "test":
        load_path = ckpt_path
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Checkpoint does not exist: {load_path}")
        preloaded_ckpt = torch.load(load_path, map_location=device, weights_only=False)
        _restore_mist_checkpoint_args(args, preloaded_ckpt)

    train_ds, val_ds, test_ds, stats = build_datasets(args, preloaded_ckpt)

    if args.model_name in MIST_MODELS and getattr(args, "mixup_alpha", 0.0) > 0:
        print(
            "[Warning] MIST domain-operator training is incompatible with ordinary "
            "cross-domain mixup; disabling --mixup_alpha."
        )
        args.mixup_alpha = 0.0

    model = build_model(args).to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] {args.model_name}: {param_count / 1e6:.2f}M trainable parameters")

    if getattr(args, "compile", False) and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
            print("[Model] torch.compile enabled")
        except Exception as exc:
            print(f"[Model] torch.compile failed ({exc}); continuing uncompiled")
    base_model = getattr(model, "_orig_mod", model)
    amp_device = "cuda" if device.type == "cuda" else "cpu"
    requested_amp = bool(getattr(args, "amp", False))
    amp_enabled = requested_amp and device.type == "cuda"
    if requested_amp and not amp_enabled:
        print("[Model] AMP requested but CUDA is unavailable; using fp32 on CPU")
    if amp_enabled:
        print(f"[Model] AMP bf16 autocast enabled on {amp_device}")
    args.amp = amp_enabled

    if args.stage == "train":
        criterion = build_criterion_v2(args).to(device)
        optimizer = torch.optim.AdamW(
            list(model.parameters()) + list(criterion.parameters()),
            lr=args.learning_rate, weight_decay=args.weight_decay
        )
        loader_workers = int(args.num_workers)
        loader_common = {
            "num_workers": loader_workers,
            "pin_memory": True,
        }
        if loader_workers > 0:
            loader_common["persistent_workers"] = bool(
                getattr(args, "persistent_workers", False)
            )
            loader_common["prefetch_factor"] = int(getattr(args, "prefetch_factor", 2))

        if args.model_name in MIST_MODELS and getattr(args, "mist_domain_balanced_batches", False):
            train_batch_sampler = DomainBalancedBatchSampler(
                train_ds,
                batch_size=args.batch_size,
                drop_last=True,
                seed=args.seed,
            )
            train_loader = DataLoader(
                train_ds, batch_sampler=train_batch_sampler, **loader_common
            )
            print(
                f"[Domain] Balanced batches enabled across "
                f"{len(train_batch_sampler.pools)} present domains"
            )
        else:
            train_loader = DataLoader(
                train_ds, batch_size=args.batch_size, shuffle=True,
                drop_last=True, **loader_common,
            )
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False, **loader_common
        )
        test_loader = DataLoader(
            test_ds, batch_size=args.batch_size, shuffle=False, **loader_common
        )

        scheduler = build_scheduler(optimizer, args, len(train_loader))
        stopper = EarlyStopping(patience=args.patience, verbose=True, path=ckpt_path)

        epoch_history = []
        epoch_metrics_path = os.path.join(args.save_dir, f"{exp_name}_epoch_metrics.csv")

        for epoch in range(args.epochs):
            base_model.train()
            losses_epoch = []
            curriculum_frac = min(1.0, 0.3 + 0.7 * epoch / max(args.epochs * 0.5, 1))

            if isinstance(criterion, ENSOCTMLossV2):
                criterion.current_epoch.fill_(epoch)

            if hasattr(base_model, "set_epoch"):
                base_model.set_epoch(epoch)

            if hasattr(base_model, "halt_temperature"):
                warmup_ep = getattr(args, "ctm_halt_warmup_epochs", 20)
                target_temp = getattr(args, "ctm_halt_temperature", 1.0)
                if epoch < warmup_ep:
                    frac = epoch / max(warmup_ep, 1)
                    base_model.halt_temperature = 3.0 * (1.0 - frac) + target_temp * frac
                else:
                    base_model.halt_temperature = target_temp

            for batch_idx, batch in enumerate(
                tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False)
            ):
                max_train_batches = int(getattr(args, "max_train_batches", 0))
                if max_train_batches > 0 and batch_idx >= max_train_batches:
                    break
                x_map, x_phys, y, init_month, target_years, domain_id = unpack_batch(batch)
                x_map = x_map.to(device, non_blocking=True)
                x_phys = None if x_phys is None else x_phys.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                init_month = init_month.to(device, non_blocking=True)
                target_years = None if target_years is None else target_years.to(device, non_blocking=True)
                domain_id = None if domain_id is None else domain_id.to(
                    device=device, dtype=torch.long, non_blocking=True
                )

                if args.mixup_alpha > 0 and np.random.rand() < 0.5:
                    lam = np.random.beta(args.mixup_alpha, args.mixup_alpha)
                    lam = max(lam, 1 - lam)
                    idx_perm = torch.randperm(x_map.size(0), device=device)
                    x_map = lam * x_map + (1 - lam) * x_map[idx_perm]
                    if x_phys is not None:
                        x_phys = lam * x_phys + (1 - lam) * x_phys[idx_perm]
                    y = lam * y + (1 - lam) * y[idx_perm]

                if args.label_noise > 0:
                    y = y + args.label_noise * torch.randn_like(y)

                with torch.autocast(device_type=amp_device, dtype=torch.bfloat16,
                                    enabled=getattr(args, "amp", False)):
                    output = model_forward(
                        model, x_map, init_month, x_phys=x_phys,
                        target_years=target_years, domain_id=domain_id,
                    )
                    pred, returned_diag = _split_model_output(output)
                    diag = _loss_diagnostics(base_model, returned_diag, domain_id)
                    loss_components = None
                    if isinstance(criterion, ENSOCTMLossV2):
                        loss_components = criterion.loss_components(
                            pred, y, init_month=init_month, model_diag=diag
                        )
                        loss = loss_components["total"]
                    else:
                        loss = criterion(pred, y)

                    if args.model_name in CTM_MODELS and args.ctm_thought_loss_weight > 0:
                        thought_loss = ctm_thought_loss_v2(diag, y, curriculum_tick_frac=curriculum_frac)
                        loss = loss + args.ctm_thought_loss_weight * thought_loss

                if not torch.isfinite(loss).all():
                    component_report = {}
                    if isinstance(loss_components, dict):
                        for name, value in loss_components.items():
                            if torch.is_tensor(value) and value.numel() == 1:
                                component_report[name] = float(value.detach().float().item())
                    nonfinite_diag = []
                    if isinstance(diag, dict):
                        nonfinite_diag = [
                            name for name, value in diag.items()
                            if torch.is_tensor(value) and not torch.isfinite(value).all()
                        ]
                    raise FloatingPointError(
                        "Non-finite training loss before backward: "
                        f"epoch={epoch + 1}, batch={batch_idx}, "
                        f"components={component_report}, "
                        f"nonfinite_diag={nonfinite_diag}, "
                        f"pred_finite={bool(torch.isfinite(pred).all())}, "
                        f"target_finite={bool(torch.isfinite(y).all())}"
                    )

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if args.grad_clip > 0:
                    try:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), args.grad_clip,
                            error_if_nonfinite=True,
                        )
                    except RuntimeError as exc:
                        bad_gradients = [
                            name for name, parameter in base_model.named_parameters()
                            if parameter.grad is not None
                            and not torch.isfinite(parameter.grad).all()
                        ]
                        raise FloatingPointError(
                            "Non-finite gradient before optimizer step: "
                            f"epoch={epoch + 1}, batch={batch_idx}, "
                            f"parameters={bad_gradients[:20]}"
                        ) from exc
                optimizer.step()
                scheduler.step()
                losses_epoch.append(float(loss.item()))

            avg_train = np.mean(losses_epoch)
            _, _, avg_val = evaluate_loader(model, val_loader, device, args, return_loss=True,
                                           criterion=criterion, epoch=epoch)
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"  Epoch {epoch+1}: train_loss={avg_train:.5f}, val_loss={avg_val:.5f}, lr={lr_now:.2e}, curriculum={curriculum_frac:.2f}")

            epoch_row = {"epoch": epoch + 1, "train_loss": avg_train, "val_loss": avg_val, "lr": lr_now}

            epoch_history.append(epoch_row)
            pd.DataFrame(epoch_history).to_csv(epoch_metrics_path, index=False)
            stopper(avg_val, model, args, stats=stats)
            if stopper.early_stop:
                print("  Early stopping triggered.")
                break

        print(f"[Train] Epoch metrics: {epoch_metrics_path}")

    if args.stage in {"train", "test"}:
        if args.stage == "train":
            load_path = ckpt_path
        else:
            load_path = ckpt_path
        ckpt = preloaded_ckpt if (args.stage == "test" and preloaded_ckpt is not None) else torch.load(load_path, map_location=device, weights_only=False)
        stats = ckpt.get("stats", stats)
        _load_model_state_flexible(model, ckpt["model_state_dict"])
        print(f"[Test] Loaded: {load_path}")
        test_loader_kwargs = {
            "num_workers": int(args.num_workers),
            "pin_memory": True,
        }
        if int(args.num_workers) > 0:
            test_loader_kwargs["persistent_workers"] = bool(
                getattr(args, "persistent_workers", False)
            )
            test_loader_kwargs["prefetch_factor"] = int(getattr(args, "prefetch_factor", 2))
        test_loader = DataLoader(
            test_ds, batch_size=args.batch_size, shuffle=False, **test_loader_kwargs
        )
        preds_state, trues_state = evaluate_loader(model, test_loader, device, args)
        vis_dir = os.path.join(args.visual_dir, f"{exp_name}/{exp_name}_{timestamp}")
        os.makedirs(vis_dir, exist_ok=True)
        preds_nino = preds_state[..., 0]
        trues_nino = trues_state[..., 0]
        acc_raw, acc_3ma, rmse = evaluate_nino34_skill_decay(preds_nino, trues_nino, stats, vis_dir, args)
        save_nino34_to_csv_and_plot(preds_nino, trues_nino, stats, vis_dir, args.output_len, None, args)
        checkpoint_record = {
            "experiment": exp_name,
            "checkpoint_name": os.path.basename(load_path),
            "checkpoint_path": os.path.abspath(load_path),
            "checkpoint_kind": "validation_best",
            "selection": ckpt.get("selection", {}),
            "sample_count": int(preds_state.shape[0]),
            "final_test_metrics": {
                "mean_acc_raw": float(np.nanmean(acc_raw)),
                "mean_acc_3ma": float(np.nanmean(acc_3ma)),
                "mean_rmse": float(np.nanmean(rmse)),
                "effective_lead": int(compute_effective_lead(acc_3ma)),
                "lead18_acc_raw": float(acc_raw[17]) if len(acc_raw) > 17 else float("nan"),
                "lead18_acc_3ma": float(acc_3ma[17]) if len(acc_3ma) > 17 else float("nan"),
                "lead18_rmse": float(rmse[17]) if len(rmse) > 17 else float("nan"),
                "mean_acc_3ma_12_18": (
                    float(np.nanmean(acc_3ma[11:18])) if len(acc_3ma) > 11 else float("nan")
                ),
                "mean_rmse_12_18": (
                    float(np.nanmean(rmse[11:18])) if len(rmse) > 11 else float("nan")
                ),
                "spb_composite": (
                    float(
                        0.60 * acc_3ma[17] + 0.40 * np.nanmean(acc_3ma[11:18])
                    )
                    if len(acc_3ma) > 17 else float("nan")
                ),
            },
        }
        with open(os.path.join(vis_dir, "test_checkpoint.json"), "w", encoding="utf-8") as handle:
            json.dump(checkpoint_record, handle, indent=2, ensure_ascii=False)
        try:
            plot_seasonal_lead_heatmap(preds_nino, trues_nino, stats, vis_dir, args, None)
            plot_init_month_lead_heatmap(preds_nino, trues_nino, stats, vis_dir, args, None)
        except Exception as exc:
            print(f"  [Warning] Heatmap failed: {exc}")
        print(f"[Test] Results: {vis_dir}")
        print(f"  Mean ACC_3MA={np.nanmean(acc_3ma):.4f}, effective_lead={compute_effective_lead(acc_3ma)} months")


def build_parser():
    p = argparse.ArgumentParser(description="ENSO forecasting training")
    p.add_argument("--stage", required=True, choices=["train", "test"])
    p.add_argument("--cmip_path", default="./data/cmip6_historical_1900_2014_global_mechanism.nc")
    p.add_argument("--obs_path", default="./data/obs_1980_2025_global_mechanism.nc")
    p.add_argument("--obs_extra_train_path", default="",
                   help="Additional OBS file for training (e.g. 1958-1978 data). All months are used for training.")
    p.add_argument("--obs_extra_start_month", type=int, default=1)
    p.add_argument("--obs_extra_start_year", type=int, default=1958)
    p.add_argument("--cmip_start_month", type=int, default=1)
    p.add_argument("--cmip_start_year", type=int, default=1900)
    p.add_argument("--obs_start_month", type=int, default=1)
    p.add_argument("--obs_start_year", type=int, default=1980)
    p.add_argument("--obs_train_end_year", type=int, default=0,
                   help="If >obs_start_year, OBS months before this year are added to training. 0=disable.")
    p.add_argument("--obs_test_start_year", type=int, default=0,
                   help="Test only uses OBS from this year onward. 0=use all OBS for test.")
    p.add_argument("--obs_test_end_year", type=int, default=0,
                   help="Test only uses OBS up to (not including) this year. 0=use to end of data.")
    p.add_argument("--obs_train_weight", type=int, default=3,
                   help="Repeat OBS train data this many times to boost its share in batches.")
    p.add_argument("--val_months_per_model", type=int, default=60)
    p.add_argument("--lat_south", type=float, default=-60.0)
    p.add_argument("--lat_north", type=float, default=60.0)

    p.add_argument(
        "--var_group", default="full8", choices=["core4", "core5", "full8"],
        help="Map-variable preset used when --map_vars is not provided.",
    )
    p.add_argument(
        "--map_vars", default=None, dest="map_vars_csv", metavar="VAR1,VAR2,...",
        help="Ordered comma-separated map variables; overrides --var_group.",
    )
    p.add_argument(
        "--phys_group", default="mechanism",
        choices=["none", "core", "recharge", "wind", "cross_basin", "mechanism"],
        help="Physical-index preset used when --phys_vars is not provided.",
    )
    p.add_argument(
        "--phys_vars", default=None, dest="phys_vars_csv", metavar="IDX1,IDX2,...",
        help="Ordered comma-separated physical indices; use '' or 'none' for none.",
    )
    p.add_argument("--no_phys", action="store_true", help="Disable all base physical indices.")
    p.add_argument("--target_vars", default="nino34,thermocline_tilt")
    p.add_argument("--phys_scaler", default="robust", choices=["robust", "standard"])
    p.add_argument("--no_calendar_features", action="store_true")
    p.add_argument("--no_map_noise", action="store_true")
    p.add_argument(
        "--spatial_mask_spec",
        default="full",
        help=(
            "Basin mask applied after map standardization. Examples: full, "
            "five_basins, basins=trop_pacific+indian, "
            "only_variables=slp+sst, "
            "without_group=atlantic__sst, only_group=indian__hc300."
        ),
    )

    p.add_argument("--exclude_event_years", default="",
                   help="E1: comma-separated calendar years whose target windows are "
                        "REMOVED from training (CMIP+OBS). Held-out strongest ENSO years, "
                        "e.g. '1982,1983,1997,1998,2015,2016,2023'.")
    p.add_argument("--event_test_years", default="",
                   help="E1: comma-separated calendar years; when set the TEST set keeps "
                        "only windows whose target covers these years (event-only OOD test).")

    p.add_argument("--save_dir", default="./checkpoints/")
    p.add_argument("--visual_dir", default="./results/")
    p.add_argument("--eval_output_len", type=int, default=None)
    p.add_argument("--exp_tag", default="ENSOCTM_OfficialPINN_v4_full8_mechanism")

    p.add_argument("--model_name", default="ENSOCTM_OfficialPINN_v4")
    p.add_argument("--input_len", type=int, default=12)
    p.add_argument("--output_len", type=int, default=24)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--learning_rate", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=25)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--persistent_workers", action="store_true",
                   help="Keep DataLoader workers alive between epochs (requires num_workers>0).")
    p.add_argument("--prefetch_factor", type=int, default=2,
                   help="Batches prefetched per DataLoader worker.")
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument("--warmup_epochs", type=int, default=10)
    p.add_argument("--min_lr_ratio", type=float, default=0.03)
    p.add_argument("--grad_clip", type=float, default=2.0)
    p.add_argument("--dropout", type=float, default=0.12)
    p.add_argument("--label_noise", type=float, default=0.0,
                   help="Gaussian noise std added to targets during training (regularization).")
    p.add_argument("--mixup_alpha", type=float, default=0.0,
                   help="Beta distribution alpha for mixup augmentation. 0=disabled, 0.2-0.4 recommended.")
    p.add_argument("--max_train_batches", type=int, default=0,
                   help="Smoke-test limit per epoch; 0 uses the complete training loader.")
    p.add_argument("--max_eval_batches", type=int, default=0,
                   help="Smoke-test limit per evaluation loader; 0 uses all batches.")

    p.add_argument("--baseline_d_model", type=int, default=128,
                   help="Hidden size for non-CTM neural baselines.")
    p.add_argument("--baseline_n_heads", type=int, default=4,
                   help="Attention heads for Transformer-style baselines.")
    p.add_argument("--baseline_depth", type=int, default=3,
                   help="Transformer encoder depth for Geoformer-style baseline.")
    p.add_argument("--baseline_ffn_mult", type=int, default=4,
                   help="Feed-forward expansion ratio for Transformer-style baselines.")
    p.add_argument("--baseline_dropout", type=float, default=0.15,
                   help="Dropout used by ENSOCNN/ENSOGeoformer.")
    p.add_argument("--cnn_width", type=int, default=48,
                   help="Base channel width for ENSOCNN.")
    p.add_argument("--cnn_layers", type=int, default=3,
                   help="Number of spatial convolution blocks for ENSOCNN.")
    p.add_argument("--cnn_temporal_layers", type=int, default=2,
                   help="GRU layers for ENSOCNN temporal encoder.")
    p.add_argument("--convlstm_hidden_channels", type=int, default=48,
                   help="Hidden channels per spatial recurrent layer in ENSOConvLSTM.")
    p.add_argument("--convlstm_layers", type=int, default=2,
                   help="Number of stacked spatial recurrent layers in ENSOConvLSTM.")
    p.add_argument("--convlstm_pool_lat", type=int, default=16,
                   help="Latitude bins after adaptive pooling in ENSOConvLSTM.")
    p.add_argument("--convlstm_pool_lon", type=int, default=32,
                   help="Longitude bins after adaptive pooling in ENSOConvLSTM.")
    p.add_argument("--convlstm_kernel_size", type=int, default=3,
                   help="Odd convolution kernel size for ENSOConvLSTM gates.")
    p.add_argument("--geoformer_pool_lat", type=int, default=8,
                   help="Latitude bins after adaptive pooling in ENSOGeoformer.")
    p.add_argument("--geoformer_pool_lon", type=int, default=16,
                   help="Longitude bins after adaptive pooling in ENSOGeoformer.")
    p.add_argument("--geoformer_patch_size", type=int, default=2,
                   help="Patch size on the pooled grid for ENSOGeoformer.")
    p.add_argument("--geoformer_decoder_depth", type=int, default=2,
                   help="Transformer decoder depth for ENSOGeoformer.")
    p.add_argument("--gl_pool_lat", type=int, default=10,
                   help="Latitude cells for GL-Geoformer after adaptive pooling.")
    p.add_argument("--gl_pool_lon", type=int, default=20,
                   help="Longitude cells for GL-Geoformer after adaptive pooling.")
    p.add_argument("--gl_temporal_depth", type=int, default=2,
                   help="Local temporal-attention layers in GL-Geoformer.")
    p.add_argument("--gl_spatial_depth", type=int, default=2,
                   help="Global spatial-attention layers in GL-Geoformer.")
    p.add_argument("--gl_decoder_depth", type=int, default=1,
                   help="Lead-query decoder layers in GL-Geoformer.")

    p.add_argument("--atmos_d_model", type=int, default=128,
                   help="Hidden size for AtmosFormer.")
    p.add_argument("--atmos_n_heads", type=int, default=4,
                   help="Attention heads for AtmosFormer.")
    p.add_argument("--atmos_pool_lat", type=int, default=10,
                   help="Latitude bins in the AtmosFormer map branch.")
    p.add_argument("--atmos_pool_lon", type=int, default=20,
                   help="Longitude bins in the AtmosFormer map branch.")
    p.add_argument("--atmos_temporal_depth", type=int, default=2,
                   help="Temporal encoder depth for AtmosFormer.")
    p.add_argument("--atmos_spatial_depth", type=int, default=2,
                   help="Global spatial encoder depth for AtmosFormer.")
    p.add_argument("--atmos_group_depth", type=int, default=2,
                   help="Masked basin-variable group encoder depth.")
    p.add_argument("--atmos_decoder_depth", type=int, default=2,
                   help="Lead-query decoder depth for AtmosFormer.")
    p.add_argument("--atmos_ffn_mult", type=int, default=3,
                   help="Feed-forward expansion ratio for AtmosFormer.")
    p.add_argument("--atmos_dropout", type=float, default=0.10,
                   help="Dropout used by AtmosFormer.")
    p.add_argument("--atmos_coalition_probability", type=float, default=0.70,
                   help="Probability of applying structured coalition dropout.")
    p.add_argument("--atmos_group_drop", type=float, default=0.10,
                   help="Per basin-variable group drop probability.")
    p.add_argument("--atmos_basin_drop", type=float, default=0.05,
                   help="Per remote-basin drop probability.")
    p.add_argument("--atmos_variable_drop", type=float, default=0.05,
                   help="Per-variable drop probability across basins.")
    p.add_argument("--atmos_no_pairwise", action="store_true",
                   help="Disable the small pairwise basin interaction head.")
    p.add_argument(
        "--atmos_mask_strategy",
        default="bernoulli",
        choices=["bernoulli", "balanced", "structured"],
        help="Coalition-dropout sampler; balanced exposes full/leave-one masks explicitly.",
    )
    p.add_argument("--atmos_full_mask_probability", type=float, default=0.50,
                   help="Balanced sampler probability for retaining every active group.")
    p.add_argument("--atmos_leave_basin_probability", type=float, default=0.20,
                   help="Balanced sampler probability for dropping one basin.")
    p.add_argument("--atmos_leave_variable_probability", type=float, default=0.15,
                   help="Balanced sampler probability for dropping one variable.")
    p.add_argument("--atmos_regime_moe", action="store_true",
                   help="Add a residual expert restricted to spring-barrier target months.")
    p.add_argument("--atmos_spring_scale", type=float, default=0.20,
                   help="Scale of the spring-regime residual expert.")
    p.add_argument("--atmos_horizon_router", action="store_true",
                   help=("Route each lead between physical-state, map-field and "
                         "basin-group representations; used by AtmosFormerSPB."))
    p.add_argument("--atmos_phase_summary_scale", type=float, default=0.25,
                   help="Scale of the physical-history phase/low-frequency condition.")
    p.add_argument("--atmos_router_prior_strength", type=float, default=1.0,
                   help="Strength of the short-to-long horizon routing prior.")
    p.add_argument("--atmos_multiscale_phase", action="store_true",
                   help="Use 3/6/12-month physical-history summaries for phase conditioning.")
    p.add_argument("--atmos_phase_film", action="store_true",
                   help="Apply zero-initialized physical phase/amplitude FiLM to lead queries.")
    p.add_argument("--atmos_phase_film_scale", type=float, default=0.20,
                   help="Maximum scale of the phase FiLM modulation.")
    p.add_argument("--atmos_phase_film_lead_power", type=float, default=1.0,
                   help="Lead-time power controlling how phase FiLM grows with horizon.")
    p.add_argument("--atmos_target_calendar_decay", type=float, default=0.0,
                   help="Exponential decay of target-month harmonics with lead; 0 keeps legacy behavior.")
    p.add_argument("--atmos_anomaly_calibration", action="store_true",
                   help=("Add an identity-initialized state/lead-conditioned gain-offset "
                         "head for long-lead anomaly amplitude calibration."))
    p.add_argument("--atmos_anomaly_gain_scale", type=float, default=0.40,
                   help="Maximum log-gain magnitude for the anomaly calibration head.")
    p.add_argument("--atmos_anomaly_offset_scale", type=float, default=0.25,
                   help="Maximum additive anomaly offset for the calibration head.")

    p.add_argument("--mist_num_domains", type=int, default=17,
                   help="Total adapter slots (16 CMIP domains plus OBS by default).")
    p.add_argument("--mist_obs_domain_id", type=int, default=16,
                   help="Global domain id assigned to every OBS/extra-OBS dataset.")
    p.add_argument("--mist_obs_mode", default="core_only", choices=["core_only", "adapter"],
                   help="Use the invariant core alone for OBS or enable its dedicated adapter.")
    p.add_argument("--mist_domain_balanced_batches", action="store_true",
                   help="Use near-uniform per-domain MIST batches for stable GroupDRO.")

    p.add_argument("--mist_d_model", type=int, default=192,
                   help="Canonical latent-state width.")
    p.add_argument("--mist_dropout", type=float, default=argparse.SUPPRESS,
                   help="MIST dropout; omitted uses the shared --dropout value.")
    p.add_argument("--mist_encoder_width", type=int, default=64,
                   help="Base width of the spatial history encoder.")
    p.add_argument("--mist_prepool_lat", type=int, default=48,
                   help="Maximum latitude size before the stride-2 encoder stem.")
    p.add_argument("--mist_prepool_lon", type=int, default=72,
                   help="Maximum longitude size before the stride-2 encoder stem.")
    p.add_argument("--mist_pool_lat", type=int, default=8,
                   help="Latitude bins used by adaptive encoder pooling.")
    p.add_argument("--mist_pool_lon", type=int, default=16,
                   help="Longitude bins used by adaptive encoder pooling.")
    p.add_argument("--mist_encoder_layers", type=int, default=2,
                   help="Number of temporal/attention history-encoder layers.")
    p.add_argument("--mist_encoder_month_embedding", action="store_true",
                   help="Add an explicit month embedding in the history encoder (ablation).")
    p.add_argument("--mist_n_heads", type=int, default=4,
                   help="Attention heads in the history encoder.")
    p.add_argument("--mist_ffn_mult", type=int, default=4,
                   help="Feed-forward expansion ratio in the history encoder.")
    p.add_argument("--mist_v2_phase_init_std", type=float, default=0.02,
                   help="V2 harmonic phase-generator coefficient initialization.")
    p.add_argument("--mist_v2_initial_damping", type=float, default=0.005,
                   help="V2 initial damping of the scale-preserving transition.")
    p.add_argument("--mist_v2_max_state_rms", type=float, default=5.0,
                   help="V2 emergency RMS bound for recurrent-state stability.")
    p.add_argument("--mist_v2_adapter_gate_init", type=float, default=0.0,
                   help="V2 raw adapter-gate initialization (0 gives gate=0.5).")
    p.add_argument("--mist_v2_adapter_up_init_std", type=float, default=0.01,
                   help="V2 up-factor initialization for learnable domain tendencies.")

    p.add_argument("--mist_phase_harmonics", type=int, default=3,
                   help="Fourier harmonics used to generate monthly phase operators.")
    p.add_argument("--mist_phase_reflections", type=int, default=8,
                   help="Householder reflections in the orthogonal phase basis.")
    p.add_argument("--mist_phase_max_generator", type=float, default=0.35,
                   help="Maximum magnitude of the monthly matrix-exponential generator.")
    p.add_argument("--mist_phase_max_condition", type=float, default=10.0,
                   help="Desired upper condition number for phase transforms.")
    p.add_argument("--mist_phase_scale_penalty", type=float, default=0.01,
                   help="Internal spectral-scale penalty used by phase diagnostics.")

    p.add_argument("--mist_transition_hidden", type=int, default=argparse.SUPPRESS,
                   help="Shared-transition hidden width; omitted uses 2*mist_d_model.")
    p.add_argument("--mist_transition_depth", type=int, default=2,
                   help="Residual MLP depth of the shared monthly transition.")
    p.add_argument("--mist_transition_dt", type=float, default=0.25,
                   help="Initial bounded residual step size of the transition.")
    p.add_argument("--mist_transition_max_dt", type=float, default=0.5,
                   help="Maximum learned residual step size.")
    p.add_argument("--mist_adapter_rank", type=int, default=8,
                   help="Rank of each simulator-specific transition residual.")
    p.add_argument("--mist_adapter_scale", type=float, default=0.15,
                   help="Maximum scale applied to low-rank domain residuals.")
    p.add_argument("--mist_adapter_dropout", type=float, default=0.15,
                   help="Per-sample adapter dropout probability.")
    p.add_argument("--mist_domain_dropout", type=float, default=0.10,
                   help="Probability of withholding an entire present domain adapter per batch.")

    p.add_argument("--mist_innovation_rank", type=int, default=4,
                   help="Rank of the seasonal latent process covariance.")
    p.add_argument("--mist_innovation_init_std", type=float, default=0.03,
                   help="Initial latent monthly innovation standard deviation.")
    p.add_argument("--mist_innovation_min_var", type=float, default=1e-5,
                   help="Numerical floor for innovation diagonal variance.")
    p.add_argument("--mist_innovation_prior_std", type=float, default=0.05,
                   help="Prior process-noise scale used by innovation regularization.")
    p.add_argument("--mist_innovation_smoothness", type=float, default=0.05,
                   help="Circular month-to-month covariance smoothness coefficient.")
    p.add_argument("--mist_innovation_factor_smoothness", type=float, default=0.01,
                   help="Circular smoothness coefficient for low-rank covariance factors.")
    p.add_argument("--mist_uncertainty_retention", type=float, default=0.92,
                   help="Initial month-to-month latent uncertainty retention.")
    p.add_argument("--mist_head_hidden", type=int, default=argparse.SUPPRESS,
                   help="Forecast-head hidden width; omitted uses mist_d_model.")
    p.add_argument("--mist_initial_observation_std", type=float, default=0.5,
                   help="Initial predictive observation-noise standard deviation.")
    p.add_argument("--mist_min_logvar", type=float, default=-8.0,
                   help="Minimum forecast log variance.")
    p.add_argument("--mist_max_logvar", type=float, default=3.0,
                   help="Maximum forecast log variance.")
    p.add_argument("--mist_process_samples", type=int, default=0,
                   help="Reparameterized process-noise paths per forward; 0 disables sampling.")

    p.add_argument("--mist_jacobian_diagnostics", action="store_true",
                   help="Estimate shared-transition directional Jacobian gain each lead.")
    p.add_argument("--mist_jacobian_probes", type=int, default=2,
                   help="Number of finite-difference Jacobian probe vectors.")
    p.add_argument("--mist_jacobian_eps", type=float, default=1e-3,
                   help="Finite-difference step for Jacobian diagnostics.")

    p.add_argument("--mist_domain_residual_weight", type=float, default=0.0,
                   help="L2/low-rank domain residual regularization weight.")
    p.add_argument("--mist_phase_condition_weight", type=float, default=0.0,
                   help="Conditioning penalty for the invertible seasonal phase transform.")
    p.add_argument("--mist_phase_condition_target", type=float, default=10.0,
                   help="Fallback maximum desired phase-transform condition number.")
    p.add_argument("--mist_innovation_weight", type=float, default=0.0,
                   help="Structured seasonal process-innovation covariance penalty weight.")
    p.add_argument("--mist_core_consistency_weight", type=float, default=0.0,
                   help="Consistency pressure between full and invariant-core forecasts.")
    p.add_argument("--mist_core_only_weight", type=float, default=0.0,
                   help="Direct target supervision weight for core_only_pred.")
    p.add_argument("--mist_group_dro_weight", type=float, default=0.0,
                   help="Worst-domain Niño+aux forecast-risk weight.")
    p.add_argument("--mist_group_dro_temperature", type=float, default=0.10,
                   help="Temperature of smooth logsumexp GroupDRO aggregation.")
    p.add_argument("--mist_group_dro_mode", default="logsumexp",
                   choices=["logsumexp", "worst"],
                   help="Smooth or hard worst-present-domain aggregation.")
    p.add_argument("--mist_loss_warmup_epochs", type=int, default=8,
                   help="Warmup for MIST regularization/core/DRO loss terms.")
    p.add_argument("--mist_history_observed_weight", type=float, default=0.0,
                   help="Observed one-month conjugacy loss inside the input history.")
    p.add_argument("--mist_history_latent_weight", type=float, default=0.0,
                   help="Canonical latent one-month transition consistency weight.")
    p.add_argument("--mist_history_core_weight", type=float, default=0.0,
                   help="Core-only observed transition loss for withheld domains.")
    p.add_argument("--mist_history_seasonal_dro_weight", type=float, default=0.0,
                   help="Smooth worst-month risk over observed conjugacy errors.")
    p.add_argument("--mist_history_seasonal_dro_temperature", type=float, default=0.10,
                   help="Temperature for data-driven worst-season transition risk.")
    p.add_argument("--mist_horizon_curriculum_epochs", type=int, default=0,
                   help="Epochs used to expand forecast supervision to all leads.")
    p.add_argument("--mist_horizon_start", type=int, default=6,
                   help="Initial supervised rollout horizon for MIST V2.")

    p.add_argument("--ctm_d_model", type=int, default=256)
    p.add_argument("--ctm_d_input", type=int, default=128)
    p.add_argument("--ctm_ticks", type=int, default=24)
    p.add_argument("--ctm_memory", type=int, default=16)
    p.add_argument("--ctm_nlm_hidden", type=int, default=64)
    p.add_argument("--ctm_n_heads", type=int, default=4)
    p.add_argument("--ctm_d_action", type=int, default=192)
    p.add_argument("--ctm_d_out", type=int, default=192)
    p.add_argument("--ctm_n_self", type=int, default=16)
    p.add_argument("--ctm_syn_hidden", type=int, default=384)
    p.add_argument("--ctm_syn_depth", type=int, default=2)
    p.add_argument("--ctm_pool_lat", type=int, default=8)
    p.add_argument("--ctm_pool_lon", type=int, default=16)
    p.add_argument("--ctm_map_pool_lat", type=int, default=16,
                   help="v4 map-token latitude resolution (default 16, vs 8 in v2/v3).")
    p.add_argument("--ctm_map_pool_lon", type=int, default=32,
                   help="v4 map-token longitude resolution (default 32, vs 16 in v2/v3).")
    p.add_argument("--ctm_temporal_hidden", type=int, default=64,
                   help="v4 per-cell temporal MLP hidden size (folds input_len into features).")
    p.add_argument("--ctm_aggregation", default="certainty_weighted", choices=["last", "most_certain", "certainty_weighted"])
    p.add_argument("--ctm_thought_loss_weight", type=float, default=0.15)
    p.add_argument("--ctm_save_attention", action="store_true")
    p.add_argument("--ctm_halt_threshold", type=float, default=0.95, help="Cumulative halt threshold for ACT.")
    p.add_argument("--ctm_halt_temperature", type=float, default=1.0,
                   help="Softmax temperature for halting distribution. Higher=more uniform.")
    p.add_argument("--ctm_halt_warmup_epochs", type=int, default=20,
                   help="Epochs before halt gate is allowed to specialize.")

    p.add_argument("--pathtrace_d_model", type=int, default=128,
                   help="Controller/token width for ENSOPathTrace.")
    p.add_argument("--pathtrace_state_dim", type=int, default=64,
                   help="Latent width of each named pathway state.")
    p.add_argument("--pathtrace_ticks", type=int, default=12,
                   help="Fixed execution ticks exposed by ENSOPathTrace.")
    p.add_argument("--pathtrace_n_heads", type=int, default=4,
                   help="Cross-attention heads inside each spatial pathway.")
    p.add_argument("--pathtrace_pool_lat", type=int, default=6,
                   help="Pooled latitude cells for ENSOPathTrace tokens.")
    p.add_argument("--pathtrace_pool_lon", type=int, default=12,
                   help="Pooled longitude cells for ENSOPathTrace tokens.")
    p.add_argument(
        "--pathtrace_pathways", default="",
        help=("Optional map grouping, e.g. "
              "'ocean:sst,hc300;wind:slp,tauu,tauv;flux:hfds'. "
              "Unlisted fields remain in other_maps."),
    )
    p.add_argument("--pathtrace_sparsity_weight", type=float, default=0.010,
                   help="Encourage selective per-forecast pathway allocation.")
    p.add_argument("--pathtrace_coverage_weight", type=float, default=0.020,
                   help="Prevent all samples from collapsing to one pathway.")
    p.add_argument("--pathtrace_convergence_weight", type=float, default=0.010,
                   help="Penalize only late-tick pathway-state drift.")
    p.add_argument("--pathtrace_update_weight", type=float, default=0.001,
                   help="Weak squared update penalty for numerical trace stability.")
    p.add_argument("--pathtrace_min_coverage", type=float, default=0.10,
                   help="Minimum mean activation per independent pathway in ENSOPathTrace_v2.")
    p.add_argument("--pathtrace_edges", default="all",
                   help=("V3 visible directed graph: 'all', 'none', or comma-separated "
                         "source->destination pairs."))
    p.add_argument("--pathtrace_decoder_rank", type=int, default=8,
                   help="Rank of the zero-preserving additive lead decoder in PathTrace v3.")
    p.add_argument("--pathtrace_evidence_step_min", type=float, default=0.05,
                   help="Minimum convex evidence-state update fraction in PathTrace v3.")
    p.add_argument("--pathtrace_evidence_step_max", type=float, default=0.55,
                   help="Maximum convex evidence-state update fraction in PathTrace v3.")
    p.add_argument("--pathtrace_state_step_min", type=float, default=0.05,
                   help="Minimum convex process-state update fraction in PathTrace v3.")
    p.add_argument("--pathtrace_state_step_max", type=float, default=0.45,
                   help="Maximum convex process-state update fraction in PathTrace v3.")
    p.add_argument("--pathtrace_probe_weight", type=float, default=0.0,
                   help="Optional weak per-input evidence probe loss; not a causal-use claim.")
    p.add_argument("--pathtrace_refinement_weight", type=float, default=0.0,
                   help="Penalize worsening forecast error between successive v3 ticks.")
    p.add_argument("--pathtrace_contraction_weight", type=float, default=0.0,
                   help="Penalize late updates that do not contract relative to early updates.")
    p.add_argument("--pathtrace_contraction_target", type=float, default=0.35,
                   help="Target late/early update-norm ratio for PathTrace v3.")
    p.add_argument("--pathtrace_message_weight", type=float, default=0.0,
                   help="Weak edge/local message magnitude regularizer for PathTrace v3.")

    p.add_argument("--aux_weight", type=float, default=0.25)
    p.add_argument("--aux_taper_start", type=int, default=10)
    p.add_argument("--aux_taper_end", type=int, default=18)
    p.add_argument("--aux_taper_min", type=float, default=0.05)
    p.add_argument("--spb_weight", type=float, default=0.20)
    p.add_argument("--spb_lead_start", type=int, default=2)
    p.add_argument("--spb_lead_end", type=int, default=16)
    p.add_argument(
        "--spring_init_weight", type=float, default=0.0,
        help=("Optional regime-aware weight for February-May initialized "
              "samples at long leads; 0 disables it."),
    )
    p.add_argument("--spring_init_lead_start", type=int, default=12)
    p.add_argument("--spring_init_lead_end", type=int, default=18)
    p.add_argument("--spring_init_warmup_epochs", type=int, default=8)
    p.add_argument("--ode_weight", type=float, default=0.06)
    p.add_argument("--ode_lead_start", type=int, default=8)
    p.add_argument("--smoothness_weight", type=float, default=0.03)
    p.add_argument("--trend_weight", type=float, default=0.03)
    p.add_argument("--adaptive_weight", type=float, default=0.05)
    p.add_argument("--adaptive_ema", type=float, default=0.95)
    p.add_argument("--max_adaptive_scale", type=float, default=1.20)
    p.add_argument("--huber_beta", type=float, default=0.5)

    p.add_argument("--nll_weight", type=float, default=0.10,
                   help="Gaussian NLL weight for uncertainty calibration.")
    p.add_argument("--nll_warmup_epochs", type=int, default=10,
                   help="Epochs over which NLL loss ramps from 0 to full weight.")
    p.add_argument("--amplitude_weight", type=float, default=0.08,
                   help="Amplitude-asymmetric loss weight for extreme events.")
    p.add_argument("--amplitude_threshold", type=float, default=1.0,
                   help="Anomaly threshold (standardized) for extreme event detection.")
    p.add_argument("--corr_weight", type=float, default=0.05,
                   help="Correlation (Pearson r) loss weight.")
    p.add_argument("--corr_lead_start", type=int, default=6,
                   help="Lead month from which correlation loss is applied.")
    p.add_argument("--batch_corr_weight", type=float, default=0.0,
                   help="Direct per-lead batch ACC loss weight (0 keeps legacy loss).")
    p.add_argument("--batch_corr_lead_start", type=int, default=6,
                   help="First 1-based lead used by direct batch ACC loss.")
    p.add_argument("--batch_corr_lead_end", type=int, default=18,
                   help="Last 1-based lead used by direct batch ACC loss.")
    p.add_argument("--batch_corr_warmup_epochs", type=int, default=8,
                   help="Warmup epochs for the direct batch ACC loss.")
    p.add_argument("--long_lead_weight", type=float, default=0.0,
                   help="Extra normalized weight ramp across long leads (0 disables).")
    p.add_argument("--long_lead_start", type=int, default=12,
                   help="First 1-based lead in the long-lead weight ramp.")
    p.add_argument("--long_lead_end", type=int, default=18,
                   help="Last 1-based lead in the long-lead weight ramp.")
    p.add_argument("--scale_weight", type=float, default=0.0,
                   help="Long-lead standard-deviation/slope matching weight.")
    p.add_argument("--scale_lead_start", type=int, default=12,
                   help="First 1-based lead used by scale matching.")
    p.add_argument("--scale_lead_end", type=int, default=18,
                   help="Last 1-based lead used by scale matching.")
    p.add_argument("--scale_warmup_epochs", type=int, default=8,
                   help="Warmup epochs for long-lead scale matching.")
    p.add_argument("--phase_weight", type=float, default=0.0,
                   help="Long-lead ENSO phase/sign loss weight.")
    p.add_argument("--phase_lead_start", type=int, default=12,
                   help="First 1-based lead used by phase loss.")
    p.add_argument("--phase_lead_end", type=int, default=18,
                   help="Last 1-based lead used by phase loss.")
    p.add_argument("--phase_threshold", type=float, default=0.5,
                   help="Absolute standardized target threshold for phase loss.")
    p.add_argument("--phase_margin", type=float, default=0.25,
                   help="Desired signed standardized margin in phase loss.")
    p.add_argument("--phase_warmup_epochs", type=int, default=8,
                   help="Warmup epochs for long-lead phase loss.")
    p.add_argument("--phase_drift_weight", type=float, default=0.0,
                   help="Low-frequency long-lead tendency/phase-drift loss weight.")
    p.add_argument("--phase_drift_lead_start", type=int, default=9,
                   help="First 1-based lead used by phase-drift loss.")
    p.add_argument("--phase_drift_lead_end", type=int, default=18,
                   help="Last 1-based lead used by phase-drift loss.")
    p.add_argument("--phase_drift_warmup_epochs", type=int, default=8,
                   help="Warmup epochs for phase-drift loss.")
    p.add_argument("--seasonal_mean_weight", type=float, default=0.0,
                   help="Long-lead target-month anomaly-mean calibration weight.")
    p.add_argument("--seasonal_mean_lead_start", type=int, default=12,
                   help="First 1-based lead used by anomaly-mean calibration.")
    p.add_argument("--seasonal_mean_lead_end", type=int, default=18,
                   help="Last 1-based lead used by anomaly-mean calibration.")
    p.add_argument("--seasonal_mean_warmup_epochs", type=int, default=8,
                   help="Warmup epochs for anomaly-mean calibration.")
    p.add_argument("--selective_weight", type=float, default=0.0,
                   help="Optional uncertainty-aware selective-learning weight.")
    p.add_argument("--selective_lead_start", type=int, default=12,
                   help="First 1-based lead used by selective learning.")
    p.add_argument("--selective_lead_end", type=int, default=18,
                   help="Last 1-based lead used by selective learning.")
    p.add_argument("--selective_min_confidence", type=float, default=0.35,
                   help="Floor for normalized selective confidence weights.")
    p.add_argument("--selective_warmup_epochs", type=int, default=8,
                   help="Warmup epochs for selective learning.")
    p.add_argument("--scale_target_ratio", type=float, default=1.0,
                   help="Desired prediction/target spread ratio in long-lead scale loss.")
    p.add_argument("--scale_slope_target", type=float, default=1.0,
                   help="Desired regression slope in long-lead scale loss.")
    p.add_argument("--ponder_weight", type=float, default=0.01,
                   help="Ponder cost weight for halting regularization.")
    p.add_argument("--halt_entropy_weight", type=float, default=0.05,
                   help="Entropy regularization weight for tick distribution diversity.")
    p.add_argument("--overshoot_weight", type=float, default=0.05,
                   help="Weight for amplitude overprediction penalty.")
    p.add_argument("--pinn_weight", type=float, default=0.0,
                   help="Weight for CTM+PINN recharge-discharge residuals.")
    p.add_argument("--pinn_energy_weight", type=float, default=0.0,
                   help="Weight for CTM+PINN latent energy-growth regularization.")
    p.add_argument("--pinn_warmup_epochs", type=int, default=8,
                   help="Epochs over which PINN loss ramps from 0 to full weight.")
    p.add_argument("--pinn_state_dim", type=int, default=4,
                   help="Number of mechanism-state channels exposed by OfficialPINN models.")
    p.add_argument("--pinn_warmup_leads", type=int, default=3,
                   help="Lead steps down-weighted inside the PINN residual.")

    p.add_argument("--pinn_season_amp", type=float, default=0.35,
                   help="M1: max fractional seasonal modulation of the 7 PINN coefficients (0 disables).")
    p.add_argument("--pinn_prior_weight", type=float, default=0.0,
                   help="M2: weight of the literature prior on the linearized recharge oscillator.")
    p.add_argument("--latent_anchor_weight", type=float, default=0.0,
                   help="M3: weight anchoring latent U/C to observed wind/cross-basin composites.")
    p.add_argument("--latent_anchor_leads", type=int, default=3,
                   help="M3: latent states averaged over the first K leads before anchoring.")
    p.add_argument("--cf_weight", type=float, default=0.0,
                   help="M5: weight of the counterfactual recharge-consistency loss.")
    p.add_argument("--cf_warmup_epochs", type=int, default=6,
                   help="M5: epochs over which the counterfactual loss ramps to full weight.")
    p.add_argument("--cf_delta", type=float, default=0.3,
                   help="M5: fractional amplification of hc/wwv in the counterfactual branch.")
    p.add_argument("--cf_frac", type=float, default=0.5,
                   help="M5: fraction of each batch used in the counterfactual branch.")
    p.add_argument("--cf_deadband", type=float, default=0.3,
                   help="M5: |WWV| (std) below which no response sign is required.")
    p.add_argument("--cf_lead_lo", type=int, default=4,
                   help="M5: first lead (1-based) where the response sign is enforced.")
    p.add_argument("--cf_lead_hi", type=int, default=14,
                   help="M5: last lead (1-based) where the response sign is enforced.")
    p.add_argument("--gate_l1_weight", type=float, default=0.0,
                   help="M4: L1 pressure on per-map-variable input gates.")

    p.add_argument("--amp", action="store_true",
                   help="[v3 lever 2] Enable bf16 autocast (mixed precision) for "
                        "forward + loss. Needs a bf16-capable GPU (Ampere+).")
    p.add_argument("--compile", action="store_true",
                   help="[v3 lever 2] torch.compile the model to fuse per-tick "
                        "kernels. Combine with --ctm_halt_train_break 0 to avoid "
                        "graph breaks from data-dependent early halting.")
    p.add_argument("--cf_every", type=int, default=1,
                   help="[v3 lever 4] Run the M5 counterfactual re-forward once "
                        "every N steps (1 = every step, as in v2; 2-3 ~halves its cost).")
    p.add_argument("--ctm_adaptive_halt", action="store_true",
                   help="[v3 lever 5] Enable PonderNet-style adaptive tick halting "
                        "with early exit. Off => v2 fixed-tick certainty aggregation.")
    p.add_argument("--ctm_halt_min_ticks", type=int, default=4,
                   help="[v3 lever 5] Minimum ticks before an early halt is allowed "
                        "(keeps a minimum 'thinking' trace).")
    p.add_argument("--ctm_halt_train_break", type=int, default=1,
                   help="[v3 lever 5] 1 = allow early break during training (real "
                        "speedup as the gate learns); 0 = halt only at inference.")

    return p


if __name__ == "__main__":
    main(build_parser().parse_args())
