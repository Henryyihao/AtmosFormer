"""Losses for the interventional, index-free ENSO pathway-trace model.

The forecast objective remains the project's tested :class:`ENSOCTMLossV2`.
The additional terms only shape the *representation* of the explicit pathway
trace:

* sparse per-forecast pathway allocation, so a forecast can name a dominant
  route instead of assigning equal importance to every field;
* batch-level pathway coverage, so sparsity cannot collapse the whole model
  onto one convenient map group;
* late-tick convergence, so the exposed execution trace is a genuine
  refinement trajectory rather than unbounded hidden-state drift;
* a weak update-size penalty for numerical stability.

For the explicit-graph v3 model the same class can additionally train weak
evidence probes, monotone forecast refinement, late/early update contraction,
and bounded visible messages.  These terms shape optimization only; deletion,
message ablation and state patching remain held-out evaluation operations.

No input-deletion or state-patching loss is used here.  Those operations are
held out for the intervention API at evaluation time; otherwise a paper could
mistake a trained-to-pass diagnostic for independent explanation faithfulness.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F

from loss_v2 import ENSOCTMLossV2


class ENSOPathTraceLoss(ENSOCTMLossV2):
    """Forecast loss plus non-tautological pathway-trace regularization."""

    def __init__(
        self,
        *args: Any,
        pathtrace_sparsity_weight: float = 0.010,
        pathtrace_coverage_weight: float = 0.020,
        pathtrace_convergence_weight: float = 0.010,
        pathtrace_update_weight: float = 0.001,
        pathtrace_min_coverage: float = 0.10,
        pathtrace_probe_weight: float = 0.0,
        pathtrace_refinement_weight: float = 0.0,
        pathtrace_contraction_weight: float = 0.0,
        pathtrace_contraction_target: float = 0.35,
        pathtrace_message_weight: float = 0.0,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.pathtrace_sparsity_weight = max(0.0, float(pathtrace_sparsity_weight))
        self.pathtrace_coverage_weight = max(0.0, float(pathtrace_coverage_weight))
        self.pathtrace_convergence_weight = max(0.0, float(pathtrace_convergence_weight))
        self.pathtrace_update_weight = max(0.0, float(pathtrace_update_weight))
        self.pathtrace_min_coverage = min(max(0.0, float(pathtrace_min_coverage)), 1.0)
        self.pathtrace_probe_weight = max(0.0, float(pathtrace_probe_weight))
        self.pathtrace_refinement_weight = max(0.0, float(pathtrace_refinement_weight))
        self.pathtrace_contraction_weight = max(0.0, float(pathtrace_contraction_weight))
        self.pathtrace_contraction_target = max(0.0, float(pathtrace_contraction_target))
        self.pathtrace_message_weight = max(0.0, float(pathtrace_message_weight))

    @staticmethod
    def _zero(reference: torch.Tensor) -> torch.Tensor:
        return reference.new_zeros(())

    @staticmethod
    def _normalized_entropy(prob: torch.Tensor, dim: int) -> torch.Tensor:
        """Entropy in [0, 1] for a categorical distribution along ``dim``."""
        n = prob.shape[dim]
        if n <= 1:
            return torch.zeros_like(prob.sum(dim=dim))
        p = prob.clamp(min=1.0e-8)
        entropy = -(p * p.log()).sum(dim=dim)
        return entropy / torch.log(p.new_tensor(float(n)))

    def loss_components(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        init_month: Optional[torch.Tensor] = None,
        model_diag: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, torch.Tensor]:
        components = super().loss_components(
            pred, target, init_month=init_month, model_diag=model_diag
        )
        components.pop("total", None)
        zero = self._zero(pred)

        path_weights = None if model_diag is None else model_diag.get("pathway_weight_trace")
        if torch.is_tensor(path_weights) and path_weights.ndim == 4:
            gate_mode = "competitive"
            if isinstance(model_diag, dict):
                gate_mode = str(model_diag.get("pathtrace_gate_mode", gate_mode)).lower()
            if gate_mode == "derived_importance":
                sample_entropy = self._normalized_entropy(path_weights, dim=2).mean()
                components["pathtrace_sparsity"] = (
                    self.pathtrace_sparsity_weight * sample_entropy
                )
                mean_weight = path_weights.mean(dim=(0, 1, 3))
                coverage_loss = torch.relu(
                    mean_weight.new_full(mean_weight.shape, self.pathtrace_min_coverage)
                    - mean_weight
                ).mean()
                components["pathtrace_coverage"] = (
                    self.pathtrace_coverage_weight * coverage_loss
                )
            elif gate_mode == "independent":
                components["pathtrace_sparsity"] = (
                    self.pathtrace_sparsity_weight * path_weights.mean()
                )
                mean_weight = path_weights.mean(dim=(0, 1, 3))
                coverage_loss = torch.relu(
                    mean_weight.new_full(mean_weight.shape, self.pathtrace_min_coverage) - mean_weight
                ).mean()
                components["pathtrace_coverage"] = (
                    self.pathtrace_coverage_weight * coverage_loss
                )
            else:
                sample_entropy = self._normalized_entropy(path_weights, dim=2).mean()
                components["pathtrace_sparsity"] = (
                    self.pathtrace_sparsity_weight * sample_entropy
                )
                mean_weight = path_weights.mean(dim=(0, 1, 3))
                coverage_loss = 1.0 - self._normalized_entropy(mean_weight, dim=0)
                components["pathtrace_coverage"] = (
                    self.pathtrace_coverage_weight * coverage_loss
                )
        else:
            components["pathtrace_sparsity"] = zero
            components["pathtrace_coverage"] = zero

        state_trace = None if model_diag is None else model_diag.get("path_state_trace")
        if torch.is_tensor(state_trace) and state_trace.ndim == 4 and state_trace.shape[1] > 1:
            ticks = state_trace.shape[1]
            start = max(1, (2 * ticks) // 3)
            late_delta = state_trace[:, start:] - state_trace[:, start - 1:-1]
            convergence = late_delta.pow(2).mean()
            components["pathtrace_convergence"] = (
                self.pathtrace_convergence_weight * convergence
            )
        else:
            components["pathtrace_convergence"] = zero

        update_trace = None if model_diag is None else model_diag.get("path_update_trace")
        if torch.is_tensor(update_trace) and update_trace.numel() > 0:
            components["pathtrace_update"] = (
                self.pathtrace_update_weight * update_trace.pow(2).mean()
            )
        else:
            components["pathtrace_update"] = zero

        probe_predictions = None if model_diag is None else model_diag.get(
            "pathway_probe_preds"
        )
        if (
            self.pathtrace_probe_weight > 0
            and torch.is_tensor(probe_predictions)
            and probe_predictions.ndim == 5
        ):
            final_probe = probe_predictions[:, -1]
            target_expanded = target.unsqueeze(1).expand(
                -1, final_probe.shape[1], -1, -1
            )
            target_dim = min(final_probe.shape[-1], target.shape[-1])
            probe_loss = F.smooth_l1_loss(
                final_probe[..., :target_dim],
                target_expanded[..., :target_dim],
                beta=self.huber_beta,
            )
            components["pathtrace_probe"] = self.pathtrace_probe_weight * probe_loss
        else:
            components["pathtrace_probe"] = zero

        thought_predictions = None if model_diag is None else model_diag.get("thought_preds")
        if (
            self.pathtrace_refinement_weight > 0
            and torch.is_tensor(thought_predictions)
            and thought_predictions.ndim == 4
            and thought_predictions.shape[1] > 1
        ):
            target_dim = min(thought_predictions.shape[-1], target.shape[-1])
            target_trace = target.unsqueeze(1).expand(
                -1, thought_predictions.shape[1], -1, -1
            )
            tick_error = F.smooth_l1_loss(
                thought_predictions[..., :target_dim],
                target_trace[..., :target_dim],
                beta=self.huber_beta,
                reduction="none",
            ).mean(dim=(2, 3))
            worsening = F.relu(tick_error[:, 1:] - tick_error[:, :-1])
            components["pathtrace_refinement"] = (
                self.pathtrace_refinement_weight * worsening.mean()
            )
        else:
            components["pathtrace_refinement"] = zero

        if (
            self.pathtrace_contraction_weight > 0
            and torch.is_tensor(update_trace)
            and update_trace.ndim == 4
            and update_trace.shape[1] > 2
        ):
            update_norm = update_trace.pow(2).mean(dim=-1).sqrt()
            width = max(update_norm.shape[1] // 3, 1)
            early = update_norm[:, :width].mean(dim=1).detach()
            late = update_norm[:, -width:].mean(dim=1)
            ratio = late / early.clamp_min(1.0e-4)
            contraction = F.relu(ratio - self.pathtrace_contraction_target).mean()
            components["pathtrace_contraction"] = (
                self.pathtrace_contraction_weight * contraction
            )
        else:
            components["pathtrace_contraction"] = zero

        if self.pathtrace_message_weight > 0 and isinstance(model_diag, dict):
            visible_messages = []
            for key in ("local_message_trace", "edge_message_trace"):
                value = model_diag.get(key)
                if torch.is_tensor(value) and value.numel() > 0:
                    visible_messages.append(value.pow(2).mean())
            if visible_messages:
                message_loss = torch.stack(visible_messages).mean()
                components["pathtrace_message"] = (
                    self.pathtrace_message_weight * message_loss
                )
            else:
                components["pathtrace_message"] = zero
        else:
            components["pathtrace_message"] = zero

        components["total"] = sum(components.values())
        return components
