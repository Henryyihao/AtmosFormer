from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ENSOCTMLossV2(nn.Module):

    def __init__(
        self,
        aux_weight: float = 0.25,
        aux_taper_start: int = 10,
        aux_taper_end: int = 18,
        aux_taper_min: float = 0.05,
        spb_weight: float = 0.20,
        spb_lead_start: int = 2,
        spb_lead_end: int = 16,
        spring_init_weight: float = 0.0,
        spring_init_lead_start: int = 12,
        spring_init_lead_end: int = 18,
        spring_init_warmup_epochs: int = 8,
        ode_weight: float = 0.06,
        ode_lead_start: int = 8,
        smoothness_weight: float = 0.03,
        trend_weight: float = 0.03,
        adaptive_weight: float = 0.05,
        adaptive_ema: float = 0.95,
        max_adaptive_scale: float = 1.20,
        nll_weight: float = 0.10,
        nll_warmup_epochs: int = 10,
        amplitude_weight: float = 0.08,
        amplitude_threshold: float = 1.0,
        corr_weight: float = 0.05,
        corr_lead_start: int = 6,
        batch_corr_weight: float = 0.0,
        batch_corr_lead_start: int = 6,
        batch_corr_lead_end: int = 18,
        batch_corr_warmup_epochs: int = 8,
        long_lead_weight: float = 0.0,
        long_lead_start: int = 12,
        long_lead_end: int = 18,
        scale_weight: float = 0.0,
        scale_lead_start: int = 12,
        scale_lead_end: int = 18,
        scale_warmup_epochs: int = 8,
        phase_weight: float = 0.0,
        phase_lead_start: int = 12,
        phase_lead_end: int = 18,
        phase_threshold: float = 0.5,
        phase_margin: float = 0.25,
        phase_warmup_epochs: int = 8,
        phase_drift_weight: float = 0.0,
        phase_drift_lead_start: int = 9,
        phase_drift_lead_end: int = 18,
        phase_drift_warmup_epochs: int = 8,
        seasonal_mean_weight: float = 0.0,
        seasonal_mean_lead_start: int = 12,
        seasonal_mean_lead_end: int = 18,
        seasonal_mean_warmup_epochs: int = 8,
        selective_weight: float = 0.0,
        selective_lead_start: int = 12,
        selective_lead_end: int = 18,
        selective_min_confidence: float = 0.35,
        selective_warmup_epochs: int = 8,
        scale_target_ratio: float = 1.0,
        scale_slope_target: float = 1.0,
        ponder_weight: float = 0.01,
        halt_entropy_weight: float = 0.05,
        overshoot_weight: float = 0.05,
        pinn_weight: float = 0.0,
        pinn_energy_weight: float = 0.0,
        pinn_warmup_epochs: int = 8,
        pinn_prior_weight: float = 0.0,
        latent_anchor_weight: float = 0.0,
        cf_weight: float = 0.0,
        cf_warmup_epochs: int = 6,
        gate_l1_weight: float = 0.0,
        mist_domain_residual_weight: float = 0.0,
        mist_phase_condition_weight: float = 0.0,
        mist_phase_condition_target: float = 10.0,
        mist_innovation_weight: float = 0.0,
        mist_core_consistency_weight: float = 0.0,
        mist_core_only_weight: float = 0.0,
        mist_group_dro_weight: float = 0.0,
        mist_group_dro_temperature: float = 0.10,
        mist_group_dro_mode: str = "logsumexp",
        mist_loss_warmup_epochs: int = 8,
        mist_history_observed_weight: float = 0.0,
        mist_history_latent_weight: float = 0.0,
        mist_history_core_weight: float = 0.0,
        mist_history_seasonal_dro_weight: float = 0.0,
        mist_history_seasonal_dro_temperature: float = 0.10,
        mist_horizon_curriculum_epochs: int = 0,
        mist_horizon_start: int = 6,
        huber_beta: float = 0.5,
    ):
        super().__init__()
        self.aux_weight = aux_weight
        self.aux_taper_start = aux_taper_start
        self.aux_taper_end = aux_taper_end
        self.aux_taper_min = aux_taper_min
        self.spb_weight = spb_weight
        self.spb_lead_start = spb_lead_start
        self.spb_lead_end = spb_lead_end
        self.spring_init_weight = max(float(spring_init_weight), 0.0)
        self.spring_init_lead_start = max(int(spring_init_lead_start), 1)
        self.spring_init_lead_end = max(
            int(spring_init_lead_end), self.spring_init_lead_start
        )
        self.spring_init_warmup_epochs = max(int(spring_init_warmup_epochs), 0)
        self.ode_weight = ode_weight
        self.ode_lead_start = ode_lead_start
        self.smoothness_weight = smoothness_weight
        self.trend_weight = trend_weight
        self.adaptive_weight = adaptive_weight
        self.adaptive_ema = adaptive_ema
        self.max_adaptive_scale = max_adaptive_scale
        self.nll_weight = nll_weight
        self.nll_warmup_epochs = nll_warmup_epochs
        self.amplitude_weight = amplitude_weight
        self.amplitude_threshold = amplitude_threshold
        self.corr_weight = corr_weight
        self.corr_lead_start = corr_lead_start
        self.batch_corr_weight = max(float(batch_corr_weight), 0.0)
        self.batch_corr_lead_start = max(int(batch_corr_lead_start), 1)
        self.batch_corr_lead_end = max(
            int(batch_corr_lead_end), self.batch_corr_lead_start
        )
        self.batch_corr_warmup_epochs = max(int(batch_corr_warmup_epochs), 0)
        self.long_lead_weight = max(float(long_lead_weight), 0.0)
        self.long_lead_start = max(int(long_lead_start), 1)
        self.long_lead_end = max(int(long_lead_end), self.long_lead_start)
        self.scale_weight = max(float(scale_weight), 0.0)
        self.scale_lead_start = max(int(scale_lead_start), 1)
        self.scale_lead_end = max(int(scale_lead_end), self.scale_lead_start)
        self.scale_warmup_epochs = max(int(scale_warmup_epochs), 0)
        self.phase_weight = max(float(phase_weight), 0.0)
        self.phase_lead_start = max(int(phase_lead_start), 1)
        self.phase_lead_end = max(int(phase_lead_end), self.phase_lead_start)
        self.phase_threshold = max(float(phase_threshold), 0.0)
        self.phase_margin = max(float(phase_margin), 0.0)
        self.phase_warmup_epochs = max(int(phase_warmup_epochs), 0)
        self.phase_drift_weight = max(float(phase_drift_weight), 0.0)
        self.phase_drift_lead_start = max(int(phase_drift_lead_start), 1)
        self.phase_drift_lead_end = max(
            int(phase_drift_lead_end), self.phase_drift_lead_start + 1
        )
        self.phase_drift_warmup_epochs = max(int(phase_drift_warmup_epochs), 0)
        self.seasonal_mean_weight = max(float(seasonal_mean_weight), 0.0)
        self.seasonal_mean_lead_start = max(int(seasonal_mean_lead_start), 1)
        self.seasonal_mean_lead_end = max(
            int(seasonal_mean_lead_end), self.seasonal_mean_lead_start
        )
        self.seasonal_mean_warmup_epochs = max(int(seasonal_mean_warmup_epochs), 0)
        self.selective_weight = max(float(selective_weight), 0.0)
        self.selective_lead_start = max(int(selective_lead_start), 1)
        self.selective_lead_end = max(
            int(selective_lead_end), self.selective_lead_start
        )
        self.selective_min_confidence = min(
            max(float(selective_min_confidence), 0.0), 1.0
        )
        self.selective_warmup_epochs = max(int(selective_warmup_epochs), 0)
        self.scale_target_ratio = max(float(scale_target_ratio), 1.0e-3)
        self.scale_slope_target = float(scale_slope_target)
        self.ponder_weight = ponder_weight
        self.halt_entropy_weight = halt_entropy_weight
        self.overshoot_weight = overshoot_weight
        self.pinn_weight = pinn_weight
        self.pinn_energy_weight = pinn_energy_weight
        self.pinn_warmup_epochs = pinn_warmup_epochs
        self.pinn_prior_weight = pinn_prior_weight
        self.latent_anchor_weight = latent_anchor_weight
        self.cf_weight = cf_weight
        self.cf_warmup_epochs = cf_warmup_epochs
        self.gate_l1_weight = gate_l1_weight
        self.mist_domain_residual_weight = float(mist_domain_residual_weight)
        self.mist_phase_condition_weight = float(mist_phase_condition_weight)
        self.mist_phase_condition_target = max(float(mist_phase_condition_target), 1.0)
        self.mist_innovation_weight = float(mist_innovation_weight)
        self.mist_core_consistency_weight = float(mist_core_consistency_weight)
        self.mist_core_only_weight = float(mist_core_only_weight)
        self.mist_group_dro_weight = float(mist_group_dro_weight)
        self.mist_group_dro_temperature = max(float(mist_group_dro_temperature), 1e-4)
        self.mist_group_dro_mode = str(mist_group_dro_mode).lower()
        if self.mist_group_dro_mode not in {"logsumexp", "worst"}:
            raise ValueError(
                "mist_group_dro_mode must be one of {'logsumexp', 'worst'}, "
                f"got {mist_group_dro_mode!r}"
            )
        self.mist_loss_warmup_epochs = max(int(mist_loss_warmup_epochs), 0)
        self.mist_history_observed_weight = float(mist_history_observed_weight)
        self.mist_history_latent_weight = float(mist_history_latent_weight)
        self.mist_history_core_weight = float(mist_history_core_weight)
        self.mist_history_seasonal_dro_weight = float(
            mist_history_seasonal_dro_weight
        )
        self.mist_history_seasonal_dro_temperature = max(
            float(mist_history_seasonal_dro_temperature), 1.0e-4
        )
        self.mist_horizon_curriculum_epochs = max(
            int(mist_horizon_curriculum_epochs), 0
        )
        self.mist_horizon_start = max(int(mist_horizon_start), 1)
        self.huber_beta = huber_beta

        self.register_buffer("ema_loss", torch.tensor(1.0))
        self.register_buffer("current_epoch", torch.tensor(0))

        self.raw_a = nn.Parameter(torch.tensor(0.0))
        self.raw_b = nn.Parameter(torch.tensor(0.5))
        self.raw_c = nn.Parameter(torch.tensor(0.0))

    def _aux_taper_weights(self, L: int, device) -> torch.Tensor:
        """Auxiliary target weight schedule: full → tapered → minimum."""
        w = torch.ones(L, device=device)
        for i in range(L):
            if i >= self.aux_taper_end:
                w[i] = self.aux_taper_min
            elif i >= self.aux_taper_start:
                frac = (i - self.aux_taper_start) / max(self.aux_taper_end - self.aux_taper_start, 1)
                w[i] = 1.0 - frac * (1.0 - self.aux_taper_min)
        return w

    def _spb_month_weights(self, init_month: torch.LongTensor, L: int, device) -> torch.Tensor:
        """Spring Predictability Barrier: boost loss for predictions crossing MAM.

        Enhanced: also boost loss for predictions *initialized* in winter (Nov-Feb)
        that must predict across the spring barrier — these are the hardest cases.
        """
        B = init_month.shape[0]
        if self.spb_weight <= 0:
            return torch.ones(B, L, device=device)
        leads = torch.arange(1, L + 1, device=device).view(1, L)
        target_months = (init_month.view(B, 1) + leads) % 12
        is_spb = ((target_months >= 2) & (target_months <= 4)).float()
        lead_mask = torch.zeros(L, device=device)
        start = min(max(self.spb_lead_start - 1, 0), L)
        end = min(max(self.spb_lead_end, start), L)
        lead_mask[start:end] = 1.0

        winter_init = ((init_month >= 10) | (init_month <= 1)).float()
        winter_boost = winter_init.view(B, 1) * 0.5

        return 1.0 + (self.spb_weight + winter_boost) * is_spb * lead_mask.view(1, L)

    def _spring_initialization_weights(
        self, init_month: torch.LongTensor, length: int, device
    ) -> torch.Tensor:
        """Emphasize long-lead forecasts initialized in February-May.

        The ordinary SPB term keys off the *target* month.  That is useful for
        the seasonal barrier itself, but it cannot distinguish a January
        initialization from a July initialization once both trajectories pass
        through spring.  This optional term follows regime-aware/selective
        learning and gives the hard spring-initialized samples a bounded,
        lead-ramped weight.  The returned matrix is normalized to mean one so
        enabling it does not silently change the global loss scale.
        """
        weights = torch.ones(init_month.shape[0], length, device=device)
        if self.spring_init_weight <= 0.0 or length <= 0:
            return weights
        start = min(max(self.spring_init_lead_start - 1, 0), length - 1)
        end = min(max(self.spring_init_lead_end, start + 1), length)
        ramp = torch.linspace(0.0, 1.0, end - start, device=device)
        spring_init = ((init_month >= 1) & (init_month <= 4)).to(dtype=weights.dtype)
        epoch = int(self.current_epoch.item())
        warmup = min(
            1.0, float(epoch + 1) / max(self.spring_init_warmup_epochs, 1)
        )
        weights[:, start:end] += (
            self.spring_init_weight
            * warmup
            * spring_init[:, None]
            * ramp[None, :]
        )
        return weights / weights.mean().clamp_min(1.0e-6)

    def _ode_loss(self, pred: torch.Tensor) -> torch.Tensor:
        """Recharge-discharge ODE consistency for the predicted Nino3.4 trajectory.

        Physics: dT/dt ≈ a*T + b*H (Bjerknes + thermocline feedback)
                 dH/dt ≈ -c*T       (recharge-discharge)
        Where T = Nino3.4, H = thermocline_tilt (or proxy)
        """
        if pred.shape[-1] < 2 or pred.shape[1] < self.ode_lead_start + 3:
            return torch.tensor(0.0, device=pred.device)

        T_pred = pred[:, self.ode_lead_start:, 0]
        H_pred = pred[:, self.ode_lead_start:, 1]

        a = -F.softplus(self.raw_a)
        b = F.softplus(self.raw_b)
        c = F.softplus(self.raw_c)

        dT = T_pred[:, 1:] - T_pred[:, :-1]
        dH = H_pred[:, 1:] - H_pred[:, :-1]
        T_vals = T_pred[:, :-1]
        H_vals = H_pred[:, :-1]

        T_tend = a * T_vals + b * H_vals
        H_tend = -c * T_vals

        n = dT.shape[1]
        lead_w = 0.5 + 0.5 * torch.linspace(0, 1, n, device=pred.device)

        loss_T = (F.mse_loss(dT, T_tend, reduction='none') * lead_w).mean()
        loss_H = (F.mse_loss(dH, H_tend, reduction='none') * lead_w).mean()
        return loss_T + loss_H

    def _smoothness_loss(self, pred: torch.Tensor) -> torch.Tensor:
        """Penalize erratic lead-to-lead jumps in predictions."""
        diff = pred[:, 2:] - 2 * pred[:, 1:-1] + pred[:, :-2]
        return diff.pow(2).mean()

    def _trend_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Penalize wrong trend direction in the prediction envelope."""
        if pred.shape[1] < 12:
            return torch.tensor(0.0, device=pred.device)
        p_trend = pred[:, 6:, 0] - pred[:, :-6, 0]
        t_trend = target[:, 6:, 0] - target[:, :-6, 0]
        sign_err = F.relu(-p_trend * t_trend)
        return sign_err.mean()

    def _nll_loss(self, pred: torch.Tensor, target: torch.Tensor,
                  logvar: torch.Tensor) -> torch.Tensor:
        """Gaussian NLL loss to calibrate uncertainty estimates.

        Trains the logvar head to output meaningful uncertainty:
        NLL = 0.5 * (logvar + (target - pred)^2 / exp(logvar))

        This incentivizes the model to:
        - Predict high logvar (large σ) when it makes large errors
        - Predict low logvar (small σ) when it's accurate
        """
        nino_pred = pred[..., 0]
        nino_true = target[..., 0]
        nino_logvar = logvar[..., 0]

        sq_err = (nino_true - nino_pred).pow(2)
        nll = 0.5 * (nino_logvar + sq_err / (torch.exp(nino_logvar) + 1e-6))
        return nll.mean()

    def _amplitude_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Asymmetric loss that penalizes underprediction of extreme events.

        Physics: Strong El Niño/La Niña events are the most impactful and
        valuable to forecast correctly. The model tends to regress toward
        the mean (amplitude attenuation). This loss explicitly penalizes
        cases where |target| > threshold but |pred| < |target|.
        """
        nino_pred = pred[..., 0]
        nino_true = target[..., 0]

        is_extreme = (nino_true.abs() > self.amplitude_threshold).float()

        if is_extreme.sum() < 1:
            return torch.tensor(0.0, device=pred.device)

        pred_amp = nino_pred.abs()
        true_amp = nino_true.abs()

        under_pred = F.relu(true_amp - pred_amp) * is_extreme

        wrong_sign = (nino_pred * nino_true < 0).float() * is_extreme
        sign_penalty = true_amp * wrong_sign

        intensity_weight = true_amp / self.amplitude_threshold
        weighted_loss = (under_pred + 0.5 * sign_penalty) * intensity_weight * is_extreme

        return weighted_loss.sum() / (is_extreme.sum() + 1e-6)

    def _overshoot_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Penalizes overprediction of event amplitude.

        Complementary to amplitude_loss: if |pred| >> |target|, the forecast
        is raising false alarms. This is especially harmful for moderate events
        (|target| in [0.5, threshold]) where the model overshoots.
        """
        nino_pred = pred[..., 0]
        nino_true = target[..., 0]

        pred_amp = nino_pred.abs()
        true_amp = nino_true.abs()

        overshoot_ratio = (pred_amp - true_amp) / (true_amp + 0.3)
        overshoot = F.relu(overshoot_ratio - 0.5)

        return overshoot.mean()

    def _correlation_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Negative Pearson correlation loss for lead times >= corr_lead_start.

        Directly optimizes temporal correlation, which is the standard ENSO
        forecast verification metric (ACC). Applied to medium-long leads where
        MSE alone may not capture pattern accuracy.
        """
        nino_pred = pred[:, self.corr_lead_start:, 0].float()
        nino_true = target[:, self.corr_lead_start:, 0].float()

        if nino_pred.shape[0] < 4:
            return torch.tensor(0.0, device=pred.device)

        pred_centered = nino_pred - nino_pred.mean(dim=1, keepdim=True)
        true_centered = nino_true - nino_true.mean(dim=1, keepdim=True)

        num = (pred_centered * true_centered).sum(dim=1)
        pred_norm = (pred_centered.pow(2).sum(dim=1) + 1e-6).sqrt()
        true_norm = (true_centered.pow(2).sum(dim=1) + 1e-6).sqrt()
        den = pred_norm * true_norm
        corr = num / den

        return (1.0 - corr).mean()

    def _batch_lead_correlation_loss(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Approximate the reported per-lead ACC during training.

        Pearson correlation is computed over samples for each forecast lead,
        rather than over the 24 leads within one sample.  It is intentionally
        separate from ``_correlation_loss`` so existing model defaults remain
        unchanged and the new objective can be ablated cleanly.
        """
        start = min(max(self.batch_corr_lead_start - 1, 0), pred.shape[1])
        end = min(max(self.batch_corr_lead_end, start + 1), pred.shape[1])
        if end - start < 1 or pred.shape[0] < 4:
            return pred.new_zeros(())

        p = pred[:, start:end, 0].float()
        t = target[:, start:end, 0].float()
        p_centered = p - p.mean(dim=0, keepdim=True)
        t_centered = t - t.mean(dim=0, keepdim=True)
        numerator = (p_centered * t_centered).sum(dim=0)
        p_norm = (p_centered.square().sum(dim=0) + 1.0e-6).sqrt()
        t_norm = (t_centered.square().sum(dim=0) + 1.0e-6).sqrt()
        corr = numerator / (p_norm * t_norm)
        return (1.0 - corr).mean()

    def _long_lead_weights(self, length: int, device) -> torch.Tensor:
        """Return normalized weights that mildly emphasize the long horizon."""
        weights = torch.ones(length, device=device)
        if self.long_lead_weight <= 0.0 or length <= 0:
            return weights
        start = min(max(self.long_lead_start - 1, 0), length - 1)
        end = min(max(self.long_lead_end, start + 1), length)
        ramp = torch.linspace(0.0, 1.0, end - start, device=device)
        weights[start:end] = 1.0 + self.long_lead_weight * ramp
        return weights / weights.mean().clamp_min(1.0e-6)

    def _long_lead_scale_loss(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Match long-lead spread and regression slope without changing ACC."""
        start = min(max(self.scale_lead_start - 1, 0), pred.shape[1])
        end = min(max(self.scale_lead_end, start + 1), pred.shape[1])
        if end <= start or pred.shape[0] < 4:
            return pred.new_zeros(())
        p = pred[:, start:end, 0].float()
        t = target[:, start:end, 0].float()
        p_centered = p - p.mean(dim=0, keepdim=True)
        t_centered = t - t.mean(dim=0, keepdim=True)
        p_var = p_centered.square().mean(dim=0)
        t_var = t_centered.square().mean(dim=0)
        p_std = (p_var + 1.0e-6).sqrt()
        t_std = (t_var + 1.0e-6).sqrt()
        desired_std = self.scale_target_ratio * t_std.clamp_min(1.0e-4)
        spread_error = torch.log(p_std / desired_std.clamp_min(1.0e-4)).square()
        covariance = (p_centered * t_centered).mean(dim=0)
        slope = covariance / t_var.clamp_min(1.0e-4)
        slope_error = (slope - self.scale_slope_target).square()
        return spread_error.mean() + 0.5 * slope_error.mean()

    def _long_lead_phase_loss(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Encourage the correct ENSO sign for sufficiently strong events."""
        start = min(max(self.phase_lead_start - 1, 0), pred.shape[1])
        end = min(max(self.phase_lead_end, start + 1), pred.shape[1])
        if end <= start:
            return pred.new_zeros(())
        p = pred[:, start:end, 0].float()
        t = target[:, start:end, 0].float()
        event = (t.abs() >= self.phase_threshold).to(dtype=p.dtype)
        if not bool(event.any()):
            return pred.new_zeros(())
        signed_prediction = p * t.sign()
        penalty = F.softplus(self.phase_margin - signed_prediction) * event
        return penalty.sum() / event.sum().clamp_min(1.0)

    def _phase_drift_loss(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Match low-frequency lead-to-lead tendencies at long horizons.

        A two-month centered difference is less sensitive to monthly noise than
        a one-step derivative, while still penalizing a systematic phase lag in
        the predicted ENSO trajectory.
        """
        start = min(max(self.phase_drift_lead_start - 1, 0), pred.shape[1])
        end = min(max(self.phase_drift_lead_end, start + 1), pred.shape[1])
        if end - start < 3:
            return pred.new_zeros(())
        p = pred[:, start:end, 0].float()
        t = target[:, start:end, 0].float()
        p_delta = p[:, 2:] - p[:, :-2]
        t_delta = t[:, 2:] - t[:, :-2]
        return F.smooth_l1_loss(p_delta, t_delta, beta=0.5)

    def _seasonal_mean_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        init_month=None,
        model_diag=None,
    ) -> torch.Tensor:
        """Keep standardized anomaly forecasts from learning a calendar bias."""
        if init_month is None:
            return pred.new_zeros(())
        start = min(max(self.seasonal_mean_lead_start - 1, 0), pred.shape[1])
        end = min(max(self.seasonal_mean_lead_end, start + 1), pred.shape[1])
        if end <= start:
            return pred.new_zeros(())
        target_months = None
        if isinstance(model_diag, dict):
            target_months = model_diag.get("target_months")
        if not torch.is_tensor(target_months) or target_months.shape[:2] != pred.shape[:2]:
            leads = torch.arange(1, pred.shape[1] + 1, device=pred.device).view(1, -1)
            target_months = (init_month.view(-1, 1).long() + leads) % 12
        target_months = target_months.to(device=pred.device, dtype=torch.long)
        errors = []
        p = pred[:, start:end, 0].float()
        t = target[:, start:end, 0].float()
        months = target_months[:, start:end]
        for lead in range(end - start):
            for month in range(12):
                selected = months[:, lead] == month
                if int(selected.sum().item()) >= 2:
                    errors.append((p[selected, lead].mean() - t[selected, lead].mean()).square())
        if not errors:
            return pred.new_zeros(())
        return torch.stack(errors).mean()

    def _selective_loss(
        self,
        nino_loss_raw: torch.Tensor,
        model_diag=None,
    ) -> torch.Tensor:
        """Downweight high-uncertainty long-lead samples after NLL warm-up."""
        if not isinstance(model_diag, dict):
            return nino_loss_raw.new_zeros(())
        logvar = model_diag.get("final_logvar")
        if not torch.is_tensor(logvar) or logvar.ndim < 3:
            return nino_loss_raw.new_zeros(())
        start = min(max(self.selective_lead_start - 1, 0), nino_loss_raw.shape[1])
        end = min(max(self.selective_lead_end, start + 1), nino_loss_raw.shape[1])
        if end <= start:
            return nino_loss_raw.new_zeros(())
        uncertainty = logvar[..., 0].float().detach()[:, start:end].clamp(-2.0, 2.0)
        confidence = torch.exp(-0.5 * uncertainty)
        confidence = confidence / confidence.mean(dim=0, keepdim=True).clamp_min(1.0e-4)
        confidence = confidence.clamp_min(self.selective_min_confidence)
        weighted = (nino_loss_raw[:, start:end].float() * confidence).mean()
        baseline = nino_loss_raw[:, start:end].float().mean()
        return weighted - baseline

    @staticmethod
    def _diag_first(model_diag, *names):
        if not isinstance(model_diag, dict):
            return None
        for name in names:
            value = model_diag.get(name, None)
            if value is not None:
                return value
        return None

    @staticmethod
    def _diag_scalar(value, reference: torch.Tensor, square: bool = False) -> torch.Tensor:
        if value is None:
            return reference.new_zeros(())
        if not torch.is_tensor(value):
            value = reference.new_tensor(float(value))
        else:
            value = value.to(device=reference.device, dtype=reference.dtype)
        if value.numel() == 0:
            return reference.new_zeros(())
        value = torch.nan_to_num(value, nan=0.0, posinf=1e4, neginf=-1e4)
        return value.pow(2).mean() if square else value.mean()

    def _base_forecast_loss_per_sample(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        init_month=None,
        lead_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Niño + tapered auxiliary loss before reducing over the batch."""
        if pred.ndim != 3 or target.ndim != 3:
            raise ValueError(
                "MIST forecast losses expect (batch, lead, target) tensors, "
                f"got pred={tuple(pred.shape)}, target={tuple(target.shape)}"
            )
        B = min(pred.shape[0], target.shape[0])
        L = min(pred.shape[1], target.shape[1])
        D = min(pred.shape[2], target.shape[2])
        pred = pred[:B, :L, :D]
        target = target[:B, :L, :D]

        nino_raw = F.smooth_l1_loss(
            pred[..., 0], target[..., 0], beta=self.huber_beta, reduction="none"
        )
        if init_month is not None:
            months = init_month[:B].to(device=pred.device, dtype=torch.long)
            nino_raw = nino_raw * self._spb_month_weights(months, L, pred.device)
        if lead_weights is not None:
            nino_raw = nino_raw * lead_weights[:L].view(1, L)
        per_sample = nino_raw.mean(dim=1)

        if D >= 2:
            aux_raw = F.smooth_l1_loss(
                pred[..., 1], target[..., 1], beta=self.huber_beta, reduction="none"
            )
            taper = self._aux_taper_weights(L, pred.device).view(1, L)
            if lead_weights is not None:
                taper = taper * lead_weights[:L].view(1, L)
            per_sample = per_sample + self.aux_weight * (aux_raw * taper).mean(dim=1)
        return per_sample

    @staticmethod
    def _masked_sample_mean(per_sample: torch.Tensor, mask) -> torch.Tensor:
        if mask is None:
            return per_sample.mean()
        if not torch.is_tensor(mask):
            mask = torch.as_tensor(mask, device=per_sample.device)
        mask = mask.to(device=per_sample.device)
        if mask.ndim == 0:
            mask = mask.expand(per_sample.shape[0])
        elif mask.shape[0] != per_sample.shape[0]:
            return per_sample.mean()
        elif mask.ndim > 1:
            mask = mask.reshape(mask.shape[0], -1).float().mean(dim=1)
        weights = mask.float().clamp_min(0.0)
        denom = weights.sum()
        if float(denom.detach().item()) <= 0:
            return per_sample.sum() * 0.0
        return (per_sample * weights).sum() / denom

    def _mist_group_dro_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        init_month,
        domain_id,
        lead_weights=None,
    ) -> torch.Tensor:
        """Smooth worst-domain risk over domains present in the current batch."""
        if domain_id is None:
            return pred.new_zeros(())
        if not torch.is_tensor(domain_id):
            domain_id = torch.as_tensor(domain_id, device=pred.device)
        domain_id = domain_id.to(device=pred.device, dtype=torch.long).reshape(-1)
        per_sample = self._base_forecast_loss_per_sample(
            pred, target, init_month, lead_weights=lead_weights
        )
        B = per_sample.shape[0]
        if domain_id.numel() != B:
            return pred.new_zeros(())

        valid = domain_id >= 0
        if not bool(valid.any()):
            return pred.new_zeros(())
        group_losses = []
        for domain in torch.unique(domain_id[valid], sorted=True):
            members = valid & (domain_id == domain)
            if bool(members.any()):
                group_losses.append(per_sample[members].mean())
        if not group_losses:
            return pred.new_zeros(())
        risks = torch.stack(group_losses)
        if self.mist_group_dro_mode == "worst":
            return risks.max()

        tau = self.mist_group_dro_temperature
        normalizer = torch.log(risks.new_tensor(float(risks.numel())))
        return tau * (torch.logsumexp(risks / tau, dim=0) - normalizer)

    def _innovation_whiteness_loss(self, innovation, reference: torch.Tensor) -> torch.Tensor:
        """Fallback for raw innovations when the model did not emit a penalty."""
        if innovation is None:
            return reference.new_zeros(())
        if not torch.is_tensor(innovation):
            innovation = torch.as_tensor(innovation, device=reference.device, dtype=reference.dtype)
        else:
            innovation = innovation.to(device=reference.device, dtype=reference.dtype)
        if innovation.numel() == 0:
            return reference.new_zeros(())
        innovation = torch.nan_to_num(innovation)
        if innovation.ndim < 2:
            return innovation.mean().pow(2)
        x = innovation.reshape(innovation.shape[0], innovation.shape[1], -1)
        mean_penalty = x.mean(dim=(0, 1)).pow(2).mean()
        if x.shape[1] < 2:
            return mean_penalty
        centered = x - x.mean(dim=(0, 1), keepdim=True)
        lag_cov = (centered[:, 1:] * centered[:, :-1]).mean(dim=(0, 1))
        variance = centered.pow(2).mean(dim=(0, 1)).clamp_min(1e-6)
        lag_corr = lag_cov / variance
        return mean_penalty + lag_corr.pow(2).mean()

    def _mist_warmup_scale(self) -> float:
        if self.mist_loss_warmup_epochs <= 0:
            return 1.0
        epoch = int(self.current_epoch.item())
        return min(1.0, float(epoch + 1) / float(self.mist_loss_warmup_epochs))

    def _mist_v2_lead_weights(self, length: int, device) -> torch.Tensor:
        """Smoothly expand supervised rollout length from short to 24 months."""

        if self.mist_horizon_curriculum_epochs <= 0 or length <= self.mist_horizon_start:
            return torch.ones(length, device=device)
        epoch = int(self.current_epoch.item())
        progress = min(
            1.0, float(epoch + 1) / float(self.mist_horizon_curriculum_epochs)
        )
        active_lead = self.mist_horizon_start + progress * (
            length - self.mist_horizon_start
        )
        leads = torch.arange(1, length + 1, device=device, dtype=torch.float32)
        weights = torch.sigmoid((active_lead - leads) / 1.5)
        return weights / weights.mean().clamp_min(1.0e-6)

    def loss_components(self, pred, target, init_month=None, model_diag=None):
        """Compute all loss components."""
        B, L, D = pred.shape
        device = pred.device
        components = {}
        is_mist_v2 = self._diag_first(model_diag, "mist_v2_model") is not None
        mist_lead_weights = (
            self._mist_v2_lead_weights(L, device) if is_mist_v2 else None
        )

        nino_pred = pred[..., 0]
        nino_true = target[..., 0]
        nino_loss_raw = F.smooth_l1_loss(nino_pred, nino_true, beta=self.huber_beta, reduction='none')

        if init_month is not None:
            spb_w = self._spb_month_weights(init_month, L, device)
            spring_init_w = self._spring_initialization_weights(init_month, L, device)
        else:
            spb_w = torch.ones(B, L, device=device)
            spring_init_w = torch.ones(B, L, device=device)
        lead_weights = self._long_lead_weights(L, device).view(1, L)
        weighted_nino = nino_loss_raw * spb_w * spring_init_w * lead_weights
        if mist_lead_weights is not None:
            weighted_nino = weighted_nino * mist_lead_weights.view(1, L)
        nino_loss = weighted_nino.mean()
        components["nino34"] = nino_loss

        if D >= 2:
            aux_pred = pred[..., 1]
            aux_true = target[..., 1]
            aux_loss_raw = F.smooth_l1_loss(aux_pred, aux_true, beta=self.huber_beta, reduction='none')
            aux_taper = self._aux_taper_weights(L, device).view(1, L)
            if mist_lead_weights is not None:
                aux_taper = aux_taper * mist_lead_weights.view(1, L)
            aux_loss = (aux_loss_raw * aux_taper).mean()
            components["aux"] = self.aux_weight * aux_loss
        else:
            components["aux"] = torch.tensor(0.0, device=device)

        if self.ode_weight > 0 and D >= 2:
            ode_loss = self._ode_loss(pred)
            components["ode"] = self.ode_weight * ode_loss
        else:
            components["ode"] = torch.tensor(0.0, device=device)

        if self.smoothness_weight > 0:
            components["smooth"] = self.smoothness_weight * self._smoothness_loss(pred[..., :1])
        else:
            components["smooth"] = torch.tensor(0.0, device=device)

        if self.trend_weight > 0:
            components["trend"] = self.trend_weight * self._trend_loss(pred, target)
        else:
            components["trend"] = torch.tensor(0.0, device=device)

        if self.adaptive_weight > 0:
            with torch.no_grad():
                sample_losses = nino_loss_raw.mean(dim=1)
                self.ema_loss = self.adaptive_ema * self.ema_loss + (1 - self.adaptive_ema) * sample_losses.mean()
                hard_mask = (sample_losses > self.ema_loss).float()
                scale = 1.0 + self.adaptive_weight * hard_mask
                scale = scale.clamp(max=self.max_adaptive_scale)
            components["adaptive"] = (nino_loss_raw * scale.unsqueeze(1)).mean() - nino_loss
        else:
            components["adaptive"] = torch.tensor(0.0, device=device)

        if self.nll_weight > 0 and model_diag is not None:
            final_logvar = model_diag.get("final_logvar", None)
            if final_logvar is None:
                logvar_hist = model_diag.get("thought_logvars", None)
                halting_dist = model_diag.get("halting_dist", None)
                if logvar_hist is not None:
                    if halting_dist is not None and halting_dist.shape[1] == logvar_hist.shape[1]:
                        w = halting_dist.view(B, -1, 1, 1)
                        final_logvar = (logvar_hist * w).sum(dim=1)
                    else:
                        final_logvar = logvar_hist[:, -1]
                else:
                    final_logvar = None

            if final_logvar is not None:
                epoch = int(self.current_epoch.item())
                warmup_scale = min(1.0, epoch / max(self.nll_warmup_epochs, 1))
                nll_loss = self._nll_loss(pred, target, final_logvar)
                components["nll"] = self.nll_weight * warmup_scale * nll_loss
            else:
                components["nll"] = torch.tensor(0.0, device=device)
        else:
            components["nll"] = torch.tensor(0.0, device=device)

        if self.amplitude_weight > 0:
            components["amplitude"] = self.amplitude_weight * self._amplitude_loss(pred, target)
        else:
            components["amplitude"] = torch.tensor(0.0, device=device)

        if self.corr_weight > 0:
            components["corr"] = self.corr_weight * self._correlation_loss(pred, target)
        else:
            components["corr"] = torch.tensor(0.0, device=device)

        if self.batch_corr_weight > 0.0:
            epoch = int(self.current_epoch.item())
            warmup = min(
                1.0,
                float(epoch + 1) / max(self.batch_corr_warmup_epochs, 1),
            )
            components["batch_corr"] = (
                self.batch_corr_weight
                * warmup
                * self._batch_lead_correlation_loss(pred, target)
            )
        else:
            components["batch_corr"] = torch.tensor(0.0, device=device)

        if self.scale_weight > 0.0:
            epoch = int(self.current_epoch.item())
            scale_warmup = min(
                1.0,
                float(epoch + 1) / max(self.scale_warmup_epochs, 1),
            )
            components["scale"] = self.scale_weight * self._long_lead_scale_loss(
                pred, target
            ) * scale_warmup
        else:
            components["scale"] = torch.tensor(0.0, device=device)

        if self.phase_weight > 0.0:
            epoch = int(self.current_epoch.item())
            phase_warmup = min(
                1.0,
                float(epoch + 1) / max(self.phase_warmup_epochs, 1),
            )
            components["phase"] = self.phase_weight * self._long_lead_phase_loss(
                pred, target
            ) * phase_warmup
        else:
            components["phase"] = torch.tensor(0.0, device=device)

        if self.phase_drift_weight > 0.0:
            epoch = int(self.current_epoch.item())
            drift_warmup = min(
                1.0,
                float(epoch + 1) / max(self.phase_drift_warmup_epochs, 1),
            )
            components["phase_drift"] = (
                self.phase_drift_weight
                * drift_warmup
                * self._phase_drift_loss(pred, target)
            )
        else:
            components["phase_drift"] = torch.tensor(0.0, device=device)

        if self.seasonal_mean_weight > 0.0:
            epoch = int(self.current_epoch.item())
            seasonal_warmup = min(
                1.0,
                float(epoch + 1) / max(self.seasonal_mean_warmup_epochs, 1),
            )
            components["seasonal_mean"] = (
                self.seasonal_mean_weight
                * seasonal_warmup
                * self._seasonal_mean_loss(
                    pred, target, init_month=init_month, model_diag=model_diag
                )
            )
        else:
            components["seasonal_mean"] = torch.tensor(0.0, device=device)

        if self.selective_weight > 0.0:
            epoch = int(self.current_epoch.item())
            selective_warmup = min(
                1.0,
                float(epoch + 1) / max(self.selective_warmup_epochs, 1),
            )
            components["selective"] = (
                self.selective_weight
                * selective_warmup
                * self._selective_loss(nino_loss_raw, model_diag=model_diag)
            )
        else:
            components["selective"] = torch.tensor(0.0, device=device)

        if self.ponder_weight > 0 and model_diag is not None:
            ponder_cost = model_diag.get("ponder_cost", None)
            if ponder_cost is not None:
                components["ponder"] = self.ponder_weight * ponder_cost
            else:
                components["ponder"] = torch.tensor(0.0, device=device)
        else:
            components["ponder"] = torch.tensor(0.0, device=device)

        if model_diag is not None:
            entropy_penalty = model_diag.get("halt_entropy_penalty", None)
            if entropy_penalty is not None:
                components["halt_entropy"] = self.halt_entropy_weight * entropy_penalty
            else:
                components["halt_entropy"] = torch.tensor(0.0, device=device)
        else:
            components["halt_entropy"] = torch.tensor(0.0, device=device)

        if self.overshoot_weight > 0:
            components["overshoot"] = self.overshoot_weight * self._overshoot_loss(pred, target)
        else:
            components["overshoot"] = torch.tensor(0.0, device=device)

        if model_diag is not None and (self.pinn_weight > 0 or self.pinn_energy_weight > 0):
            epoch = int(self.current_epoch.item())
            warmup_scale = min(1.0, epoch / max(self.pinn_warmup_epochs, 1))

            pinn_residual = model_diag.get("pinn_residual", None)
            if pinn_residual is not None:
                components["pinn"] = self.pinn_weight * warmup_scale * pinn_residual
            else:
                components["pinn"] = torch.tensor(0.0, device=device)

            pinn_energy = model_diag.get("pinn_energy", None)
            if pinn_energy is not None:
                components["pinn_energy"] = self.pinn_energy_weight * warmup_scale * pinn_energy
            else:
                components["pinn_energy"] = torch.tensor(0.0, device=device)
        else:
            components["pinn"] = torch.tensor(0.0, device=device)
            components["pinn_energy"] = torch.tensor(0.0, device=device)

        if model_diag is not None:
            zero = torch.tensor(0.0, device=device)
            prior = model_diag.get("pinn_prior", None)
            components["pinn_prior"] = (
                self.pinn_prior_weight * prior if (self.pinn_prior_weight > 0 and prior is not None) else zero
            )
            anchor = model_diag.get("latent_anchor", None)
            components["latent_anchor"] = (
                self.latent_anchor_weight * anchor if (self.latent_anchor_weight > 0 and anchor is not None) else zero
            )
            cf = model_diag.get("cf_violation", None)
            if self.cf_weight > 0 and cf is not None:
                epoch = int(self.current_epoch.item())
                cf_scale = min(1.0, epoch / max(self.cf_warmup_epochs, 1))
                components["cf"] = self.cf_weight * cf_scale * cf
            else:
                components["cf"] = zero
            gate = model_diag.get("gate_l1", None)
            components["gate_l1"] = (
                self.gate_l1_weight * gate if (self.gate_l1_weight > 0 and gate is not None) else zero
            )
        else:
            components["pinn_prior"] = torch.tensor(0.0, device=device)
            components["latent_anchor"] = torch.tensor(0.0, device=device)
            components["cf"] = torch.tensor(0.0, device=device)
            components["gate_l1"] = torch.tensor(0.0, device=device)

        mist_scale = self._mist_warmup_scale()
        zero = pred.new_zeros(())

        domain_penalty = self._diag_first(
            model_diag,
            "domain_residual_penalty",
            "domain_residual_loss",
            "adapter_penalty",
        )
        if domain_penalty is None:
            raw_residual = self._diag_first(
                model_diag, "domain_residual", "domain_residual_values"
            )
            domain_penalty = (
                self._diag_scalar(raw_residual, pred, square=True)
                if raw_residual is not None else None
            )
        components["mist_domain_residual"] = (
            self.mist_domain_residual_weight
            * mist_scale
            * self._diag_scalar(domain_penalty, pred)
            if self.mist_domain_residual_weight > 0 and domain_penalty is not None
            else zero
        )

        phase_penalty = self._diag_first(
            model_diag,
            "phase_condition_penalty",
            "phase_condition_loss",
            "phase_regularizer",
        )
        if phase_penalty is None:
            condition = self._diag_first(
                model_diag, "phase_condition_number", "phase_condition"
            )
            if condition is not None:
                if not torch.is_tensor(condition):
                    condition = pred.new_tensor(float(condition))
                else:
                    condition = condition.to(device=device, dtype=pred.dtype)
                condition = torch.nan_to_num(condition, nan=self.mist_phase_condition_target,
                                              posinf=1e4, neginf=1.0).clamp_min(1.0)
                threshold = condition.new_tensor(self.mist_phase_condition_target)
                phase_penalty = F.relu(torch.log(condition) - torch.log(threshold)).pow(2).mean()
        components["mist_phase_condition"] = (
            self.mist_phase_condition_weight
            * mist_scale
            * self._diag_scalar(phase_penalty, pred)
            if self.mist_phase_condition_weight > 0 and phase_penalty is not None
            else zero
        )

        innovation_penalty = self._diag_first(
            model_diag,
            "innovation_penalty",
            "innovation_loss",
            "innovation_nll",
            "innovation_whiteness_penalty",
        )
        if innovation_penalty is None:
            raw_innovation = self._diag_first(model_diag, "innovation", "innovations")
            innovation_penalty = (
                self._innovation_whiteness_loss(raw_innovation, pred)
                if raw_innovation is not None else None
            )
        components["mist_innovation"] = (
            self.mist_innovation_weight
            * mist_scale
            * self._diag_scalar(innovation_penalty, pred)
            if self.mist_innovation_weight > 0 and innovation_penalty is not None
            else zero
        )

        history_observed = self._diag_first(model_diag, "history_observed_loss")
        history_latent = self._diag_first(model_diag, "history_latent_loss")
        history_core = self._diag_first(model_diag, "history_core_observed_loss")
        components["mist_history_observed"] = (
            self.mist_history_observed_weight
            * mist_scale
            * self._diag_scalar(history_observed, pred)
            if self.mist_history_observed_weight > 0 and history_observed is not None
            else zero
        )
        components["mist_history_latent"] = (
            self.mist_history_latent_weight
            * mist_scale
            * self._diag_scalar(history_latent, pred)
            if self.mist_history_latent_weight > 0 and history_latent is not None
            else zero
        )
        components["mist_history_core"] = (
            self.mist_history_core_weight
            * mist_scale
            * self._diag_scalar(history_core, pred)
            if self.mist_history_core_weight > 0 and history_core is not None
            else zero
        )
        month_losses = self._diag_first(model_diag, "history_month_losses")
        month_present = self._diag_first(model_diag, "history_month_present")
        if (
            self.mist_history_seasonal_dro_weight > 0
            and torch.is_tensor(month_losses)
        ):
            month_losses = month_losses.to(device=device, dtype=pred.dtype).reshape(-1)
            if torch.is_tensor(month_present) and month_present.numel() == month_losses.numel():
                present = month_present.to(device=device, dtype=torch.bool).reshape(-1)
                month_losses = month_losses[present]
            if month_losses.numel() > 0:
                tau = self.mist_history_seasonal_dro_temperature
                seasonal_risk = tau * (
                    torch.logsumexp(month_losses / tau, dim=0)
                    - torch.log(month_losses.new_tensor(float(month_losses.numel())))
                )
                components["mist_history_seasonal_dro"] = (
                    self.mist_history_seasonal_dro_weight
                    * mist_scale
                    * seasonal_risk
                )
            else:
                components["mist_history_seasonal_dro"] = zero
        else:
            components["mist_history_seasonal_dro"] = zero

        core_pred = self._diag_first(
            model_diag,
            "core_only_pred",
            "shared_core_pred",
            "core_pred",
            "pred_core",
        )
        if torch.is_tensor(core_pred):
            core_pred = core_pred.to(device=device, dtype=pred.dtype)
        consistency = self._diag_first(
            model_diag, "core_consistency_loss", "core_only_consistency"
        )
        if consistency is None and torch.is_tensor(core_pred):
            Bc = min(core_pred.shape[0], pred.shape[0])
            Lc = min(core_pred.shape[1], pred.shape[1])
            Dc = min(core_pred.shape[2], pred.shape[2])
            consistency = F.smooth_l1_loss(
                core_pred[:Bc, :Lc, :Dc],
                pred[:Bc, :Lc, :Dc].detach(),
                beta=self.huber_beta,
            )
        components["mist_core_consistency"] = (
            self.mist_core_consistency_weight
            * mist_scale
            * self._diag_scalar(consistency, pred)
            if self.mist_core_consistency_weight > 0 and consistency is not None
            else zero
        )

        if self.mist_core_only_weight > 0 and torch.is_tensor(core_pred):
            core_per_sample = self._base_forecast_loss_per_sample(
                core_pred, target, init_month, lead_weights=mist_lead_weights
            )
            core_mask = self._diag_first(
                model_diag,
                "core_forecast_mask",
                "core_only_mask",
                "heldout_domain_mask",
            )
            core_forecast = self._masked_sample_mean(core_per_sample, core_mask)
            components["mist_core_only"] = (
                self.mist_core_only_weight * mist_scale * core_forecast
            )
        else:
            components["mist_core_only"] = zero

        if self.mist_group_dro_weight > 0:
            diag_domain_id = self._diag_first(model_diag, "domain_id")
            group_dro = self._mist_group_dro_loss(
                pred, target, init_month, diag_domain_id,
                lead_weights=mist_lead_weights,
            )
            components["mist_group_dro"] = (
                self.mist_group_dro_weight * mist_scale * group_dro
            )
        else:
            components["mist_group_dro"] = zero

        total = sum(components.values())
        components["total"] = total
        return components

    def forward(self, pred, target, init_month=None, model_diag=None):
        return self.loss_components(pred, target, init_month=init_month, model_diag=model_diag)["total"]


def ctm_regression_thought_loss_v2(
    model_diag,
    target: torch.Tensor,
    huber_beta: float = 0.5,
    curriculum_tick_frac: float = 1.0,
) -> torch.Tensor:
    """Enhanced CTM thought loss with curriculum learning.

    Curriculum: early in training (curriculum_tick_frac < 1), only supervise
    later ticks. This gives early ticks freedom to explore before being
    constrained. As training progresses, curriculum_tick_frac → 1.0.
    """
    if not isinstance(model_diag, dict) or "thought_preds" not in model_diag:
        return torch.tensor(0.0, device=target.device)

    pred_hist = model_diag["thought_preds"]
    certainty = model_diag.get("thought_certainty", None)
    if pred_hist.ndim != 4:
        return torch.tensor(0.0, device=target.device)

    B, ticks, L, D = pred_hist.shape
    target_exp = target.unsqueeze(1).expand(B, ticks, L, D)

    min_d = min(D, target.shape[-1])
    losses = F.smooth_l1_loss(
        pred_hist[..., :min_d], target_exp[..., :min_d],
        beta=huber_beta, reduction="none"
    ).mean(dim=(2, 3))

    start_tick = max(0, min(int(ticks * (1.0 - curriculum_tick_frac)), ticks - 1))
    if start_tick > 0:
        mask = torch.zeros(ticks, device=target.device)
        mask[start_tick:] = 1.0
        losses = losses * mask.view(1, -1)

    active_losses = losses[:, start_tick:]
    if active_losses.shape[1] == 0:
        active_losses = losses[:, -1:]
        start_tick = ticks - 1
    best_idx = active_losses.argmin(dim=1) + start_tick
    b = torch.arange(B, device=target.device)

    if certainty is not None and certainty.shape[1] > start_tick:
        certain_idx = certainty[:, start_tick:].argmax(dim=1) + start_tick
    else:
        certain_idx = torch.full_like(best_idx, ticks - 1)

    halting_dist = model_diag.get("halting_dist", None)
    if halting_dist is not None:
        halt_selected = halting_dist.argmax(dim=1)
        halt_loss = losses[b, halt_selected].mean()
        return 0.4 * losses[b, best_idx].mean() + 0.3 * losses[b, certain_idx].mean() + 0.3 * halt_loss

    return 0.5 * (losses[b, best_idx].mean() + losses[b, certain_idx].mean())


ctm_thought_loss_v2 = ctm_regression_thought_loss_v2
