"""Add a reproduatmosle cold-event example from the existing paired forecast cache.

The original seven-event tables and evaluator metadata remain the record of the
original selection. Supplementary tables explicitly identify the post-hoc choice.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events_dir", type=Path, default=PROJECT_ROOT / "results_spb_core_claim" / "event_backtest")
    parser.add_argument("--event_year", type=int, default=1988)
    args = parser.parse_args()
    existing = pd.read_csv(args.events_dir / "event_hindcasts_all.csv")
    identities = existing[["model_key", "configuration", "seed"]].drop_duplicates()
    if len(identities) != 1:
        raise ValueError("This supplement requires the original single-checkpoint paired event experiment")
    identity = identities.iloc[0]
    run_dir = args.events_dir / f"{identity.model_key}_{identity.configuration}_seed{int(identity.seed)}"
    raw_path = run_dir / "predictions_raw.npz"
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    target_index = metadata["target_vars"].index("nino34")
    masks = sorted(existing.mask_name.unique())
    if args.event_year in set(existing.event_year):
        raise ValueError("The additional year is already in the original event list")

    with np.load(raw_path, allow_pickle=False) as raw:
        dates = pd.to_datetime(raw["sample_dates"], format="%Y-%m")
        truth = raw["truth"][..., target_index]
        predictions = {mask: raw[f"pred__{mask}"][..., target_index] for mask in masks}
        if truth.shape != (len(dates), 24) or any(p.shape != truth.shape for p in predictions.values()):
            raise ValueError("Expected complete paired 24-month forecasts")
        if not np.isfinite(truth).all() or any(not np.isfinite(p).all() for p in predictions.values()):
            raise ValueError("Non-finite cached forecasts or observations")
        if not dates.is_unique or not np.array_equal(dates.month - 1, raw["init_month"]):
            raise ValueError("Invalid cached initialization dates")

        for mask, frame in existing.groupby("mask_name"):
            sample_ids = frame.sample_id.to_numpy(int)
            lead_ids = frame.lead.to_numpy(int) - 1
            np.testing.assert_array_equal(dates[sample_ids].strftime("%Y-%m"), frame.init_date.to_numpy())
            np.testing.assert_allclose(predictions[mask][sample_ids, lead_ids], frame.pred_nino34, rtol=0, atol=1e-12)
            np.testing.assert_allclose(truth[sample_ids, lead_ids], frame.true_nino34, rtol=0, atol=1e-12)
            for row in frame.itertuples():
                expected = dates[row.sample_id] + pd.DateOffset(months=row.lead)
                if expected.strftime("%Y-%m") != row.target_date:
                    raise ValueError("Original target dates do not match cached initialization dates")

        screening_rows = []
        for sample_id, date in enumerate(dates):
            if date.month != 4:
                continue
            y = truth[sample_id]
            djf_anomaly = float(y[7:10].mean())
            if djf_anomaly > -0.5:
                continue
            p = predictions["full"][sample_id]
            b = predictions["state_only"][sample_id]
            screening_rows.append({
                "event_year": int(date.year),
                "init_date": date.strftime("%Y-%m"),
                "observed_djf_nino34": djf_anomaly,
                "predicted_djf_nino34": float(p[7:10].mean()),
                "full_rmse_1_24": float(np.sqrt(np.mean((p - y) ** 2))),
                "full_rmse_6_18": float(np.sqrt(np.mean((p[5:18] - y[5:18]) ** 2))),
                "full_mae_1_24": float(np.mean(np.abs(p - y))),
                "full_trajectory_r_1_24": float(np.corrcoef(p, y)[0, 1]),
                "index_only_rmse_1_24": float(np.sqrt(np.mean((b - y) ** 2))),
                "already_in_original_figure": int(date.year) in set(existing.event_year),
                "added_to_figure": int(date.year) == args.event_year,
            })
        screening = pd.DataFrame(screening_rows).sort_values(["full_rmse_1_24", "event_year"]).reset_index(drop=True)
        screening.insert(0, "rmse_rank", np.arange(1, len(screening) + 1))
        if args.event_year not in set(screening.event_year):
            raise ValueError("The requested year does not meet the observed cold-event screening criterion")
        sample_id = int(np.flatnonzero((dates.year == args.event_year) & (dates.month == 4))[0])
        date = dates[sample_id]
        rows = []
        for mask in masks:
            for lead_id in range(truth.shape[1]):
                rows.append({
                    "model_key": str(identity.model_key),
                    "configuration": str(identity.configuration),
                    "seed": int(identity.seed),
                    "mask_name": mask,
                    "sample_id": sample_id,
                    "event_year": args.event_year,
                    "init_date": date.strftime("%Y-%m"),
                    "target_date": (date + pd.DateOffset(months=lead_id + 1)).strftime("%Y-%m"),
                    "lead": lead_id + 1,
                    "pred_nino34": float(predictions[mask][sample_id, lead_id]),
                    "true_nino34": float(truth[sample_id, lead_id]),
                })

    selection = "post_hoc_lowest_full_rmse_1_24" if int(screening.iloc[0].event_year) == args.event_year else "post_hoc_requested_cold_event"
    pd.DataFrame(rows, columns=existing.columns).to_csv(args.events_dir / "event_hindcasts_additional.csv", index=False)
    pd.DataFrame([{"selection": selection, "event_year": args.event_year, "phase": "cold"}]).to_csv(args.events_dir / "selected_additional_events.csv", index=False)
    screening.to_csv(args.events_dir / "cold_event_screening.csv", index=False)
    provenance = {
        "additional_event_year": args.event_year,
        "selection": selection,
        "criterion": "April initialization; observed first December-February mean Nino3.4 <= -0.5 degC; rank full-mask RMSE over leads 1-24",
        "candidate_count": len(screening),
        "source_predictions": str(raw_path.relative_to(PROJECT_ROOT)) if raw_path.is_relative_to(PROJECT_ROOT) else str(raw_path),
        "source_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        "checkpoint_selection": metadata["checkpoint_selection"],
        "phys_vars": metadata["phys_vars"],
        "target_vars": metadata["target_vars"],
        "seed": int(identity.seed),
        "mask_names": masks,
        "forecast_months": 24,
        "forecast_cache_matches_original_events": True,
        "retrained_or_reselected_checkpoint": False,
        "interpretation": "Post-hoc illustrative cold-event example from the existing OBS-best exploratory checkpoint; not an independent validation or an official event catalogue.",
    }
    (args.events_dir / "additional_event_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(screening.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"[Event supplement] wrote {len(rows)} paired forecast rows for {args.event_year}")


if __name__ == "__main__":
    main()
