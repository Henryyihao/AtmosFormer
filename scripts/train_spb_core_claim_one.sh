set -euo pipefail


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

PYTHON="${PYTHON:-python}"
ACTION="${ACTION:-train}"
MODEL_NAME="${MODEL_NAME:-AtmosFormerSPB}"
CONFIG_ID="${CONFIG_ID:-global_slp_sst_tauu}"
CLAIM_FAMILY="${CLAIM_FAMILY:-global}"
CLAIM_ROLE="${CLAIM_ROLE:-reference}"
MAP_VARS="${MAP_VARS:-slp,sst,hc300,tauu}"
PHYS_VARS="${PHYS_VARS:-nino34,wwv,thermocline_tilt}"
SPATIAL_MASK_SPEC="${SPATIAL_MASK_SPEC:-full}"
SEED="${SEED:-2025}"
EXP_TAG="${EXP_TAG:-SPBCore_${MODEL_NAME}_${CONFIG_ID}_seed${SEED}}"

DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/data}"
SAVE_DIR="${SAVE_DIR:-${PROJECT_ROOT}/checkpoints_spb_core_claim}"
VISUAL_DIR="${VISUAL_DIR:-${PROJECT_ROOT}/results_spb_core_claim_training}"
METADATA_DIR="${METADATA_DIR:-${PROJECT_ROOT}/results_spb_core_claim/metadata}"

EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
PERSISTENT_WORKERS="${PERSISTENT_WORKERS:-1}"
AMP="${AMP:-1}"
COMPILE="${COMPILE:-0}"
BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-1000}"
RUN_ANALYSIS="${RUN_ANALYSIS:-1}"

case "${ACTION}" in
  train|test|analysis) ;;
  *) echo "ACTION must be train, test, or analysis; received ${ACTION}" >&2; exit 2 ;;
esac
if [[ ! "${BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "BATCH_SIZE must be a positive integer" >&2
  exit 2
fi
if [[ ! "${NUM_WORKERS}" =~ ^[0-9]+$ ]]; then
  echo "NUM_WORKERS must be a non-negative integer" >&2
  exit 2
fi
if (( NUM_WORKERS > 0 )) && [[ ! "${PREFETCH_FACTOR}" =~ ^[1-9][0-9]*$ ]]; then
  echo "PREFETCH_FACTOR must be positive when NUM_WORKERS>0" >&2
  exit 2
fi

case "${MODEL_NAME}" in
  AtmosFormerSPB|AtmosFormer)
    MODEL_NAME="AtmosFormerSPB"
    MODEL_KEY="atmosformer"
    MODEL_ARGS=(
      --model_name AtmosFormerSPB
      --atmos_d_model "${ATMOS_D_MODEL:-96}"
      --atmos_n_heads "${ATMOS_N_HEADS:-4}"
      --atmos_pool_lat "${ATMOS_POOL_LAT:-10}"
      --atmos_pool_lon "${ATMOS_POOL_LON:-20}"
      --atmos_temporal_depth "${ATMOS_TEMPORAL_DEPTH:-2}"
      --atmos_spatial_depth "${ATMOS_SPATIAL_DEPTH:-2}"
      --atmos_group_depth "${ATMOS_GROUP_DEPTH:-2}"
      --atmos_decoder_depth "${ATMOS_DECODER_DEPTH:-1}"
      --atmos_ffn_mult "${ATMOS_FFN_MULT:-3}"
      --atmos_dropout "${ATMOS_DROPOUT:-0.12}"
      --atmos_coalition_probability "${ATMOS_COALITION_PROBABILITY:-1.0}"
      --atmos_mask_strategy balanced
      --atmos_full_mask_probability "${ATMOS_FULL_MASK_PROBABILITY:-0.55}"
      --atmos_leave_basin_probability "${ATMOS_LEAVE_BASIN_PROBABILITY:-0.20}"
      --atmos_leave_variable_probability "${ATMOS_LEAVE_VARIABLE_PROBABILITY:-0.15}"
      --atmos_group_drop "${ATMOS_GROUP_DROP:-0.04}"
      --atmos_basin_drop "${ATMOS_BASIN_DROP:-0.02}"
      --atmos_variable_drop "${ATMOS_VARIABLE_DROP:-0.02}"
      --atmos_spring_scale "${ATMOS_SPRING_SCALE:-0.12}"
      --atmos_phase_summary_scale "${ATMOS_PHASE_SUMMARY_SCALE:-0.20}"
      --atmos_router_prior_strength "${ATMOS_ROUTER_PRIOR_STRENGTH:-0.85}"
      --atmos_multiscale_phase
      --atmos_phase_film
      --atmos_phase_film_scale "${ATMOS_PHASE_FILM_SCALE:-0.18}"
      --atmos_phase_film_lead_power "${ATMOS_PHASE_FILM_LEAD_POWER:-1.0}"
      --atmos_target_calendar_decay "${ATMOS_TARGET_CALENDAR_DECAY:-0.65}"
      --atmos_anomaly_gain_scale "${ATMOS_ANOMALY_GAIN_SCALE:-0.40}"
      --atmos_anomaly_offset_scale "${ATMOS_ANOMALY_OFFSET_SCALE:-0.25}"
      --atmos_regime_moe
      --atmos_horizon_router
      --atmos_anomaly_calibration
    )
    ;;
  ENSOCNN|CNN)
    MODEL_NAME="ENSOCNN"
    MODEL_KEY="cnn"
    MODEL_ARGS=(
      --model_name ENSOCNN
      --baseline_d_model "${CNN_D_MODEL:-128}"
      --baseline_dropout "${CNN_DROPOUT:-0.15}"
      --cnn_width "${CNN_WIDTH:-48}"
      --cnn_layers "${CNN_LAYERS:-3}"
      --cnn_temporal_layers "${CNN_TEMPORAL_LAYERS:-2}"
    )
    ;;
  ENSOConvLSTM|ConvLSTM)
    MODEL_NAME="ENSOConvLSTM"
    MODEL_KEY="convlstm"
    MODEL_ARGS=(
      --model_name ENSOConvLSTM
      --baseline_d_model "${CONVLSTM_D_MODEL:-128}"
      --baseline_dropout "${CONVLSTM_DROPOUT:-0.15}"
      --convlstm_hidden_channels "${CONVLSTM_HIDDEN_CHANNELS:-48}"
      --convlstm_layers "${CONVLSTM_LAYERS:-2}"
      --convlstm_pool_lat "${CONVLSTM_POOL_LAT:-16}"
      --convlstm_pool_lon "${CONVLSTM_POOL_LON:-32}"
      --convlstm_kernel_size "${CONVLSTM_KERNEL_SIZE:-3}"
    )
    ;;
  ENSOGeoformer|Geoformer)
    MODEL_NAME="ENSOGeoformer"
    MODEL_KEY="geoformer"
    MODEL_ARGS=(
      --model_name ENSOGeoformer
      --baseline_d_model "${GEO_D_MODEL:-128}"
      --baseline_n_heads "${GEO_HEADS:-4}"
      --baseline_depth "${GEO_DEPTH:-3}"
      --baseline_ffn_mult "${GEO_FFN_MULT:-4}"
      --baseline_dropout "${GEO_DROPOUT:-0.15}"
      --geoformer_decoder_depth "${GEO_DECODER_DEPTH:-2}"
      --geoformer_pool_lat "${GEO_POOL_LAT:-8}"
      --geoformer_pool_lon "${GEO_POOL_LON:-16}"
      --geoformer_patch_size "${GEO_PATCH_SIZE:-2}"
    )
    ;;
  *)
    echo "Unsupported MODEL_NAME=${MODEL_NAME}" >&2
    exit 2
    ;;
esac

LOADER_ARGS=(--num_workers "${NUM_WORKERS}")
if (( NUM_WORKERS > 0 )); then
  LOADER_ARGS+=(--prefetch_factor "${PREFETCH_FACTOR}")
  if [[ "${PERSISTENT_WORKERS}" == "1" ]]; then
    LOADER_ARGS+=(--persistent_workers)
  fi
fi
RUNTIME_ARGS=()
if [[ "${AMP}" == "1" ]]; then
  RUNTIME_ARGS+=(--amp)
fi
if [[ "${COMPILE}" == "1" ]]; then
  RUNTIME_ARGS+=(--compile)
fi

COMMON_ARGS=(
  --cmip_path "${CMIP_PATH:-${DATA_DIR}/cmip6_historical_1900_2014_global_mechanism.nc}"
  --obs_path "${OBS_PATH:-${DATA_DIR}/obs_1980_2025_global_mechanism.nc}"
  --obs_extra_train_path "${OBS_EXTRA_TRAIN_PATH:-${DATA_DIR}/obs_1958_1978_global_mechanism.nc}"
  --obs_extra_start_year 1958
  --obs_train_end_year "${OBS_TRAIN_END_YEAR:-1980}"
  --obs_test_start_year "${OBS_TEST_START_YEAR:-1980}"
  --obs_test_end_year "${OBS_TEST_END_YEAR:-0}"
  --obs_train_weight "${OBS_TRAIN_WEIGHT:-8}"
  --val_months_per_model "${VAL_MONTHS_PER_MODEL:-60}"
  --map_vars "${MAP_VARS}"
  --phys_vars "${PHYS_VARS}"
  --target_vars nino34,thermocline_tilt
  --spatial_mask_spec "${SPATIAL_MASK_SPEC}"
  --input_len 12
  --output_len 24
  --epochs "${EPOCHS}"
  --batch_size "${BATCH_SIZE}"
  --learning_rate "${LEARNING_RATE:-1.5e-4}"
  --warmup_epochs "${WARMUP_EPOCHS:-8}"
  --min_lr_ratio 0.03
  --weight_decay "${WEIGHT_DECAY:-4e-4}"
  --patience "${PATIENCE:-15}"
  --grad_clip 2.0
  --dropout 0.12
  --label_noise 0.0
  --mixup_alpha 0.0
  --seed "${SEED}"
  --save_dir "${SAVE_DIR}"
  --visual_dir "${VISUAL_DIR}"
  --exp_tag "${EXP_TAG}"
  --max_train_batches "${MAX_TRAIN_BATCHES:-0}"
  --max_eval_batches "${MAX_EVAL_BATCHES:-0}"

  --aux_weight "${AUX_WEIGHT:-0.15}"
  --aux_taper_start 10
  --aux_taper_end 18
  --ode_weight "${ODE_WEIGHT:-0.01}"
  --ode_lead_start 8
  --spb_weight "${SPB_WEIGHT:-0.15}"
  --spb_lead_start 2
  --spb_lead_end 18
  --spring_init_weight "${SPRING_INIT_WEIGHT:-0.10}"
  --spring_init_lead_start 12
  --spring_init_lead_end 18
  --spring_init_warmup_epochs "${SPRING_INIT_WARMUP_EPOCHS:-10}"
  --smoothness_weight "${SMOOTHNESS_WEIGHT:-0.003}"
  --trend_weight "${TREND_WEIGHT:-0.02}"
  --adaptive_weight "${ADAPTIVE_WEIGHT:-0.02}"
  --nll_weight "${NLL_WEIGHT:-0.04}"
  --nll_warmup_epochs 8
  --amplitude_weight "${AMPLITUDE_WEIGHT:-0.04}"
  --amplitude_threshold "${AMPLITUDE_THRESHOLD:-0.8}"
  --corr_weight "${CORR_WEIGHT:-0.03}"
  --corr_lead_start 6
  --overshoot_weight "${OVERSHOOT_WEIGHT:-0.04}"
  --batch_corr_weight "${BATCH_CORR_WEIGHT:-0.05}"
  --batch_corr_lead_start 6
  --batch_corr_lead_end 18
  --batch_corr_warmup_epochs "${BATCH_CORR_WARMUP_EPOCHS:-10}"
  --long_lead_weight "${LONG_LEAD_WEIGHT:-0.15}"
  --long_lead_start 12
  --long_lead_end 18
  --scale_weight "${SCALE_WEIGHT:-0.03}"
  --scale_lead_start 12
  --scale_lead_end 18
  --scale_warmup_epochs "${SCALE_WARMUP_EPOCHS:-10}"
  --scale_target_ratio "${SCALE_TARGET_RATIO:-0.82}"
  --scale_slope_target "${SCALE_SLOPE_TARGET:-0.78}"
  --phase_weight "${PHASE_WEIGHT:-0.01}"
  --phase_lead_start 12
  --phase_lead_end 18
  --phase_threshold "${PHASE_THRESHOLD:-0.5}"
  --phase_margin "${PHASE_MARGIN:-0.20}"
  --phase_warmup_epochs "${PHASE_WARMUP_EPOCHS:-10}"
  --phase_drift_weight "${PHASE_DRIFT_WEIGHT:-0.02}"
  --phase_drift_lead_start 9
  --phase_drift_lead_end 18
  --phase_drift_warmup_epochs "${PHASE_DRIFT_WARMUP_EPOCHS:-10}"
  --seasonal_mean_weight "${SEASONAL_MEAN_WEIGHT:-0.015}"
  --seasonal_mean_lead_start 12
  --seasonal_mean_lead_end 18
  --seasonal_mean_warmup_epochs "${SEASONAL_MEAN_WARMUP_EPOCHS:-10}"
  --selective_weight "${SELECTIVE_WEIGHT:-0.0}"
  --selective_lead_start 12
  --selective_lead_end 18
  --selective_min_confidence "${SELECTIVE_MIN_CONFIDENCE:-0.35}"
  --selective_warmup_epochs "${SELECTIVE_WARMUP_EPOCHS:-10}"
)

mkdir -p "${SAVE_DIR}" "${VISUAL_DIR}" "${METADATA_DIR}"
export ACTION MODEL_NAME MODEL_KEY CONFIG_ID CLAIM_FAMILY CLAIM_ROLE MAP_VARS PHYS_VARS
export SPATIAL_MASK_SPEC SEED EXP_TAG EPOCHS BATCH_SIZE NUM_WORKERS PREFETCH_FACTOR
export AMP COMPILE
METADATA_FILE="${METADATA_DIR}/${EXP_TAG}.json"
if [[ "${ACTION}" != analysis || ! -s "${METADATA_FILE}" ]]; then
  "${PYTHON}" - "${METADATA_FILE}" <<'PY'
import json
import os
import sys

def split(value):
    return [item for item in value.split(",") if item]

payload = {
    "experiment": os.environ["EXP_TAG"],
    "model_name": os.environ["MODEL_NAME"],
    "model_key": os.environ["MODEL_KEY"],
    "configuration": os.environ["CONFIG_ID"],
    "claim_family": os.environ["CLAIM_FAMILY"],
    "claim_role": os.environ["CLAIM_ROLE"],
    "map_variables": split(os.environ["MAP_VARS"]),
    "physical_indices": split(os.environ["PHYS_VARS"]),
    "spatial_mask_spec": os.environ["SPATIAL_MASK_SPEC"],
    "seed": int(os.environ["SEED"]),
    "epochs": int(os.environ["EPOCHS"]),
    "runtime_batch_size": int(os.environ["BATCH_SIZE"]),
    "training_batch_size": int(os.environ["BATCH_SIZE"]),
    "num_workers": int(os.environ["NUM_WORKERS"]),
    "prefetch_factor": int(os.environ["PREFETCH_FACTOR"]),
    "amp_bf16": os.environ["AMP"] == "1",
    "torch_compile": os.environ["COMPILE"] == "1",
    "selection_protocol": "OBS-best upper-bound, spb_composite",
    "event_candidates": [1982, 1987, 1997, 2009, 2015, 2020, 2023],
}
with open(sys.argv[1], "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, ensure_ascii=False)
PY
else
  echo "[SPB core] preserving training metadata during analysis: ${METADATA_FILE}"
fi

echo "[SPB core] action=${ACTION} model=${MODEL_NAME} config=${CONFIG_ID} seed=${SEED}"
echo "[SPB core] maps=${MAP_VARS} mask=${SPATIAL_MASK_SPEC} batch=${BATCH_SIZE} workers=${NUM_WORKERS} prefetch=${PREFETCH_FACTOR}"

if [[ "${ACTION}" == "train" || "${ACTION}" == "test" ]]; then
  "${PYTHON}" train_v2.py --stage "${ACTION}" \
    "${COMMON_ARGS[@]}" "${LOADER_ARGS[@]}" "${RUNTIME_ARGS[@]}" "${MODEL_ARGS[@]}"
fi

if [[ "${RUN_ANALYSIS}" != "1" ]]; then
  echo "[Skip] post-training SPB analysis disabled"
  exit 0
fi

RESULT_ROOT="${VISUAL_DIR}/${EXP_TAG}"
RESULT_DIR="$(find "${RESULT_ROOT}" -mindepth 1 -maxdepth 1 -type d -name "${EXP_TAG}_*" -exec test -f '{}/test_checkpoint.json' ';' -print 2>/dev/null | sort | tail -n 1)"
if [[ -z "${RESULT_DIR}" ]]; then
  echo "No completed test result found below ${RESULT_ROOT}" >&2
  exit 1
fi
if [[ "${ACTION}" != "analysis" || ! -s "${RESULT_DIR}/spb_summary.json" || "${FORCE_ANALYSIS:-0}" == "1" ]]; then
  "${PYTHON}" "${SCRIPT_DIR}/analyze_glgeo_spb.py" \
    --result_dir "${RESULT_DIR}" \
    --metadata "${METADATA_FILE}" \
    --input_len 12 \
    --obs_start_month 1 \
    --bootstrap_samples "${BOOTSTRAP_SAMPLES}" \
    --seed "${SEED}"
else
  echo "[Skip] analysis exists: ${RESULT_DIR}/spb_summary.json"
fi
