set -euo pipefail


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

PYTHON="${PYTHON:-python}"
RUN_BACKTEST="${RUN_BACKTEST:-1}"
RUN_PLOTS="${RUN_PLOTS:-1}"
FORCE_BACKTEST="${FORCE_BACKTEST:-0}"
SPB_SEED="${SPB_SEED:-}"
ONLY_SPB="${ONLY_SPB:-}"
if [[ -z "${ONLY_SPB}" ]]; then
  if [[ -n "${SPB_SEED}" ]]; then
    ONLY_SPB=1
  else
    ONLY_SPB=0
  fi
fi

CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${PROJECT_ROOT}/checkpoints_spb_core_claim}"
TRAINING_RESULTS_DIR="${TRAINING_RESULTS_DIR:-${PROJECT_ROOT}/results_spb_core_claim_training}"
SUMMARY_DIR="${SUMMARY_DIR:-${PROJECT_ROOT}/results_spb_core_claim/summary}"
EVENTS_DIR="${EVENTS_DIR:-${PROJECT_ROOT}/results_spb_core_claim/event_backtest}"
FIGURE_DIR="${FIGURE_DIR:-${PROJECT_ROOT}/results_spb_core_claim/figures_nature}"
OBS_PATH="${OBS_PATH:-${PROJECT_ROOT}/data/obs_1980_2025_global_mechanism.nc}"
MANIFEST="${MANIFEST:-${PROJECT_ROOT}/configs/atmos_spb_evidence_manifest.csv}"

EVENT_CONFIG="${EVENT_CONFIG:-global_slp_sst_tauu}"
EVENT_SEEDS="${EVENT_SEEDS:-best}"
BEST_SEED_METRIC="${BEST_SEED_METRIC:-nino34_mean_acc_6_18}"
EVENT_MASKS="${EVENT_MASKS:-full,state_only,variable_pair_slp_sst,basin_tp,basin_pair_tp_atlantic,basin_triplet_tp_indian_atlantic}"
EVENT_YEARS="${EVENT_YEARS:-1982,1987,1997,2009,2015,2020,2023}"
EVENT_BATCH_SIZE="${EVENT_BATCH_SIZE:-128}"
DEVICE="${DEVICE:-auto}"

[[ "${RUN_BACKTEST}" == 0 || "${RUN_BACKTEST}" == 1 ]] || { echo "RUN_BACKTEST must be 0 or 1" >&2; exit 2; }
[[ "${RUN_PLOTS}" == 0 || "${RUN_PLOTS}" == 1 ]] || { echo "RUN_PLOTS must be 0 or 1" >&2; exit 2; }
[[ "${ONLY_SPB}" == 0 || "${ONLY_SPB}" == 1 ]] || { echo "ONLY_SPB must be 0 or 1" >&2; exit 2; }
[[ -z "${SPB_SEED}" || "${SPB_SEED}" =~ ^[0-9]+$ ]] || { echo "SPB_SEED must be an integer" >&2; exit 2; }

if [[ "${RUN_BACKTEST}" == 1 ]]; then
  command=(
    "${PYTHON}" "${SCRIPT_DIR}/backtest_spb_core_events.py"
    --checkpoint_root "${CHECKPOINT_ROOT}"
    --output_dir "${EVENTS_DIR}"
    --config "${EVENT_CONFIG}"
    --seeds "${EVENT_SEEDS}"
    --training_results_dir "${TRAINING_RESULTS_DIR}"
    --best_seed_metric "${BEST_SEED_METRIC}"
    --mask_names "${EVENT_MASKS}"
    --event_years "${EVENT_YEARS}"
    --obs_path "${OBS_PATH}"
    --manifest "${MANIFEST}"
    --batch_size "${EVENT_BATCH_SIZE}"
    --device "${DEVICE}"
  )
  [[ "${FORCE_BACKTEST}" == 1 ]] && command+=(--force)
  echo "[Run] SPB event backtest: config=${EVENT_CONFIG} seeds=${EVENT_SEEDS}"
  "${command[@]}"
fi

if [[ "${RUN_PLOTS}" == 1 ]]; then
  echo "[Run] title-free Nature figures"
  plot_command=(
    "${PYTHON}" "${SCRIPT_DIR}/plot_spb_core_claim_nature.py"
    --summary_dir "${SUMMARY_DIR}"
    --training_results_dir "${TRAINING_RESULTS_DIR}"
    --events_dir "${EVENTS_DIR}"
    --output_dir "${FIGURE_DIR}"
  )
  [[ -n "${SPB_SEED}" ]] && plot_command+=(--spb_seed "${SPB_SEED}")
  [[ "${ONLY_SPB}" == 1 ]] && plot_command+=(--only_spb)
  "${plot_command[@]}"
fi

echo "[Done] outputs: ${EVENTS_DIR} and ${FIGURE_DIR}"
