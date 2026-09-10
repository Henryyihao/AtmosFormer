set -euo pipefail


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

PYTHON="${PYTHON:-python}"
MANIFEST="${MANIFEST:-${PROJECT_ROOT}/configs/spb_core_claim_experiments.csv}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${PROJECT_ROOT}/checkpoints_spb_core_claim}"
TRAIN_VIS_ROOT="${TRAIN_VIS_ROOT:-${PROJECT_ROOT}/results_spb_core_claim_training}"
RESULT_ROOT="${RESULT_ROOT:-${PROJECT_ROOT}/results_spb_core_claim}"
LOG_DIR="${LOG_DIR:-${RESULT_ROOT}/logs}"
SUMMARY_DIR="${SUMMARY_DIR:-${RESULT_ROOT}/summary}"
DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/data}"

RUN_TRAINING="${RUN_TRAINING:-1}"
RUN_ANALYSIS="${RUN_ANALYSIS:-1}"
RUN_SUMMARY="${RUN_SUMMARY:-1}"
RUN_MODEL_SIZES="${RUN_MODEL_SIZES:-1}"
DRY_RUN="${DRY_RUN:-0}"
FORCE="${FORCE:-0}"
RUN_MODELS="${RUN_MODELS:-AtmosFormerSPB}"
RUN_CONFIGS="${RUN_CONFIGS:-}"
SEEDS_ATMOSFORMER="${SEEDS_ATMOSFORMER:-2025,2026,2027,2028,2029}"
SEEDS_MULTIARCH="${SEEDS_MULTIARCH:-2025,2026,2027,2028,2029}"

EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-128}"
ATMOS_BATCH_SIZE="${ATMOS_BATCH_SIZE:-${BATCH_SIZE}}"
CNN_BATCH_SIZE="${CNN_BATCH_SIZE:-32}"
GEO_BATCH_SIZE="${GEO_BATCH_SIZE:-32}"
CONVLSTM_BATCH_SIZE="${CONVLSTM_BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
PERSISTENT_WORKERS="${PERSISTENT_WORKERS:-1}"
AMP="${AMP:-1}"
COMPILE="${COMPILE:-0}"
BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-1000}"

mkdir -p "${CHECKPOINT_ROOT}" "${TRAIN_VIS_ROOT}" "${RESULT_ROOT}" "${LOG_DIR}" "${SUMMARY_DIR}"
[[ -f "${MANIFEST}" ]] || { echo "Missing manifest: ${MANIFEST}" >&2; exit 2; }

for value in "${RUN_TRAINING}" "${RUN_ANALYSIS}" "${RUN_SUMMARY}" "${RUN_MODEL_SIZES}" "${DRY_RUN}" "${FORCE}"; do
  [[ "${value}" == 0 || "${value}" == 1 ]] || { echo "Runner flags must be 0 or 1" >&2; exit 2; }
done

TASKS=()
while IFS= read -r task; do
  [[ -n "${task}" ]] && TASKS+=("${task}")
done < <("${PYTHON}" - "${MANIFEST}" "${RUN_MODELS}" "${RUN_CONFIGS}" \
  "${SEEDS_ATMOSFORMER}" "${SEEDS_MULTIARCH}" <<'PY'
import csv
import sys

manifest, models_text, configs_text, atmos_seeds_text, multi_seeds_text = sys.argv[1:]
aliases = {
    "AtmosFormerSPB": ("atmosformer", "AtmosFormerSPB"),
    "AtmosFormer": ("atmosformer", "AtmosFormerSPB"),
    "ENSOCNN": ("cnn", "ENSOCNN"),
    "CNN": ("cnn", "ENSOCNN"),
    "ENSOConvLSTM": ("convlstm", "ENSOConvLSTM"),
    "ConvLSTM": ("convlstm", "ENSOConvLSTM"),
    "ENSOGeoformer": ("geoformer", "ENSOGeoformer"),
    "Geoformer": ("geoformer", "ENSOGeoformer"),
}
models = []
for value in models_text.split(","):
    value = value.strip()
    if not value:
        continue
    if value not in aliases:
        raise SystemExit(f"Unsupported model: {value}")
    if aliases[value] not in models:
        models.append(aliases[value])
configs = {value.strip() for value in configs_text.split(",") if value.strip()}

def seeds(text):
    result = [value.strip() for value in text.split(",") if value.strip()]
    if not result or any(not value.isdigit() for value in result):
        raise SystemExit(f"Invalid seed list: {text}")
    return result

atmos_seeds = seeds(atmos_seeds_text)
multi_seeds = seeds(multi_seeds_text)
with open(manifest, newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))
for key, model_name in models:
    for row in rows:
        config = row["config_id"]
        if configs and config not in configs:
            continue
        enabled = row["atmosformer"] if key == "atmosformer" else row["multi_arch"]
        if enabled != "1":
            continue
        selected_seeds = atmos_seeds if key == "atmosformer" else multi_seeds
        for seed in selected_seeds:
            values = (
                key, model_name, config, row["claim_family"], row["role"],
                row["map_vars"], row["spatial_mask_spec"], seed,
            )
            print("\t".join(values))
PY
)

if (( ${#TASKS[@]} == 0 )); then
  echo "No tasks selected. Check RUN_MODELS and RUN_CONFIGS." >&2
  exit 2
fi
if (( ${#TASKS[@]} > 100 )); then
  echo "Planned training tasks=${#TASKS[@]}, above the 100-task limit." >&2
  exit 2
fi
echo "[SPB core] planned_training_tasks=${#TASKS[@]} limit=100 epochs=${EPOCHS}"
echo "[SPB core] batch default=${BATCH_SIZE} atmos=${ATMOS_BATCH_SIZE} cnn=${CNN_BATCH_SIZE} convlstm=${CONVLSTM_BATCH_SIZE} geo=${GEO_BATCH_SIZE}"
echo "[SPB core] workers=${NUM_WORKERS} prefetch=${PREFETCH_FACTOR} persistent=${PERSISTENT_WORKERS} amp=${AMP}"

checkpoint_path() {
  local tag="$1"
  if [[ -s "${CHECKPOINT_ROOT}/${tag}.pth" ]]; then
    printf '%s\n' "${CHECKPOINT_ROOT}/${tag}.pth"
  else
    return 1
  fi
}

find_task_file() {
  local key="$1" config="$2" tag="$3" filename="$4"
  find "${TRAIN_VIS_ROOT}/${key}/${config}/${tag}" -type f -name "${filename}" -size +0c -print -quit 2>/dev/null || true
}

valid_task_summary() {
  local path="$1" key="$2" config="$3" seed="$4"
  "${PYTHON}" - "${path}" "${key}" "${config}" "${seed}" <<'PY' >/dev/null 2>&1
import json
import sys

path, key, config, seed = sys.argv[1:]
with open(path, encoding="utf-8") as handle:
    payload = json.load(handle)
if str(payload.get("model_key")) != key:
    raise SystemExit(1)
if str(payload.get("configuration")) != config:
    raise SystemExit(1)
if int(payload.get("seed", -1)) != int(seed):
    raise SystemExit(1)
PY
}

run_logged() {
  local label="$1"
  shift
  if [[ "${DRY_RUN}" == 1 ]]; then
    printf '[DryRun] %s:' "${label}"
    printf ' %q' "$@"
    printf '\n'
    return 0
  fi
  echo "[Run] ${label}" | tee -a "${LOG_DIR}/${label}.log"
  "$@" 2>&1 | tee -a "${LOG_DIR}/${label}.log"
}

run_task() {
  local key="$1" model="$2" config="$3" family="$4" role="$5" maps="$6" mask="$7" seed="$8"
  local tag="SPBCore_${key}_${config}_seed${seed}"
  local checkpoint record summary action batch
  checkpoint="$(checkpoint_path "${tag}" 2>/dev/null || true)"
  record="$(find_task_file "${key}" "${config}" "${tag}" test_checkpoint.json)"
  summary="$(find_task_file "${key}" "${config}" "${tag}" spb_summary.json)"
  if [[ -n "${summary}" ]] && ! valid_task_summary "${summary}" "${key}" "${config}" "${seed}"; then
    echo "[Repair] invalid task summary will be regenerated: ${summary}"
    summary=""
  fi

  if [[ "${FORCE}" != 1 && -n "${summary}" ]]; then
    echo "[Skip] complete ${tag}"
    return 0
  fi
  if [[ "${FORCE}" == 1 ]]; then
    action=train
  elif [[ -n "${record}" ]]; then
    action=analysis
  elif [[ -n "${checkpoint}" ]]; then
    echo "[Restart] checkpoint lacks a final test record: ${tag}"
    action=train
  else
    action=train
  fi

  if [[ "${action}" == train && "${RUN_TRAINING}" != 1 ]]; then
    echo "[Skip] training disabled for incomplete task ${tag}"
    return 0
  fi
  if [[ "${action}" == analysis && "${RUN_ANALYSIS}" != 1 ]]; then
    echo "[Skip] analysis disabled for ${tag}"
    return 0
  fi

  batch="${BATCH_SIZE}"
  [[ "${key}" == atmosformer ]] && batch="${ATMOS_BATCH_SIZE}"
  [[ "${key}" == cnn ]] && batch="${CNN_BATCH_SIZE}"
  [[ "${key}" == convlstm ]] && batch="${CONVLSTM_BATCH_SIZE}"
  [[ "${key}" == geoformer ]] && batch="${GEO_BATCH_SIZE}"
  run_logged "${tag}" env \
    ACTION="${action}" MODEL_NAME="${model}" CONFIG_ID="${config}" \
    CLAIM_FAMILY="${family}" CLAIM_ROLE="${role}" MAP_VARS="${maps}" \
    PHYS_VARS="nino34,wwv,thermocline_tilt" SPATIAL_MASK_SPEC="${mask}" \
    SEED="${seed}" EXP_TAG="${tag}" EPOCHS="${EPOCHS}" BATCH_SIZE="${batch}" \
    NUM_WORKERS="${NUM_WORKERS}" PREFETCH_FACTOR="${PREFETCH_FACTOR}" \
    PERSISTENT_WORKERS="${PERSISTENT_WORKERS}" AMP="${AMP}" COMPILE="${COMPILE}" \
    RUN_ANALYSIS="${RUN_ANALYSIS}" \
    BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES}" SAVE_DIR="${CHECKPOINT_ROOT}" \
    VISUAL_DIR="${TRAIN_VIS_ROOT}/${key}/${config}" \
    METADATA_DIR="${RESULT_ROOT}/metadata" DATA_DIR="${DATA_DIR}" \
    bash "${SCRIPT_DIR}/train_spb_core_claim_one.sh"
}

if [[ "${RUN_MODEL_SIZES}" == 1 ]]; then
  run_logged report_core_model_sizes \
    "${PYTHON}" "${SCRIPT_DIR}/report_core_model_sizes.py" \
    --map_vars slp,sst,hc300,tauu \
    --output "${RESULT_ROOT}/model_sizes.csv"
fi

for task in "${TASKS[@]}"; do
  IFS=$'\t' read -r key model config family role maps mask seed <<< "${task}"
  run_task "${key}" "${model}" "${config}" "${family}" "${role}" "${maps}" "${mask}" "${seed}"
done

if [[ "${RUN_SUMMARY}" == 1 && "${DRY_RUN}" != 1 ]]; then
  run_logged summarize_core_claim \
    "${PYTHON}" "${SCRIPT_DIR}/summarize_spb_core_claim.py" \
    --training_results_dir "${TRAIN_VIS_ROOT}" --manifest "${MANIFEST}" \
    --output_dir "${SUMMARY_DIR}" --atmos_seeds "${SEEDS_ATMOSFORMER}" \
    --multi_seeds "${SEEDS_MULTIARCH}"
fi
