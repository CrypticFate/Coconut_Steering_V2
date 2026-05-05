#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/coconut_dryrun.yaml}"
MODEL_ID_OVERRIDE="${2:-}"

LOG_DIR="outputs/logs"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
MASTER_LOG="${LOG_DIR}/run_all_${RUN_TS}.log"

mkdir -p "${LOG_DIR}" outputs/checkpoints outputs/vectors outputs/alpha outputs/results

run_step() {
  local step_name="$1"
  shift
  local step_log="${LOG_DIR}/${step_name}_${RUN_TS}.log"
  echo
  echo "========== ${step_name} =========="
  echo "Command: $*"
  echo "Step log: ${step_log}"
  "$@" 2>&1 | tee "${step_log}" | tee -a "${MASTER_LOG}"
}

echo "Master log: ${MASTER_LOG}" | tee "${MASTER_LOG}"
echo "Config: ${CONFIG_PATH}" | tee -a "${MASTER_LOG}"
if [[ -n "${MODEL_ID_OVERRIDE}" ]]; then
  echo "Model override: ${MODEL_ID_OVERRIDE}" | tee -a "${MASTER_LOG}"
fi

run_step "00_setup" bash scripts/00_setup.sh
run_step "01_prepare_splits" python scripts/01_prepare_splits.py
chmod 444 data/splits/test.json
echo "Locked test split: data/splits/test.json" | tee -a "${MASTER_LOG}"

if [[ -n "${MODEL_ID_OVERRIDE}" ]]; then
  run_step "02_train_stagewise" python scripts/02_train_stagewise.py "${CONFIG_PATH}" --model_id "${MODEL_ID_OVERRIDE}"
else
  run_step "02_train_stagewise" python scripts/02_train_stagewise.py "${CONFIG_PATH}"
fi

run_step "03_extract_vectors" python scripts/03_extract_vectors.py "${CONFIG_PATH}"
run_step "04_check_gradient" python scripts/04_check_gradient.py "${CONFIG_PATH}"
run_step "05_tune_alpha" python scripts/05_tune_alpha.py "${CONFIG_PATH}"
run_step "06_evaluate" python scripts/06_evaluate.py "${CONFIG_PATH}"
run_step "phase4_summary" python -m json.tool outputs/results/phase4_summary.json

echo
echo "All phases completed successfully."
echo "Master log: ${MASTER_LOG}"
