#!/usr/bin/env bash
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT" || { echo "cannot cd to repo root"; exit 1; }

export CUDA_VISIBLE_DEVICES=0

PY="${PY:-$REPO_ROOT/.venv/Scripts/python.exe}"
[ -x "$PY" ] || PY="$REPO_ROOT/.venv/bin/python"
MODEL="${MODEL:-Qwen/Qwen3-4B}"
MEM_THRESHOLD="${MEM_THRESHOLD:-3000}"
POLL_INTERVAL="${POLL_INTERVAL:-60}"
DRY_RUN="${DRY_RUN:-0}"

TS_DIR="experiments/experiment_2_qwen3_4b/results/10_tile_size_and_shape"
LOG_DIR="$TS_DIR/logs"
DS_DIR="$TS_DIR/downstream"
MANIFEST="$LOG_DIR/manifest.tsv"
LOCKDIR="$TS_DIR/.queue.lock"

mkdir -p "$LOG_DIR" "$DS_DIR" "$TS_DIR/T16" "$TS_DIR/T8" "$TS_DIR/T4" "$TS_DIR/T1" "$TS_DIR/probe"

if [ "$DRY_RUN" != "1" ]; then
  if ! mkdir "$LOCKDIR" 2>/dev/null; then
    echo "Another tile_size_sweep_queue appears to be running (lock: $LOCKDIR)."
    echo "If that is stale, remove it with:  rmdir '$LOCKDIR'"
    exit 1
  fi
  trap 'rmdir "$LOCKDIR" 2>/dev/null' EXIT INT TERM
fi

if [ ! -f "$MANIFEST" ]; then
  printf "timestamp\tlabel\texit_code\tlog\n" > "$MANIFEST"
fi

preflight_wait() {
  while true; do
    local used
    used=$(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -n1 | tr -d ' ')
    if [ -z "$used" ]; then
      echo "[preflight] nvidia-smi unavailable -- proceeding without GPU gate."
      return 0
    fi
    if [ "$used" -lt "$MEM_THRESHOLD" ] 2>/dev/null; then
      echo "[preflight] GPU0 free (${used} MiB used < ${MEM_THRESHOLD}); starting next job."
      return 0
    fi
    echo "[preflight] GPU0 busy (${used} MiB used >= ${MEM_THRESHOLD}); waiting ${POLL_INTERVAL}s..."
    sleep "$POLL_INTERVAL"
  done
}

FAILS=0
JOBS=0
run_job() {
  local label="$1"; shift
  JOBS=$((JOBS + 1))
  local log="$LOG_DIR/${label}.log"
  if [ "$DRY_RUN" = "1" ]; then
    printf "DRY %02d  %-34s  %s\n" "$JOBS" "$label" "$*"
    return 0
  fi
  preflight_wait
  echo "[$(date '+%F %T')] START $label"
  {
    echo "### $label"
    echo "### $(date '+%F %T')"
    echo "### CMD: $*"
    echo "############################################################"
  } > "$log"
  "$@" >> "$log" 2>&1
  local rc=$?
  printf "%s\t%s\t%d\t%s\n" "$(date '+%F %T')" "$label" "$rc" "$log" >> "$MANIFEST"
  if [ "$rc" -ne 0 ]; then
    FAILS=$((FAILS + 1))
    echo "[$(date '+%F %T')] END   $label  exit=$rc  *** FAILED (continuing) ***"
  else
    echo "[$(date '+%F %T')] END   $label  exit=0"
  fi
  return 0
}

pct() { "$PY" -c "print(int(round(float($1)*100)))"; }

wm() { "$PY" scripts/run_tile_pruning.py --model "$MODEL" --whole-model "$@"; }
ds() { "$PY" scripts/run_downstream_eval.py --model "$MODEL" "$@"; }
us() { "$PY" scripts/run_unstructured_pruning.py --model "$MODEL" "$@"; }

TASKS=(hellaswag piqa arc_easy)

run_job "probe" "$PY" experiments/experiment_2_qwen3_4b/scripts/probe_tile_purity.py --model "$MODEL" \
  --layers 0 6 12 18 24 30 35 \
  --tiles 1 2 4 8 16 32 --q 0.02 0.05 0.10 --scope per_matrix \
  --maps wanda magnitude --null shuffle within_col --nperm 40 \
  --out "$TS_DIR/probe"

run_job "ds_anchor32_sgr_p5" ds --method sparsegpt_recon --tile-size 32 --prune-ratio 0.05 \
  --policy uniform --calib-samples 64 --calib-seqlen 512 --tasks "${TASKS[@]}" \
  --output "$DS_DIR/ds_anchor32_sgr_p5.json"

for T in 16 8; do
  run_job "ppl_recon_T${T}_p5" wm --method sparsegpt_recon --tile-size "$T" --prune-ratio 0.05 \
    --calib-samples 128 --calib-seqlen 512 --eval-frac 0.2 --experiment-dir "$TS_DIR/T$T"
  for S in 1 2 3 4 5; do
    run_job "ppl_random_T${T}_p5_s${S}" wm --method random --tile-size "$T" --prune-ratio 0.05 \
      --seeds "$S" --eval-frac 0.2 --experiment-dir "$TS_DIR/T$T"
  done
  run_job "ppl_wanda_T${T}_p5" wm --method wanda --tile-size "$T" --prune-ratio 0.05 \
    --calib-samples 128 --eval-frac 0.2 --experiment-dir "$TS_DIR/T$T"
  run_job "ppl_wanda_recon_T${T}_p5" wm --method wanda_recon --tile-size "$T" --prune-ratio 0.05 \
    --calib-samples 128 --eval-frac 0.2 --experiment-dir "$TS_DIR/T$T"
  run_job "ppl_magnitude_T${T}_p5" wm --method magnitude --tile-size "$T" --prune-ratio 0.05 \
    --eval-frac 0.2 --experiment-dir "$TS_DIR/T$T"
  for S in 1 2 3; do
    run_job "ppl_random_recon_T${T}_p5_s${S}" wm --method random_recon --tile-size "$T" --prune-ratio 0.05 \
      --seeds "$S" --calib-samples 128 --eval-frac 0.2 --experiment-dir "$TS_DIR/T$T"
  done
done

for T in 16 8; do
  for R in 0.10 0.20; do
    P=$(pct "$R")
    run_job "ppl_recon_T${T}_p${P}" wm --method sparsegpt_recon --tile-size "$T" --prune-ratio "$R" \
      --calib-samples 128 --calib-seqlen 512 --eval-frac 0.2 --experiment-dir "$TS_DIR/T$T"
  done
done

run_job "ppl_recon_T4_p5" wm --method sparsegpt_recon --tile-size 4 --prune-ratio 0.05 \
  --calib-samples 128 --calib-seqlen 512 --eval-frac 0.2 --experiment-dir "$TS_DIR/T4"
run_job "ppl_recon_T4_p10" wm --method sparsegpt_recon --tile-size 4 --prune-ratio 0.10 \
  --calib-samples 128 --calib-seqlen 512 --eval-frac 0.2 --experiment-dir "$TS_DIR/T4"
for S in 1 2 3 4 5; do
  run_job "ppl_random_T4_p5_s${S}" wm --method random --tile-size 4 --prune-ratio 0.05 \
    --seeds "$S" --eval-frac 0.2 --experiment-dir "$TS_DIR/T4"
done
run_job "ppl_magnitude_T4_p5" wm --method magnitude --tile-size 4 --prune-ratio 0.05 \
  --eval-frac 0.2 --experiment-dir "$TS_DIR/T4"
run_job "ppl_wanda_recon_T4_p5" wm --method wanda_recon --tile-size 4 --prune-ratio 0.05 \
  --calib-samples 128 --eval-frac 0.2 --experiment-dir "$TS_DIR/T4"
run_job "ppl_wanda_T4_p5" wm --method wanda --tile-size 4 --prune-ratio 0.05 \
  --calib-samples 128 --eval-frac 0.2 --experiment-dir "$TS_DIR/T4"

for T in 8 16 4; do
  for R in 0.05 0.10 0.20 0.01 0.02; do
    P=$(pct "$R")
    run_job "ds_recon_T${T}_p${P}" ds --method sparsegpt_recon --tile-size "$T" --prune-ratio "$R" \
      --policy uniform --calib-samples 64 --calib-seqlen 512 --tasks "${TASKS[@]}" \
      --output "$DS_DIR/ds_sgr_p${P}_uniform_T${T}.json"
  done
done

for T in 8 16 4; do
  run_job "ds_wanda_recon_T${T}_p5" ds --method wanda_recon --tile-size "$T" --prune-ratio 0.05 \
    --calib-samples 64 --tasks "${TASKS[@]}" \
    --output "$DS_DIR/ds_wanda_recon_p5_uniform_T${T}.json"
  run_job "ds_wanda_T${T}_p5" ds --method wanda --tile-size "$T" --prune-ratio 0.05 \
    --calib-samples 64 --tasks "${TASKS[@]}" \
    --output "$DS_DIR/ds_wanda_p5_uniform_T${T}.json"
  run_job "ds_random_T${T}_p5_s1" ds --method random --tile-size "$T" --prune-ratio 0.05 \
    --seed 1 --tasks "${TASKS[@]}" \
    --output "$DS_DIR/ds_random_p5_uniform_T${T}_s1.json"
  run_job "ds_random_recon_T${T}_p5_s1" ds --method random_recon --tile-size "$T" --prune-ratio 0.05 \
    --seed 1 --calib-samples 64 --tasks "${TASKS[@]}" \
    --output "$DS_DIR/ds_random_recon_p5_uniform_T${T}_s1.json"
done

for R in 0.05 0.10 0.20 0.30 0.40 0.50; do
  P=$(pct "$R")
  run_job "us_magnitude_maskonly_p${P}_T1" us --method magnitude --prune-ratio "$R" --repair none \
    --tasks "${TASKS[@]}" --output "$DS_DIR/us_magnitude_maskonly_p${P}_T1.json"
  run_job "us_wanda_maskonly_p${P}_T1" us --method wanda --prune-ratio "$R" --repair none \
    --calib-samples 64 --tasks "${TASKS[@]}" --output "$DS_DIR/us_wanda_maskonly_p${P}_T1.json"
  run_job "us_sparsegpt_p${P}_T1" us --method unstructured_sparsegpt --prune-ratio "$R" \
    --calib-samples 64 --tasks "${TASKS[@]}" --output "$DS_DIR/us_sparsegpt_p${P}_T1.json"
done

echo
echo "############################################################"
echo "  tile_size_sweep_queue complete: $JOBS jobs, $FAILS failed."
echo "  manifest: $MANIFEST"
echo "############################################################"
[ "$FAILS" -eq 0 ]
