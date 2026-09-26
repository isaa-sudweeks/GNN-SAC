#!/bin/bash
#SBATCH --job-name=replay-production
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --gres=gpu:a100:1
#SBATCH --qos=standby

set -euo pipefail

: "${REPLAY_VALIDATION_ROOT:?Set the isolated validation root}"
: "${REPLAY_VALIDATION_CASE:?Set a unique case name}"
: "${REPLAY_VALIDATION_MODE:?Set performance or correctness}"

REPO=$REPLAY_VALIDATION_ROOT/repo
RESULTS=$REPLAY_VALIDATION_ROOT/results/$REPLAY_VALIDATION_CASE
mkdir -p "$RESULTS" "$REPLAY_VALIDATION_ROOT/tmp/$SLURM_JOB_ID" \
  "$REPLAY_VALIDATION_ROOT/cache/$SLURM_JOB_ID"
export TMPDIR=$REPLAY_VALIDATION_ROOT/tmp/$SLURM_JOB_ID
export TORCHINDUCTOR_CACHE_DIR=$REPLAY_VALIDATION_ROOT/cache/$SLURM_JOB_ID
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH=$REPO/sac:$REPO:${PYTHONPATH:-}
cd "$REPO"

git rev-parse HEAD > "$RESULTS/git_head.txt"
nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv,noheader \
  > "$RESULTS/gpu.csv"

if [[ "$REPLAY_VALIDATION_MODE" == performance ]]; then
  : "${REPLAY_VALIDATION_CHECKPOINT:?Set the checkpoint for performance mode}"
  : "${REPLAY_VALIDATION_BACKEND:?Set legacy or torchrl_tensor}"
  : "${REPLAY_VALIDATION_STORAGE:?Set cpu_pinned, cuda, or auto}"
  uv run --offline --no-sync python scripts/validate_production_replay.py \
    --mode performance --checkpoint "$REPLAY_VALIDATION_CHECKPOINT" \
    --backend "$REPLAY_VALIDATION_BACKEND" --storage "$REPLAY_VALIDATION_STORAGE" \
    --warmup 10 --sample-repeats 100 --blocks 3 --updates 100 \
    --output "$RESULTS/report.json" > "$RESULTS/run.log" 2> "$RESULTS/run.err"
elif [[ "$REPLAY_VALIDATION_MODE" == correctness ]]; then
  : "${REPLAY_VALIDATION_TENSOR:?Set the tensor checkpoint}"
  CORRECTNESS_ARGS=(
    --correctness-updates "${REPLAY_CORRECTNESS_UPDATES:-1000}"
  )
  if [[ "${REPLAY_SAME_BACKEND_CONTROL:-false}" == true ]]; then
    CORRECTNESS_ARGS+=(--same-backend-control)
  else
    : "${REPLAY_VALIDATION_LEGACY:?Set the legacy checkpoint}"
    CORRECTNESS_ARGS+=(--legacy-checkpoint "$REPLAY_VALIDATION_LEGACY")
  fi
  uv run --offline --no-sync python scripts/validate_production_replay.py \
    --mode correctness --tensor-checkpoint "$REPLAY_VALIDATION_TENSOR" \
    --sample-repeats 100 "${CORRECTNESS_ARGS[@]}" \
    --output "$RESULTS/report.json" \
    > "$RESULTS/run.log" 2> "$RESULTS/run.err"
else
  printf 'Unknown validation mode: %s\n' "$REPLAY_VALIDATION_MODE" >&2
  exit 2
fi

printf 'ok\n' > "$RESULTS/status"
