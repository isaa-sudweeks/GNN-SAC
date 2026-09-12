#!/bin/bash
#SBATCH --job-name=torchrl-replay
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=5
#SBATCH --mem=128G
#SBATCH --gres=gpu:a100:1
#SBATCH --qos=standby

set -uo pipefail

: "${REPLAY_BENCH_ROOT:?Set REPLAY_BENCH_ROOT to the timestamped benchmark directory}"
: "${REPLAY_REPO:?Set REPLAY_REPO to its dedicated worktree}"

RESULTS=${REPLAY_BENCH_ROOT}/results
TMP_ROOT=${REPLAY_BENCH_ROOT}/tmp/${SLURM_JOB_ID}
CACHE_ROOT=${REPLAY_BENCH_ROOT}/cache/${SLURM_JOB_ID}
mkdir -p "$RESULTS" "$TMP_ROOT" "$CACHE_ROOT"
export TMPDIR=$TMP_ROOT
export TORCHINDUCTOR_CACHE_DIR=$CACHE_ROOT
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH=${REPLAY_REPO}/sac:${REPLAY_REPO}:${PYTHONPATH:-}
cd "$REPLAY_REPO" || exit 2

git rev-parse HEAD > "$RESULTS/git_head.txt"
nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv,noheader \
  > "$RESULTS/gpu.csv"
nvidia-smi --query-gpu=timestamp,memory.used,memory.free --format=csv,noheader,nounits \
  --loop=1 > "$RESULTS/gpu_memory.csv" &
MONITOR_PID=$!
trap 'kill "$MONITOR_PID" 2>/dev/null || true' EXIT

if uv run --offline --no-sync python -m unittest tests.test_tensor_gnn_buffer -v \
    > "$RESULTS/cuda_equivalence.log" 2>&1; then
  printf 'ok\n' > "$RESULTS/cuda_equivalence.status"
else
  printf 'failed:%s\n' "$?" > "$RESULTS/cuda_equivalence.status"
fi

run_replay_case() {
  local name=$1
  shift
  local output=$RESULTS/replay_${name}.json
  local log=$RESULTS/replay_${name}.log
  if uv run --offline --no-sync python scripts/benchmark_gnn_replay.py \
      --device cuda --prototype --node-counts 4,6,8 --feature-dim 6 \
      --batch-size 255 --entries-per-task 100000 --warmup 10 --repeats 100 \
      --checkpoint-repeats 3 --optimizer-repeats 20 --output "$output" \
      "$@" > "$log" 2>&1; then
    printf 'ok\n' > "$RESULTS/replay_${name}.status"
  else
    printf 'failed:%s\n' "$?" > "$RESULTS/replay_${name}.status"
  fi
}

run_training_case() {
  local name=$1
  local backend=$2
  local storage=$3
  local case_dir=$RESULTS/training_${name}
  mkdir -p "$case_dir"
  if uv run --offline --no-sync python sac/gnn_train.py \
      platform=supercomputer sim_backend=mjx sac_backend=gnn mjx_impl=warp \
      warp_graph_mode=warp_staged \
      'truss_topologies=[tetrahedron,octahedron,henneberg_n6_1tube_2]' \
      num_envs=384 steps=9600 seed=11 seed_steps=0 pretrain_steps=0 \
      batch_size=255 replay_ratio=10.0 update_every_vector_steps=1 pcgrad=false \
      replay_backend="$backend" replay_storage="$storage" \
      eval_freq=1000000000 eval_at_end=false progress_freq=1000000000 \
      checkpoint_freq=0 save_agent=false save_csv=false save_video=false \
      enable_wandb=false resume_from_checkpoint=null \
      profiling.enabled=true profiling.warmup_vector_steps=5 \
      profiling.active_vector_steps=20 profiling.trace_enabled=false \
      profiling.output_dir="$case_dir/profiling" work_dir="$case_dir" \
      hydra.run.dir="$case_dir/hydra" > "$case_dir/training.log" 2>&1; then
    printf 'ok\n' > "$case_dir/status"
  else
    printf 'failed:%s\n' "$?" > "$case_dir/status"
  fi
}

run_replay_case cpu_pinned --storage cpu_pinned
run_replay_case cuda --storage cuda
run_replay_case auto --storage auto
run_replay_case auto_mixed --storage auto --replay-gpu-fraction 1 \
  --replay-gpu-max-gb 0.06 --replay-gpu-reserve-gb 0

run_training_case legacy legacy auto
run_training_case auto torchrl_tensor auto

kill "$MONITOR_PID" 2>/dev/null || true
wait "$MONITOR_PID" 2>/dev/null || true
trap - EXIT
printf 'complete\n' > "$RESULTS/job.status"
