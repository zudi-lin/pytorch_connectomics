#!/bin/bash
# One XML setup per array task. Pass RUN_DIR, CODE_ROOT, and optionally verify.
# Submit with --array=0-N%8 and explicit --output/--error paths inside RUN_DIR.
# This host's CPU jobs also use its sole partition, gpu; no GPU is requested.
#SBATCH --job-name=janelia_ingest
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00

set -euo pipefail
RUN_DIR="${1:?pass the directory containing tasks.json}"
CODE_ROOT="${2:-${SLURM_SUBMIT_DIR:?}}"
MODE="${3:-run}"
CONDA_EXE="${CONDA_EXE:-/data/donglai/miniforge3/bin/conda}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export ZARR_THREADING__MAX_WORKERS="${SLURM_CPUS_PER_TASK:-4}"
export ZARR_ASYNC__CONCURRENCY="${SLURM_CPUS_PER_TASK:-4}"

echo "host=$(hostname) job=${SLURM_JOB_ID:-local} mode=$MODE start=$(date -Is)"
ARGS=("$MODE" --manifest "$RUN_DIR/tasks.json")
if [[ "$MODE" == run ]]; then
    ARGS+=(--index "${SLURM_ARRAY_TASK_ID:?submit as a job array}")
elif [[ "$MODE" != verify ]]; then
    echo "unsupported mode: $MODE" >&2
    exit 2
fi

exec "$CONDA_EXE" run --no-capture-output -n pytc python \
    "$CODE_ROOT/tutorials/liconn_ingest/janelia_batch.py" "${ARGS[@]}"
