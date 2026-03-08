#!/bin/bash
#SBATCH -J imitune_fsd50k
#SBATCH -p gpushort
#SBATCH -n 8
#SBATCH -t 1:0:0
#SBATCH --mem-per-cpu=11G
#SBATCH --gres=gpu:1
#SBATCH -o %x.o%j
#SBATCH -e %x.e%j

# gpushort is for short runs.
# For long runs on Apocrita restricted GPU nodes, submit with:
# sbatch -p gpu -A pilot_gpu -t 72:0:0 run_fsd50k.sh

set -euo pipefail

module load miniforge
conda activate imitune
module load cuda/12.6.2-gcc-12.2.0
module load cudnn/9.2.0.82-12-cuda-12.6.2-gcc-12.2.0
module load ffmpeg

cd /data/home/acw777/imitune-vectors

# Configure these per run/environment:
FSD50K_ROOT="${FSD50K_ROOT:-/gpfs/scratch/${USER}/FSD50K}"
MODE="${MODE:-process-only}"  # process-only | upload-only | full
TAGS_FILE="${TAGS_FILE:-}"

CMD=(
  python process_fsd50k.py
  --fsd50k-root "$FSD50K_ROOT"
)

if [[ -n "$TAGS_FILE" ]]; then
  CMD+=(--tags-file "$TAGS_FILE")
fi

case "$MODE" in
  process-only)
    CMD+=(--process-only)
    ;;
  upload-only)
    CMD+=(--upload-only)
    ;;
  full)
    ;;
  *)
    echo "Invalid MODE: $MODE (use process-only, upload-only, or full)"
    exit 1
    ;;
esac

echo "Running: ${CMD[*]}"
"${CMD[@]}"
