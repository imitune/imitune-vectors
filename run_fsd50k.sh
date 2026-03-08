#!/bin/bash
#SBATCH -J imitune_fsd50k
#SBATCH -p gpu
#SBATCH -A pilot_gpu
#SBATCH -n 8
#SBATCH -t 72:0:0
#SBATCH --mem-per-cpu=16G
#SBATCH --gres=gpu:1
#SBATCH -o %x.o%j
#SBATCH -e %x.e%j

set -euo pipefail

module load miniforge
conda activate imitune
module load cuda/12.6.2-gcc-12.2.0
module load cudnn/9.2.0.82-12-cuda-12.6.2-gcc-12.2.0
module load ffmpeg

cd /data/home/acw777/imitune-vectors

# Configure these per run/environment:
FSD50K_ROOT="${FSD50K_ROOT:-/data/home/acw777/datasets/FSD50K}"
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
