#!/bin/bash
#SBATCH -J imitune_laion
#SBATCH -p gpushort
#SBATCH -n 8
#SBATCH -t 1:0:0
#SBATCH --mem-per-cpu=11G
#SBATCH --gres=gpu:1
#SBATCH -o %x.o%j
#SBATCH -e %x.e%j

# gpushort is for short runs.
# For long runs on Apocrita restricted GPU nodes, submit with:
# sbatch -p gpu -A pilot_gpu -t 72:0:0 scripts/run.sh

set -euo pipefail

module load miniforge
conda activate imitune
module load cuda/12.6.2-gcc-12.2.0
module load cudnn/9.2.0.82-12-cuda-12.6.2-gcc-12.2.0
module load ffmpeg

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"
python process_freesound.py --blocked-labels-file config/blocked_model_labels_v1.txt
