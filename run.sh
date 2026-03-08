#!/bin/bash
#SBATCH -J imitune_laion
#SBATCH -p gpu
#SBATCH -A pilot_gpu
#SBATCH -n 8
#SBATCH -t 240:0:0
#SBATCH --mem-per-cpu=11G
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
python process_freesound.py --process-only --tags-file tags_v1.txt
