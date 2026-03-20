#!/bin/bash
#SBATCH -J fsd50k_download
#SBATCH -p compute
#SBATCH -n 1
#SBATCH -t 08:00:00
#SBATCH --mem-per-cpu=2G
#SBATCH -o %x.o%j
#SBATCH -e %x.e%j

set -euo pipefail

RECORD_ID="4060432"
BASE_URL="https://zenodo.org/records/${RECORD_ID}/files"

# Destination root on scratch (override when submitting if needed)
SCRATCH_ROOT="${SCRATCH_ROOT:-/gpfs/scratch/${USER}}"
TARGET_DIR="${TARGET_DIR:-${SCRATCH_ROOT}/FSD50K}"
DOWNLOAD_DIR="${DOWNLOAD_DIR:-${TARGET_DIR}/downloads}"

mkdir -p "${DOWNLOAD_DIR}"
mkdir -p "${TARGET_DIR}"

echo "Downloading FSD50K to: ${DOWNLOAD_DIR}"

FILES=(
  "FSD50K.dev_audio.z01"
  "FSD50K.dev_audio.z02"
  "FSD50K.dev_audio.z03"
  "FSD50K.dev_audio.z04"
  "FSD50K.dev_audio.z05"
  "FSD50K.dev_audio.zip"
  "FSD50K.eval_audio.z01"
  "FSD50K.eval_audio.zip"
  "FSD50K.ground_truth.zip"
  "FSD50K.metadata.zip"
  "FSD50K.doc.zip"
)

download_file() {
  local file="$1"
  local url="${BASE_URL}/${file}?download=1"
  local out="${DOWNLOAD_DIR}/${file}"

  if [[ -s "${out}" ]]; then
    echo "Already present, skipping: ${file}"
    return
  fi

  echo "Downloading: ${file}"
  curl -fL --retry 5 --retry-delay 10 -o "${out}" "${url}"
}

for file in "${FILES[@]}"; do
  download_file "${file}"
done

echo "Unpacking metadata/documentation archives..."
unzip -o "${DOWNLOAD_DIR}/FSD50K.ground_truth.zip" -d "${TARGET_DIR}"
unzip -o "${DOWNLOAD_DIR}/FSD50K.metadata.zip" -d "${TARGET_DIR}"
unzip -o "${DOWNLOAD_DIR}/FSD50K.doc.zip" -d "${TARGET_DIR}"

echo "Reassembling and unpacking split dev audio archive..."
pushd "${DOWNLOAD_DIR}" >/dev/null
zip -s 0 FSD50K.dev_audio.zip --out FSD50K.dev_audio.unsplit.zip
unzip -o FSD50K.dev_audio.unsplit.zip -d "${TARGET_DIR}"

echo "Reassembling and unpacking split eval audio archive..."
zip -s 0 FSD50K.eval_audio.zip --out FSD50K.eval_audio.unsplit.zip
unzip -o FSD50K.eval_audio.unsplit.zip -d "${TARGET_DIR}"
popd >/dev/null

echo
echo "FSD50K download complete."
echo "Dataset root: ${TARGET_DIR}"
echo "Use with processor: python process_fsd50k.py --fsd50k-root ${TARGET_DIR}"
