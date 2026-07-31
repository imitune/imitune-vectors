# FSD50K workflow for ThatSoundLikeMe vectors

This adds a separate pipeline for the packaged FSD50K release and uploads to a dedicated Pinecone index (`imitune-fsd50k`).

## 0) Download FSD50K to scratch (Slurm CPU job)

Submit the downloader job:

```bash
sbatch scripts/download_fsd50k.sh
```

Optional overrides at submit time:

```bash
sbatch --export=ALL,SCRATCH_ROOT=/gpfs/scratch/$USER,TARGET_DIR=/gpfs/scratch/$USER/FSD50K scripts/download_fsd50k.sh
```

This produces an extracted dataset root at `TARGET_DIR` containing:

- `FSD50K.dev_audio/`
- `FSD50K.eval_audio/`
- `FSD50K.ground_truth/`
- `FSD50K.metadata/`

## 1) Prepare local FSD50K files

After extracting the Zenodo archives, your root folder should contain:

- `FSD50K.dev_audio/`
- `FSD50K.eval_audio/`
- `FSD50K.ground_truth/`
- `FSD50K.metadata/`

## 2) Extract embeddings only

```bash
python process_fsd50k.py \
  --fsd50k-root /path/to/FSD50K \
  --process-only
```

## 3) Upload existing embeddings only

```bash
python process_fsd50k.py --upload-only
```

## 4) Full run (extract + upload)

```bash
python process_fsd50k.py --fsd50k-root /path/to/FSD50K
```

## Optional filters / smoke tests

- Exclude labels by term list (one term per line):

```bash
python process_fsd50k.py \
  --fsd50k-root /path/to/FSD50K \
  --tags-file config/tags_v1.txt \
  --process-only
```

## HPC job

Use `scripts/run_fsd50k.sh` and set environment variables before `sbatch` if needed:

```bash
export FSD50K_ROOT=/gpfs/scratch/$USER/FSD50K
export MODE=full
sbatch scripts/run_fsd50k.sh
```

`scripts/run_fsd50k.sh` defaults to `gpushort` with `1:0:0` walltime (short queue).

For long GPU runs on Apocrita, use the restricted pair from QMUL docs:

```bash
sbatch -p gpu -A pilot_gpu -t 72:0:0 scripts/run_fsd50k.sh
```

The script uses QMUL's standard single-GPU template: `-p gpushort`, `-n 8`, `--gres=gpu:1`, `--mem-per-cpu=11G`.

If your group has access to the restricted GPU partition/account pair, submit with:

```bash
sbatch -p gpu -A pilot_gpu scripts/run_fsd50k.sh
```

If you get `Invalid account or account/partition combination specified`, your account is not valid for that partition. Use the default script submission (`gpushort`) or your group's allowed `-p/-A` combination.

Defaults:

- FSD50K root: `/gpfs/scratch/$USER/FSD50K`
- Output JSON: `fsd50k_embeddings.json`
- Pinecone index: `imitune-fsd50k`
