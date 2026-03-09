#!/usr/bin/env python3
"""
Process the packaged FSD50K dataset:
1. Read FSD50K metadata/ground-truth from local extracted files
2. Extract embeddings with ONNX model
3. Upload to Pinecone index (default: imitune-fsd50k)
"""

import argparse
import csv
import getpass
import json
import os
from pathlib import Path
from typing import Any, Optional

import librosa
import numpy as np
import soundfile as sf
from pinecone import Pinecone
from tqdm import tqdm

from process_freesound import (
    BATCH_SIZE,
    CLIP_DURATION_SECONDS,
    MODEL_PATH,
    SAMPLE_RATE,
    create_onnx_session,
    extract_embeddings_batch,
    load_excluded_tags,
    should_exclude,
)

DEFAULT_OUTPUT_FILENAME = "fsd50k_embeddings.json"
DEFAULT_INDEX_NAME = "imitune-fsd50k"
DEFAULT_FSD50K_ROOT = Path(
    os.getenv("FSD50K_ROOT", f"/gpfs/scratch/{os.getenv('USER', 'unknown')}/FSD50K")
)


def find_required_path(root: Path, relative: str, path_type: str = "dir") -> Path:
    """Find a path directly under root or recursively as fallback."""
    direct = root / relative
    if direct.exists():
        return direct

    matches = [p for p in root.rglob(Path(relative).name) if p.exists()]
    if matches:
        return matches[0]

    raise FileNotFoundError(f"Could not find {path_type}: {relative} under {root}")


def load_csv_metadata(csv_path: Path, split_name: str) -> dict[str, dict[str, Any]]:
    """Load FSD50K ground-truth CSV and return mapping by clip id."""
    items: dict[str, dict[str, Any]] = {}

    with open(csv_path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            clip_id = str(row.get("fname", "")).strip()
            if not clip_id:
                continue

            labels_raw = str(row.get("labels", "")).strip()
            labels = [label.strip() for label in labels_raw.split(",") if label.strip()]

            items[clip_id] = {
                "split": split_name,
                "labels": labels,
            }

            if "split" in row and row["split"]:
                items[clip_id]["train_val_split"] = row["split"].strip()

    return items


def load_clip_info(json_path: Path) -> dict[str, dict[str, Any]]:
    """Load optional FSD50K clip metadata JSON keyed by clip id."""
    if not json_path.exists():
        return {}

    with open(json_path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)

    normalized: dict[str, dict[str, Any]] = {}
    for key, value in raw.items():
        normalized[str(key)] = value if isinstance(value, dict) else {}
    return normalized


def build_audio_index(audio_dir: Path) -> dict[str, Path]:
    """Index wav files by filename stem (Freesound id)."""
    index: dict[str, Path] = {}
    for wav_file in audio_dir.rglob("*.wav"):
        index[wav_file.stem] = wav_file
    return index


def preprocess_audio(
    audio_path: Path,
    target_sr: int,
    clip_duration_seconds: float,
) -> Optional[np.ndarray]:
    """Load wav, mono mixdown, resample, and pad/trim to clip duration."""
    try:
        waveform, sr = sf.read(str(audio_path), dtype="float32")

        if len(waveform.shape) > 1:
            waveform = np.mean(waveform, axis=1)

        if sr != target_sr:
            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=target_sr)

        max_samples = int(clip_duration_seconds * target_sr)
        if len(waveform) > max_samples:
            waveform = waveform[:max_samples]
        elif len(waveform) < max_samples:
            waveform = np.pad(waveform, (0, max_samples - len(waveform)), mode="constant")

        return waveform.astype(np.float32)
    except Exception:
        return None


def clip_url(clip_id: str) -> str:
    """Construct Freesound short URL from id."""
    return f"https://freesound.org/s/{clip_id}/"


def resolve_output_json_path(
    fsd50k_root: Path,
    output_json: Optional[Path] = None,
) -> Path:
    """Resolve the embeddings JSON output path.

    By default, use a sibling directory named after the dataset root with a
    `_vectors` suffix, e.g. `/path/FSD50K` -> `/path/FSD50K_vectors/fsd50k_embeddings.json`.
    """
    if output_json is not None:
        return output_json

    root_path = fsd50k_root.expanduser()
    output_dir = root_path.parent / f"{root_path.name}_vectors"
    return output_dir / DEFAULT_OUTPUT_FILENAME


def process_fsd50k(
    excluded_labels: set[str],
    fsd50k_root: Path = DEFAULT_FSD50K_ROOT,
    output_json: Optional[Path] = None,
) -> list[dict[str, Any]]:
    """Extract embeddings from FSD50K and return records."""
    output_json_path = resolve_output_json_path(fsd50k_root, output_json)

    print("=" * 60)
    print("FSD50K Processor")
    print("=" * 60)

    print(f"\n1. Loading ONNX model from {MODEL_PATH}...")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    session = create_onnx_session(MODEL_PATH)

    print("\n2. Resolving FSD50K paths...")
    gt_dir = find_required_path(fsd50k_root, "FSD50K.ground_truth", "directory")
    metadata_dir = find_required_path(fsd50k_root, "FSD50K.metadata", "directory")
    print(f"   Ground truth: {gt_dir}")
    print(f"   Metadata:     {metadata_dir}")

    split_config = {
        "dev": {
            "csv": gt_dir / "dev.csv",
            "audio": find_required_path(fsd50k_root, "FSD50K.dev_audio", "directory"),
            "clips_info": metadata_dir / "dev_clips_info_FSD50K.json",
        },
        "eval": {
            "csv": gt_dir / "eval.csv",
            "audio": find_required_path(fsd50k_root, "FSD50K.eval_audio", "directory"),
            "clips_info": metadata_dir / "eval_clips_info_FSD50K.json",
        },
    }

    selected_splits = ["dev", "eval"]
    if not selected_splits:
        raise ValueError("No valid splits selected. Use any of: dev eval")

    print(f"\n3. Loading clip index for splits: {', '.join(selected_splits)}")

    all_items: list[dict[str, Any]] = []
    for split_name in selected_splits:
        cfg = split_config[split_name]
        if not cfg["csv"].exists():
            raise FileNotFoundError(f"Missing CSV for split '{split_name}': {cfg['csv']}")

        csv_map = load_csv_metadata(cfg["csv"], split_name)
        clip_info = load_clip_info(cfg["clips_info"])
        audio_index = build_audio_index(cfg["audio"])

        print(
            f"   {split_name}: {len(csv_map):,} labels | {len(audio_index):,} wav files"
        )

        for clip_id, meta in csv_map.items():
            audio_path = audio_index.get(clip_id)
            if audio_path is None:
                continue

            clip_meta = clip_info.get(clip_id, {})
            uploader = clip_meta.get("uploader") or clip_meta.get("username")
            tags = clip_meta.get("tags") if isinstance(clip_meta.get("tags"), list) else []

            all_items.append(
                {
                    "clip_id": clip_id,
                    "audio_path": audio_path,
                    "labels": meta.get("labels", []),
                    "split": split_name,
                    "train_val_split": meta.get("train_val_split"),
                    "uploader": uploader,
                    "tags": tags,
                }
            )

    print(f"\n4. Processing {len(all_items):,} clips...")
    print(f"   Sample rate: {SAMPLE_RATE} Hz")
    print(f"   Clip duration for embedding: {CLIP_DURATION_SECONDS}s")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   Output JSON: {output_json_path}")
    if excluded_labels:
        print(f"   Filtering enabled with {len(excluded_labels)} excluded tags")
    else:
        print("   Filtering disabled (no tags file provided)")

    output_json_path.parent.mkdir(parents=True, exist_ok=True)

    stats = {
        "total": 0,
        "filtered_labels": 0,
        "audio_failed": 0,
        "embedded": 0,
    }

    embeddings_data: list[dict[str, Any]] = []
    batch_waveforms: list[np.ndarray] = []
    batch_items: list[dict[str, Any]] = []

    progress = tqdm(all_items, desc="Embedding", unit=" clips", dynamic_ncols=True)

    for item in progress:
        stats["total"] += 1

        labels = item.get("labels", [])
        if should_exclude(labels, excluded_labels):
            stats["filtered_labels"] += 1
            continue

        waveform = preprocess_audio(item["audio_path"], SAMPLE_RATE, CLIP_DURATION_SECONDS)
        if waveform is None:
            stats["audio_failed"] += 1
            continue

        batch_waveforms.append(waveform)
        batch_items.append(item)

        if len(batch_waveforms) >= BATCH_SIZE:
            embeddings = extract_embeddings_batch(session, batch_waveforms)
            for emb, meta in zip(embeddings, batch_items):
                clip_id = str(meta["clip_id"])
                stats["embedded"] += 1
                embeddings_data.append(
                    {
                        "id": f"fsd50k-{clip_id}",
                        "embedding": emb.tolist(),
                        "freesound_url": clip_url(clip_id),
                        "freesound_id": int(clip_id),
                        "split": meta["split"],
                        "labels": meta.get("labels", []),
                        "uploader": meta.get("uploader") or "",
                    }
                )

            batch_waveforms = []
            batch_items = []

        progress.set_postfix_str(
            f"✓ {stats['embedded']:,} | filter {stats['filtered_labels']:,} | fail {stats['audio_failed']:,}"
        )

    if batch_waveforms:
        embeddings = extract_embeddings_batch(session, batch_waveforms)
        for emb, meta in zip(embeddings, batch_items):
            clip_id = str(meta["clip_id"])
            stats["embedded"] += 1
            embeddings_data.append(
                {
                    "id": f"fsd50k-{clip_id}",
                    "embedding": emb.tolist(),
                    "freesound_url": clip_url(clip_id),
                    "freesound_id": int(clip_id),
                    "split": meta["split"],
                    "labels": meta.get("labels", []),
                    "uploader": meta.get("uploader") or "",
                }
            )

    print(f"\n5. Saving {len(embeddings_data):,} embeddings to {output_json_path}...")
    with open(output_json_path, "w", encoding="utf-8") as handle:
        json.dump(embeddings_data, handle)

    print("\n" + "=" * 60)
    print("FSD50K Processing Statistics")
    print("=" * 60)
    print(f"  Total scanned:      {stats['total']:,}")
    print(f"  Filtered (labels):  {stats['filtered_labels']:,}")
    print(f"  Failed audio:       {stats['audio_failed']:,}")
    print(f"  Embedded:           {stats['embedded']:,}")
    print("=" * 60)

    return embeddings_data


def upload_to_pinecone(
    embeddings_data: Optional[list[dict[str, Any]]] = None,
    fsd50k_root: Path = DEFAULT_FSD50K_ROOT,
    output_json: Optional[Path] = None,
):
    """Upload prepared embeddings to Pinecone index."""
    print("\n6. Uploading to Pinecone...")

    output_json_path = resolve_output_json_path(fsd50k_root, output_json)

    if embeddings_data is None:
        if not output_json_path.exists():
            raise FileNotFoundError(f"Embeddings JSON not found: {output_json_path}")
        print(f"   Loading embeddings from {output_json_path}...")
        with open(output_json_path, "r", encoding="utf-8") as handle:
            embeddings_data = json.load(handle)

    print(f"   Total embeddings to upload: {len(embeddings_data):,}")

    api_key = os.getenv("PINECONE_API_KEY") or getpass.getpass("Pinecone API Key: ")
    if not api_key:
        raise ValueError("Pinecone API key is required")

    pc = Pinecone(api_key=api_key)
    index = pc.Index(DEFAULT_INDEX_NAME)
    print(f"   Connected to index: {DEFAULT_INDEX_NAME}")

    batch_size = 100
    print(f"   Uploading in batches of {batch_size}...")

    for i in tqdm(range(0, len(embeddings_data), batch_size), desc="Uploading"):
        batch = embeddings_data[i : i + batch_size]
        vectors = [
            {
                "id": item["id"],
                "values": item["embedding"],
                "metadata": {
                    "freesound_url": item.get("freesound_url", ""),
                },
            }
            for item in batch
        ]

        index.upsert(vectors=vectors)

    print("\n   Upload complete!")
    print(index.describe_index_stats())


def main():
    parser = argparse.ArgumentParser(
        description="Process FSD50K dataset and upload embeddings to Pinecone"
    )
    parser.add_argument(
        "--fsd50k-root",
        type=Path,
        default=DEFAULT_FSD50K_ROOT,
        help="Path containing FSD50K.dev_audio/, FSD50K.eval_audio/, FSD50K.ground_truth/, FSD50K.metadata/",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional output JSON path. Defaults to <fsd50k-root>_vectors/fsd50k_embeddings.json.",
    )
    parser.add_argument(
        "--tags-file",
        type=str,
        default=None,
        help="Path to text file with tags to exclude (one per line). If not provided, no tag filtering is applied.",
    )
    parser.add_argument(
        "--process-only",
        action="store_true",
        help="Only extract embeddings; do not upload",
    )
    parser.add_argument(
        "--upload-only",
        action="store_true",
        help="Skip extraction and upload from existing JSON",
    )

    args = parser.parse_args()

    if args.process_only and args.upload_only:
        raise ValueError("Use only one of --process-only or --upload-only")

    excluded_labels = load_excluded_tags(args.tags_file)

    if args.upload_only:
        upload_to_pinecone(
            fsd50k_root=args.fsd50k_root,
            output_json=args.output_json,
        )
        return

    embeddings_data = process_fsd50k(
        excluded_labels=excluded_labels,
        fsd50k_root=args.fsd50k_root,
        output_json=args.output_json,
    )

    if not args.process_only:
        upload_to_pinecone(
            embeddings_data,
            fsd50k_root=args.fsd50k_root,
            output_json=args.output_json,
        )


if __name__ == "__main__":
    main()
