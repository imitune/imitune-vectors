#!/usr/bin/env python3
"""
Process the FreeSound-LAION-640k dataset:
1. Download from HuggingFace
2. Filter clips (<20s, no speech/singing/music)
3. Extract embeddings with ONNX model (GPU)
4. Upload to Pinecone
"""

import os

# Disable torchcodec to avoid compatibility issues - use soundfile instead
os.environ["HF_AUDIO_DECODER"] = "soundfile"

import getpass
import json
from collections import Counter
from pathlib import Path
from typing import Any, Optional

import datasets
import librosa
import numpy as np
import onnxruntime as ort
import soundfile as sf
from datasets import load_dataset
from pinecone import Pinecone
from tqdm import tqdm

from audio_tag_filter import (
    DEFAULT_BLOCKED_LABELS_PATH,
    DEFAULT_TAGGER_BATCH_SIZE,
    DEFAULT_TAGGER_MODEL_ID,
    DEFAULT_TAGGER_THRESHOLD,
    MAX_AUDIT_EXAMPLES,
    AudioTagFilter,
    load_blocked_labels,
)

# --- Configuration ---
MODEL_PATH = Path("model_v1.onnx")
OUTPUT_JSON = Path("freesound_embeddings.json")
FILTER_AUDIT_JSON = Path("freesound_filter_audit.json")
INDEX_NAME = "imitune-search"
SAMPLE_RATE = 16000  # Standard for audio embeddings
MAX_DURATION_SECONDS = 20
CLIP_DURATION_SECONDS = 10  # Extract embeddings from first 10 seconds
BATCH_SIZE = 64  # For GPU inference


def load_excluded_tags(tags_file: Optional[str] = None) -> set[str]:
    """Load excluded tags from a file, or disable filtering when none is provided."""
    if tags_file:
        try:
            with open(tags_file, "r", encoding="utf-8") as f:
                tags = {line.strip().lower() for line in f if line.strip()}
            print(f"   Loaded {len(tags)} excluded tags from {tags_file}")
            return tags
        except FileNotFoundError:
            print(f"   Warning: Tags file {tags_file} not found, filtering disabled")
        except Exception as e:
            print(f"   Warning: Error loading tags file {tags_file}: {e}, filtering disabled")

    return set()


def should_exclude(tags: list[str], excluded_tags: set[str]) -> bool:
    """Check if any tag matches our exclusion list."""
    if not tags:
        return False

    tags_lower = {t.lower().strip() for t in tags}

    for tag in tags_lower:
        # Direct match
        if tag in excluded_tags:
            return True
        # Partial match - if any excluded term is contained in the tag
        for excluded in excluded_tags:
            if excluded in tag or tag in excluded:
                return True

    return False


def get_audio_duration(audio_bytes: bytes) -> Optional[float]:
    """Get duration of audio from bytes."""
    try:
        import io

        with io.BytesIO(audio_bytes) as f:
            info = sf.info(f)
            return info.duration
    except Exception:
        return None


def load_and_preprocess_audio(
    audio_bytes: bytes,
    target_sr: int = SAMPLE_RATE,
    max_duration: float = CLIP_DURATION_SECONDS,
) -> Optional[np.ndarray]:
    """Load audio bytes, resample, and trim to max_duration."""
    try:
        import io

        with io.BytesIO(audio_bytes) as f:
            waveform, sr = sf.read(f, dtype="float32")

        # Convert stereo to mono if needed
        if len(waveform.shape) > 1:
            waveform = np.mean(waveform, axis=1)

        # Resample if needed
        if sr != target_sr:
            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=target_sr)

        # Trim to max_duration (first N seconds)
        max_samples = int(max_duration * target_sr)
        if len(waveform) > max_samples:
            waveform = waveform[:max_samples]

        # Pad if shorter than max_duration
        if len(waveform) < max_samples:
            waveform = np.pad(
                waveform, (0, max_samples - len(waveform)), mode="constant"
            )

        return waveform.astype(np.float32)

    except Exception as e:
        print(f"Error loading audio: {e}")
        return None


def create_onnx_session(model_path: Path) -> ort.InferenceSession:
    """Create ONNX inference session with GPU support."""
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    # Set thread count explicitly to avoid pthread_setaffinity_np warnings on HPC clusters
    sess_options.intra_op_num_threads = 1
    sess_options.inter_op_num_threads = 1
    # Suppress shape mismatch warnings (we handle reshaping ourselves)
    sess_options.log_severity_level = 3  # 3 = Error only, suppresses warnings

    session = ort.InferenceSession(
        str(model_path), sess_options=sess_options, providers=providers
    )

    # Check which provider is being used
    active_provider = session.get_providers()[0]
    print(f"ONNX Runtime using: {active_provider}")

    # Print model input/output info for debugging
    input_info = session.get_inputs()[0]
    output_info = session.get_outputs()[0]
    print(f"   Model input:  {input_info.name} {input_info.shape}")
    print(f"   Model output: {output_info.name} {output_info.shape}")

    return session


def extract_embeddings_batch(
    session: ort.InferenceSession, waveforms: list[np.ndarray]
) -> np.ndarray:
    """Extract embeddings for a batch of waveforms."""
    # Stack waveforms into batch: (batch_size, samples)
    batch = np.stack(waveforms, axis=0).astype(np.float32)

    # Check model's expected input shape
    input_info = session.get_inputs()[0]
    expected_dims = len(input_info.shape)

    # Add leading dimension if model expects 3D input (1, batch, samples)
    if expected_dims == 3 and len(batch.shape) == 2:
        batch = np.expand_dims(batch, axis=0)  # (1, batch_size, samples)

    # Run inference
    input_name = input_info.name
    output_name = session.get_outputs()[0].name

    embeddings = session.run([output_name], {input_name: batch})[0]

    # Handle different output shapes
    # Model outputs (1, batch_size, embedding_dim) -> squeeze to (batch_size, embedding_dim)
    if len(embeddings.shape) == 3 and embeddings.shape[0] == 1:
        embeddings = embeddings.squeeze(0)

    return embeddings


def construct_freesound_url(username: str, freesound_id: int) -> str:
    """Construct FreeSound URL from username and FreeSound ID."""
    return f"https://freesound.org/people/{username}/sounds/{freesound_id}/"


def construct_freesound_embed_url(freesound_id: int) -> str:
    """Construct FreeSound embed/player URL from ID."""
    return f"https://freesound.org/s/{freesound_id}/"


def process_dataset(
    audio_tag_filter: AudioTagFilter,
    filter_audit_json: Path = FILTER_AUDIT_JSON,
):
    """Main processing function."""
    print("=" * 60)
    print("FreeSound-LAION-640k Dataset Processor")
    print("=" * 60)

    # Load ONNX model
    print(f"\n1. Loading ONNX model from {MODEL_PATH}...")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

    session = create_onnx_session(MODEL_PATH)

    print(f"\n2. Loading audio tagger model: {audio_tag_filter.model_id}")
    print(f"   Device: {audio_tag_filter.device}")
    print(f"   Threshold: {audio_tag_filter.threshold}")
    print(f"   Blocked denylist terms: {len(audio_tag_filter.blocked_terms)}")
    print(f"   Matched model labels: {len(audio_tag_filter.blocked_model_labels)}")
    print(f"   Model sample rate: {audio_tag_filter.sample_rate} Hz")

    # Load dataset
    print("\n3. Loading dataset from HuggingFace...")
    print("   Dataset: benjamin-paine/freesound-laion-640k")

    dataset = load_dataset(
        "benjamin-paine/freesound-laion-640k",
        split="train",
        streaming=True,  # Use streaming to avoid downloading all at once
    ).cast_column("audio", datasets.Audio(sampling_rate=SAMPLE_RATE, decode=True))

    # Process and filter
    print("\n4. Processing and filtering clips...")
    print(f"   - Max duration: {MAX_DURATION_SECONDS}s")
    print(f"   - Clip duration for embedding: {CLIP_DURATION_SECONDS}s")
    print(f"   - Tagger batch size: {audio_tag_filter.batch_size}")
    print(f"   - Filter audit JSON: {filter_audit_json}")

    embeddings_data: list[dict[str, Any]] = []
    batch_waveforms: list[np.ndarray] = []
    batch_metadata: list[dict[str, Any]] = []
    filter_batch_waveforms: list[np.ndarray] = []
    filter_batch_metadata: list[dict[str, Any]] = []

    stats = {
        "total_processed": 0,
        "filtered_duration": 0,
        "filtered_tagger": 0,
        "failed_audio": 0,
        "successful": 0,
        "last_checkpoint": 0,  # Track last checkpoint to avoid duplicate saves
    }

    CHECKPOINT_INTERVAL = 1000  # Save every 1000 successful embeddings

    # Create output directory
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    filter_audit_json.parent.mkdir(parents=True, exist_ok=True)

    blocked_label_counts: Counter[str] = Counter()
    filter_audit: dict[str, Any] = {
        "model_id": audio_tag_filter.model_id,
        "threshold": audio_tag_filter.threshold,
        "blocked_terms": sorted(audio_tag_filter.blocked_terms),
        "blocked_model_labels": audio_tag_filter.blocked_model_labels,
        "total_scored": 0,
        "rejected": 0,
        "kept": 0,
        "blocked_label_counts": {},
        "sample_rejections": [],
    }

    def save_checkpoint() -> None:
        with open(OUTPUT_JSON, "w", encoding="utf-8") as handle:
            json.dump(embeddings_data, handle)

    def save_filter_audit() -> None:
        filter_audit["blocked_label_counts"] = dict(
            blocked_label_counts.most_common()
        )
        with open(filter_audit_json, "w", encoding="utf-8") as handle:
            json.dump(filter_audit, handle, indent=2)

    def flush_embedding_batch() -> None:
        nonlocal batch_waveforms, batch_metadata
        if not batch_waveforms:
            return

        try:
            embeddings = extract_embeddings_batch(session, batch_waveforms)
            for emb, meta in zip(embeddings, batch_metadata):
                idx = stats["successful"] + 1
                embeddings_data.append(
                    {
                        "id": f"{idx:012d}",
                        "embedding": emb.tolist(),
                        "freesound_url": construct_freesound_url(
                            meta["username"], meta["freesound_id"]
                        ),
                    }
                )
                stats["successful"] += 1
        except Exception as e:
            print(f"\nError processing batch: {e}")
        finally:
            batch_waveforms = []
            batch_metadata = []

    def _serialize_predictions(
        predictions: list[Any],
        limit: int = 3,
    ) -> list[dict[str, Any]]:
        return [
            {
                "label": prediction.label,
                "score": round(float(prediction.score), 6),
            }
            for prediction in predictions[:limit]
        ]

    def flush_filter_batch() -> None:
        nonlocal filter_batch_waveforms, filter_batch_metadata
        if not filter_batch_waveforms:
            return

        results = audio_tag_filter.predict_batch(
            filter_batch_waveforms,
            sampling_rate=SAMPLE_RATE,
        )
        filter_audit["total_scored"] += len(results)

        for result, waveform, meta in zip(
            results, filter_batch_waveforms, filter_batch_metadata
        ):
            if result.is_blocked:
                stats["filtered_tagger"] += 1
                filter_audit["rejected"] += 1
                for prediction in result.blocked_predictions:
                    blocked_label_counts[prediction.label] += 1

                if len(filter_audit["sample_rejections"]) < MAX_AUDIT_EXAMPLES:
                    filter_audit["sample_rejections"].append(
                        {
                            "freesound_id": meta["freesound_id"],
                            "username": meta["username"],
                            "blocked_predictions": _serialize_predictions(
                                result.blocked_predictions
                            ),
                            "top_predictions": _serialize_predictions(
                                result.top_predictions,
                                limit=5,
                            ),
                        }
                    )
                continue

            filter_audit["kept"] += 1
            batch_waveforms.append(waveform)
            batch_metadata.append(meta)
            if len(batch_waveforms) >= BATCH_SIZE:
                flush_embedding_batch()

        filter_batch_waveforms = []
        filter_batch_metadata = []

    progress = tqdm(
        dataset,
        desc="Processing",
        unit=" clips",
        dynamic_ncols=True,
        bar_format="{l_bar}{bar}| {n_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
    )

    for item in progress:
        stats["total_processed"] += 1

        # Get metadata
        username = item.get("username", "")
        freesound_id = item.get("freesound_id", 0)

        # Get audio data
        audio_data = item.get("audio", {})
        if not audio_data:
            stats["failed_audio"] += 1
            continue

        # Check audio format - datasets library provides audio as dict with 'array' and 'sampling_rate'
        if isinstance(audio_data, dict):
            waveform = audio_data.get("array")
            sr = audio_data.get("sampling_rate", SAMPLE_RATE)

            if waveform is None:
                stats["failed_audio"] += 1
                continue

            waveform = np.array(waveform, dtype=np.float32)

            # Calculate duration
            duration = len(waveform) / sr

            # Filter by duration
            if duration > MAX_DURATION_SECONDS:
                stats["filtered_duration"] += 1
                continue

            # Resample if needed
            if sr != SAMPLE_RATE:
                waveform = librosa.resample(waveform, orig_sr=sr, target_sr=SAMPLE_RATE)

            # Convert stereo to mono if needed
            if len(waveform.shape) > 1:
                waveform = np.mean(waveform, axis=1)

            # Pad/trim to clip duration
            max_samples = int(CLIP_DURATION_SECONDS * SAMPLE_RATE)
            if len(waveform) > max_samples:
                waveform = waveform[:max_samples]
            elif len(waveform) < max_samples:
                waveform = np.pad(
                    waveform, (0, max_samples - len(waveform)), mode="constant"
                )
        else:
            stats["failed_audio"] += 1
            continue

        # Score audio in tagger batches before it reaches the embedding queue
        filter_batch_waveforms.append(waveform)
        filter_batch_metadata.append(
            {
                "username": username,
                "freesound_id": freesound_id,
            }
        )

        if len(filter_batch_waveforms) >= audio_tag_filter.batch_size:
            flush_filter_batch()

        # Update progress with detailed stats
        keep_rate = (
            (stats["successful"] / stats["total_processed"] * 100)
            if stats["total_processed"] > 0
            else 0
        )
        progress.set_postfix_str(
            f"✓ {stats['successful']:,} kept | "
            f"✗ {stats['filtered_tagger'] + stats['filtered_duration']:,} filtered | "
            f"⚠ {stats['failed_audio']} failed | "
            f"({keep_rate:.1f}% keep rate)"
        )

        # Save checkpoint periodically
        if (
            stats["successful"] > 0
            and stats["successful"] - stats["last_checkpoint"] >= CHECKPOINT_INTERVAL
        ):
            progress.write(
                f"   💾 Checkpoint: Saving {len(embeddings_data):,} embeddings..."
            )
            save_checkpoint()
            save_filter_audit()
            stats["last_checkpoint"] = stats["successful"]

    flush_filter_batch()
    flush_embedding_batch()

    # Save final results
    print(f"\n5. Saving {len(embeddings_data)} embeddings to {OUTPUT_JSON}...")
    save_checkpoint()
    save_filter_audit()

    # Print stats
    print("\n" + "=" * 60)
    print("Processing Statistics:")
    print("=" * 60)
    print(f"  Total processed:      {stats['total_processed']:,}")
    print(f"  Filtered (duration):  {stats['filtered_duration']:,}")
    print(f"  Filtered (tagger):    {stats['filtered_tagger']:,}")
    print(f"  Failed audio:         {stats['failed_audio']:,}")
    print(f"  Successful:           {stats['successful']:,}")
    print(f"  Filter audit JSON:    {filter_audit_json}")
    print("=" * 60)

    return embeddings_data


def upload_to_pinecone(embeddings_data: Optional[list] = None):
    """Upload embeddings to Pinecone."""
    print("\n5. Uploading to Pinecone...")

    if embeddings_data is None:
        if not OUTPUT_JSON.exists():
            print(f"Error: {OUTPUT_JSON} not found. Run processing first.")
            return

        print(f"   Loading embeddings from {OUTPUT_JSON}...")
        with open(OUTPUT_JSON, "r", encoding="utf-8") as f:
            embeddings_data = json.load(f)

    print(f"   Total embeddings to upload: {len(embeddings_data):,}")

    # Get API key
    api_key = os.getenv("PINECONE_API_KEY") or getpass.getpass("Pinecone API Key: ")
    if not api_key:
        raise ValueError("Pinecone API Key is required")

    # Connect to Pinecone
    pc = Pinecone(api_key=api_key)
    index = pc.Index(INDEX_NAME)
    print(f"   Connected to index: {INDEX_NAME}")

    # Upsert in batches
    batch_size = 100
    print(f"   Uploading in batches of {batch_size}...")

    for i in tqdm(range(0, len(embeddings_data), batch_size), desc="Uploading"):
        batch = embeddings_data[i : i + batch_size]

        vectors_to_upsert = [
            {
                "id": item["id"],
                "values": item["embedding"],
                "metadata": {"freesound_url": item["freesound_url"]},
            }
            for item in batch
        ]

        try:
            index.upsert(vectors=vectors_to_upsert)
        except Exception as e:
            print(f"\nError upserting batch {i // batch_size + 1}: {e}")

    print("\n   Upload complete!")
    print(index.describe_index_stats())


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Process FreeSound-LAION-640k dataset and upload to Pinecone"
    )
    parser.add_argument(
        "--upload-only",
        action="store_true",
        help="Skip processing, only upload existing embeddings to Pinecone",
    )
    parser.add_argument(
        "--process-only",
        action="store_true",
        help="Only process dataset, skip Pinecone upload",
    )
    parser.add_argument(
        "--blocked-labels-file",
        type=str,
        default=str(DEFAULT_BLOCKED_LABELS_PATH),
        help="Path to text file with model output labels/terms to exclude.",
    )
    parser.add_argument(
        "--tagger-model",
        type=str,
        default=DEFAULT_TAGGER_MODEL_ID,
        help="Hugging Face model id for the audio tagger.",
    )
    parser.add_argument(
        "--tagger-threshold",
        type=float,
        default=DEFAULT_TAGGER_THRESHOLD,
        help="Clipwise threshold above which a blocked predicted class rejects the clip.",
    )
    parser.add_argument(
        "--tagger-batch-size",
        type=int,
        default=DEFAULT_TAGGER_BATCH_SIZE,
        help="Batch size for model-based audio tag filtering.",
    )
    parser.add_argument(
        "--filter-audit-json",
        type=Path,
        default=FILTER_AUDIT_JSON,
        help="Where to write the model filter audit summary.",
    )
    parser.add_argument(
        "--tags-file",
        type=str,
        default=None,
        help=argparse.SUPPRESS,
    )

    args = parser.parse_args()

    blocked_labels_file = args.blocked_labels_file
    if args.tags_file:
        print(
            "Warning: --tags-file is deprecated for FreeSound filtering; "
            "treating it as --blocked-labels-file."
        )
        blocked_labels_file = args.tags_file

    if args.upload_only:
        upload_to_pinecone()
    else:
        blocked_labels = load_blocked_labels(blocked_labels_file)
        audio_tag_filter = AudioTagFilter(
            model_id=args.tagger_model,
            blocked_terms=blocked_labels,
            threshold=args.tagger_threshold,
            batch_size=args.tagger_batch_size,
        )
        embeddings_data = process_dataset(
            audio_tag_filter=audio_tag_filter,
            filter_audit_json=args.filter_audit_json,
        )

        if args.process_only:
            return

        upload_to_pinecone(embeddings_data)


if __name__ == "__main__":
    main()
