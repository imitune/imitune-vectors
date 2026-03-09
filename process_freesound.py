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
TAG_RESULTS_JSONL = Path("freesound_tag_results.jsonl")
TAG_AUDIT_JSON = Path("freesound_tagging_audit.json")
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


def load_streaming_dataset():
    """Load the streaming Hugging Face Freesound dataset."""
    return load_dataset(
        "benjamin-paine/freesound-laion-640k",
        split="train",
        streaming=True,  # Use streaming to avoid downloading all at once
    ).cast_column("audio", datasets.Audio(sampling_rate=SAMPLE_RATE, decode=True))


def _serialize_predictions(
    predictions: list[Any],
    limit: Optional[int] = None,
) -> list[dict[str, Any]]:
    selected = predictions if limit is None else predictions[:limit]
    return [
        {
            "label": prediction.label,
            "score": round(float(prediction.score), 6),
        }
        for prediction in selected
    ]


def prepare_dataset_item(item: dict[str, Any]) -> tuple[dict[str, Any], Optional[np.ndarray], str]:
    """Normalize one dataset item into metadata + waveform or a skip reason."""
    metadata = {
        "username": item.get("username", ""),
        "freesound_id": item.get("freesound_id", 0),
    }

    audio_data = item.get("audio", {})
    if not audio_data or not isinstance(audio_data, dict):
        return metadata, None, "failed_audio"

    waveform = audio_data.get("array")
    sr = audio_data.get("sampling_rate", SAMPLE_RATE)
    if waveform is None:
        return metadata, None, "failed_audio"

    waveform = np.array(waveform, dtype=np.float32)
    duration = len(waveform) / sr
    if duration > MAX_DURATION_SECONDS:
        return metadata, None, "filtered_duration"

    if sr != SAMPLE_RATE:
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=SAMPLE_RATE)

    if len(waveform.shape) > 1:
        waveform = np.mean(waveform, axis=1)

    max_samples = int(CLIP_DURATION_SECONDS * SAMPLE_RATE)
    if len(waveform) > max_samples:
        waveform = waveform[:max_samples]
    elif len(waveform) < max_samples:
        waveform = np.pad(waveform, (0, max_samples - len(waveform)), mode="constant")

    return metadata, waveform.astype(np.float32), "ok"


def load_tag_results(tag_results_jsonl: Path) -> dict[str, dict[str, Any]]:
    """Load saved tagger outputs keyed by Freesound id."""
    if not tag_results_jsonl.exists():
        raise FileNotFoundError(
            f"Tag results not found: {tag_results_jsonl}. Run tagging first or omit --process-only."
        )

    results: dict[str, dict[str, Any]] = {}
    with open(tag_results_jsonl, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            results[str(record["freesound_id"])] = record

    if not results:
        raise ValueError(f"Tag results file is empty: {tag_results_jsonl}")

    return results


def blocked_predictions_from_record(
    record: dict[str, Any],
    threshold: float,
) -> list[dict[str, Any]]:
    """Return saved matched predictions above threshold."""
    return [
        prediction
        for prediction in record.get("matched_predictions", [])
        if float(prediction["score"]) >= threshold
    ]


def tag_dataset(
    audio_tag_filter: AudioTagFilter,
    tag_results_jsonl: Path = TAG_RESULTS_JSONL,
    tag_audit_json: Path = TAG_AUDIT_JSON,
) -> None:
    """Run the tagger once and persist reusable scores for later filtering."""
    print("=" * 60)
    print("FreeSound-LAION-640k Tagger")
    print("=" * 60)

    print(f"\n1. Loading audio tagger model: {audio_tag_filter.model_id}")
    print(f"   Device: {audio_tag_filter.device}")
    print(f"   Blocked denylist terms: {len(audio_tag_filter.blocked_terms)}")
    print(f"   Matched model labels: {len(audio_tag_filter.blocked_model_labels)}")
    print(f"   Model sample rate: {audio_tag_filter.sample_rate} Hz")
    print(f"   Tagger batch size: {audio_tag_filter.batch_size}")

    # Load dataset
    print("\n2. Loading dataset from HuggingFace...")
    print("   Dataset: benjamin-paine/freesound-laion-640k")
    dataset = load_streaming_dataset()

    # Process and tag
    print("\n3. Tagging clips...")
    print(f"   - Max duration: {MAX_DURATION_SECONDS}s")
    print(f"   - Clip duration for tagging: {CLIP_DURATION_SECONDS}s")
    print(f"   - Tag output JSONL: {tag_results_jsonl}")
    print(f"   - Tag audit JSON: {tag_audit_json}")

    batch_waveforms: list[np.ndarray] = []
    batch_metadata: list[dict[str, Any]] = []

    stats = {
        "total_processed": 0,
        "filtered_duration": 0,
        "failed_audio": 0,
        "tagged": 0,
    }

    tag_results_jsonl.parent.mkdir(parents=True, exist_ok=True)
    tag_audit_json.parent.mkdir(parents=True, exist_ok=True)

    matched_label_counts: Counter[str] = Counter()
    tag_audit: dict[str, Any] = {
        "model_id": audio_tag_filter.model_id,
        "blocked_terms": sorted(audio_tag_filter.blocked_terms),
        "blocked_model_labels": audio_tag_filter.blocked_model_labels,
        "total_tagged": 0,
        "matched_label_counts": {},
        "sample_predictions": [],
    }

    def save_tag_audit() -> None:
        tag_audit["matched_label_counts"] = dict(
            matched_label_counts.most_common()
        )
        with open(tag_audit_json, "w", encoding="utf-8") as handle:
            json.dump(tag_audit, handle, indent=2)

    progress = tqdm(
        dataset,
        desc="Tagging",
        unit=" clips",
        dynamic_ncols=True,
        bar_format="{l_bar}{bar}| {n_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
    )

    with open(tag_results_jsonl, "w", encoding="utf-8") as tag_handle:
        def flush_tag_batch() -> None:
            nonlocal batch_waveforms, batch_metadata
            if not batch_waveforms:
                return

            results = audio_tag_filter.predict_batch(
                batch_waveforms,
                sampling_rate=SAMPLE_RATE,
            )
            for result, meta in zip(results, batch_metadata):
                stats["tagged"] += 1
                tag_audit["total_tagged"] += 1
                if result.matched_predictions:
                    matched_label_counts[result.matched_predictions[0].label] += 1

                record = {
                    "freesound_id": meta["freesound_id"],
                    "username": meta["username"],
                    "matched_predictions": _serialize_predictions(
                        result.matched_predictions
                    ),
                    "top_predictions": _serialize_predictions(
                        result.top_predictions,
                        limit=5,
                    ),
                }
                tag_handle.write(json.dumps(record) + "\n")

                if len(tag_audit["sample_predictions"]) < MAX_AUDIT_EXAMPLES:
                    tag_audit["sample_predictions"].append(record)

            tag_handle.flush()
            batch_waveforms = []
            batch_metadata = []

        for item in progress:
            stats["total_processed"] += 1
            metadata, waveform, status = prepare_dataset_item(item)
            if status == "filtered_duration":
                stats["filtered_duration"] += 1
                continue
            if status != "ok" or waveform is None:
                stats["failed_audio"] += 1
                continue

            batch_waveforms.append(waveform)
            batch_metadata.append(metadata)

            if len(batch_waveforms) >= audio_tag_filter.batch_size:
                flush_tag_batch()

            progress.set_postfix_str(
                f"✓ {stats['tagged']:,} tagged | "
                f"✗ {stats['filtered_duration']:,} duration | "
                f"⚠ {stats['failed_audio']} failed"
            )

        flush_tag_batch()

    save_tag_audit()

    print("\n" + "=" * 60)
    print("Tagging Statistics:")
    print("=" * 60)
    print(f"  Total processed:      {stats['total_processed']:,}")
    print(f"  Filtered (duration):  {stats['filtered_duration']:,}")
    print(f"  Failed audio:         {stats['failed_audio']:,}")
    print(f"  Tagged clips:         {stats['tagged']:,}")
    print(f"  Tag output JSONL:     {tag_results_jsonl}")
    print(f"  Tag audit JSON:       {tag_audit_json}")
    print("=" * 60)


def process_dataset(
    tag_results_jsonl: Path = TAG_RESULTS_JSONL,
    filter_threshold: float = DEFAULT_TAGGER_THRESHOLD,
    filter_audit_json: Path = FILTER_AUDIT_JSON,
    embedding_batch_size: int = BATCH_SIZE,
) -> list[dict[str, Any]]:
    """Extract embeddings using previously saved tagger results for filtering."""
    print("=" * 60)
    print("FreeSound-LAION-640k Embedding Processor")
    print("=" * 60)

    print(f"\n1. Loading ONNX model from {MODEL_PATH}...")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    session = create_onnx_session(MODEL_PATH)

    print(f"\n2. Loading saved tag results from {tag_results_jsonl}...")
    tag_results = load_tag_results(tag_results_jsonl)
    print(f"   Loaded {len(tag_results):,} tagged clips")
    print(f"   Filter threshold: {filter_threshold}")

    print("\n3. Loading dataset from HuggingFace...")
    print("   Dataset: benjamin-paine/freesound-laion-640k")
    dataset = load_streaming_dataset()

    print("\n4. Processing and filtering clips...")
    print(f"   - Max duration: {MAX_DURATION_SECONDS}s")
    print(f"   - Clip duration for embedding: {CLIP_DURATION_SECONDS}s")
    print(f"   - Embedding batch size: {embedding_batch_size}")
    print(f"   - Filter audit JSON: {filter_audit_json}")

    embeddings_data: list[dict[str, Any]] = []
    batch_waveforms: list[np.ndarray] = []
    batch_metadata: list[dict[str, Any]] = []

    stats = {
        "total_processed": 0,
        "filtered_duration": 0,
        "filtered_tagger": 0,
        "failed_audio": 0,
        "missing_tags": 0,
        "successful": 0,
        "last_checkpoint": 0,
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    filter_audit_json.parent.mkdir(parents=True, exist_ok=True)

    blocked_label_counts: Counter[str] = Counter()
    filter_audit: dict[str, Any] = {
        "tag_results_jsonl": str(tag_results_jsonl),
        "threshold": filter_threshold,
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

    progress = tqdm(
        dataset,
        desc="Embedding",
        unit=" clips",
        dynamic_ncols=True,
        bar_format="{l_bar}{bar}| {n_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
    )

    for item in progress:
        stats["total_processed"] += 1
        metadata, waveform, status = prepare_dataset_item(item)
        if status == "filtered_duration":
            stats["filtered_duration"] += 1
            continue
        if status != "ok" or waveform is None:
            stats["failed_audio"] += 1
            continue

        record = tag_results.get(str(metadata["freesound_id"]))
        if record is None:
            stats["missing_tags"] += 1
            continue

        filter_audit["total_scored"] += 1
        blocked_predictions = blocked_predictions_from_record(record, filter_threshold)
        if blocked_predictions:
            stats["filtered_tagger"] += 1
            filter_audit["rejected"] += 1
            for prediction in blocked_predictions:
                blocked_label_counts[prediction["label"]] += 1

            if len(filter_audit["sample_rejections"]) < MAX_AUDIT_EXAMPLES:
                filter_audit["sample_rejections"].append(
                    {
                        "freesound_id": metadata["freesound_id"],
                        "username": metadata["username"],
                        "blocked_predictions": blocked_predictions[:5],
                        "top_predictions": record.get("top_predictions", [])[:5],
                    }
                )
            continue

        filter_audit["kept"] += 1
        batch_waveforms.append(waveform)
        batch_metadata.append(metadata)

        if len(batch_waveforms) >= embedding_batch_size:
            flush_embedding_batch()

        keep_rate = (
            (stats["successful"] / stats["total_processed"] * 100)
            if stats["total_processed"] > 0
            else 0
        )
        progress.set_postfix_str(
            f"✓ {stats['successful']:,} kept | "
            f"✗ {stats['filtered_tagger'] + stats['filtered_duration']:,} filtered | "
            f"⚠ {stats['failed_audio']} failed | "
            f"⌕ {stats['missing_tags']} missing tags | "
            f"({keep_rate:.1f}% keep rate)"
        )

        if (
            stats["successful"] > 0
            and stats["successful"] - stats["last_checkpoint"] >= 1000
        ):
            progress.write(
                f"   💾 Checkpoint: Saving {len(embeddings_data):,} embeddings..."
            )
            save_checkpoint()
            save_filter_audit()
            stats["last_checkpoint"] = stats["successful"]

    flush_embedding_batch()
    print(f"\n5. Saving {len(embeddings_data)} embeddings to {OUTPUT_JSON}...")
    save_checkpoint()
    save_filter_audit()

    print("\n" + "=" * 60)
    print("Embedding Statistics:")
    print("=" * 60)
    print(f"  Total processed:      {stats['total_processed']:,}")
    print(f"  Filtered (duration):  {stats['filtered_duration']:,}")
    print(f"  Filtered (tagger):    {stats['filtered_tagger']:,}")
    print(f"  Failed audio:         {stats['failed_audio']:,}")
    print(f"  Missing tag records:  {stats['missing_tags']:,}")
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
        help="Only extract embeddings using an existing tag results file.",
    )
    parser.add_argument(
        "--tag-only",
        action="store_true",
        help="Only run audio tagging and write tag results; do not extract embeddings.",
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
        help="Clipwise threshold used when filtering embeddings from saved tagger output.",
    )
    parser.add_argument(
        "--tagger-batch-size",
        type=int,
        default=DEFAULT_TAGGER_BATCH_SIZE,
        help="Batch size for the tagging pass.",
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=BATCH_SIZE,
        help="Batch size for embedding extraction.",
    )
    parser.add_argument(
        "--tag-results-jsonl",
        type=Path,
        default=TAG_RESULTS_JSONL,
        help="Where to write/read the reusable tagger output.",
    )
    parser.add_argument(
        "--tag-audit-json",
        type=Path,
        default=TAG_AUDIT_JSON,
        help="Where to write the tagging audit summary.",
    )
    parser.add_argument(
        "--filter-audit-json",
        type=Path,
        default=FILTER_AUDIT_JSON,
        help="Where to write the model filter audit summary.",
    )
    parser.add_argument(
        "--retag",
        action="store_true",
        help="Force regeneration of the tag results file before embedding extraction.",
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

    selected_modes = sum(
        [
            int(args.upload_only),
            int(args.process_only),
            int(args.tag_only),
        ]
    )
    if selected_modes > 1:
        raise ValueError("Use at most one of --upload-only, --process-only, or --tag-only")

    if args.upload_only:
        upload_to_pinecone()
        return

    if args.process_only:
        process_dataset(
            tag_results_jsonl=args.tag_results_jsonl,
            filter_threshold=args.tagger_threshold,
            filter_audit_json=args.filter_audit_json,
            embedding_batch_size=args.embedding_batch_size,
        )
        return

    should_tag = args.tag_only or args.retag or not args.tag_results_jsonl.exists()
    if should_tag:
        blocked_labels = load_blocked_labels(blocked_labels_file)
        audio_tag_filter = AudioTagFilter(
            model_id=args.tagger_model,
            blocked_terms=blocked_labels,
            threshold=args.tagger_threshold,
            batch_size=args.tagger_batch_size,
        )
        tag_dataset(
            audio_tag_filter=audio_tag_filter,
            tag_results_jsonl=args.tag_results_jsonl,
            tag_audit_json=args.tag_audit_json,
        )
    else:
        print(f"Using existing tag results from {args.tag_results_jsonl}")

    if args.tag_only:
        return

    embeddings_data = process_dataset(
        tag_results_jsonl=args.tag_results_jsonl,
        filter_threshold=args.tagger_threshold,
        filter_audit_json=args.filter_audit_json,
        embedding_batch_size=args.embedding_batch_size,
    )
    upload_to_pinecone(embeddings_data)


if __name__ == "__main__":
    main()
