#!/usr/bin/env python3
"""Inspect embedding JSON files and report basic sanity statistics."""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np


COMMON_INPUT_CANDIDATES = [
    Path("fsd50k_embeddings.json"),
    Path("freesound_embeddings.json"),
]


def format_float(value: float) -> str:
    """Format float values compactly for terminal output."""
    return f"{value:,.6f}"


def resolve_input_path(input_path: Optional[Path]) -> Path:
    """Resolve input path or choose a common default in the current directory."""
    if input_path is not None:
        return input_path.expanduser()

    existing = [path for path in COMMON_INPUT_CANDIDATES if path.exists()]
    if len(existing) == 1:
        return existing[0]
    if len(existing) > 1:
        names = ", ".join(str(path) for path in existing)
        raise ValueError(f"Multiple candidate files found, please pass one explicitly: {names}")

    raise FileNotFoundError(
        "No input JSON provided and no default embeddings JSON found in the current directory"
    )


def load_records(input_path: Path) -> list[dict[str, Any]]:
    """Load embedding records from JSON."""
    with open(input_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, list):
        raise ValueError("Expected a top-level JSON list of embedding records")

    records = [item for item in data if isinstance(item, dict)]
    if not records:
        raise ValueError("No embedding records found in JSON")

    return records


def collect_embeddings(records: Iterable[dict[str, Any]]) -> tuple[np.ndarray, dict[str, int]]:
    """Collect valid embeddings into a 2D float32 array and track skip counts."""
    skip_counts = Counter(
        {
            "missing_embedding": 0,
            "invalid_embedding": 0,
            "wrong_dimension": 0,
        }
    )

    rows: list[np.ndarray] = []
    expected_dim: Optional[int] = None

    for record in records:
        embedding = record.get("embedding")
        if embedding is None:
            skip_counts["missing_embedding"] += 1
            continue

        try:
            vector = np.asarray(embedding, dtype=np.float32)
        except (TypeError, ValueError):
            skip_counts["invalid_embedding"] += 1
            continue

        if vector.ndim != 1 or vector.size == 0:
            skip_counts["invalid_embedding"] += 1
            continue

        if expected_dim is None:
            expected_dim = int(vector.shape[0])
        elif vector.shape[0] != expected_dim:
            skip_counts["wrong_dimension"] += 1
            continue

        rows.append(vector)

    if not rows:
        raise ValueError("No valid embeddings found after filtering malformed records")

    return np.stack(rows, axis=0), dict(skip_counts)


def summarize_ids(records: Iterable[dict[str, Any]]) -> tuple[int, int]:
    """Return total and duplicate id counts."""
    ids = [str(record.get("id")) for record in records if record.get("id") is not None]
    duplicate_count = sum(count - 1 for count in Counter(ids).values() if count > 1)
    return len(ids), duplicate_count


def print_summary(
    input_path: Path,
    total_records: int,
    embeddings: np.ndarray,
    skip_counts: dict[str, int],
    id_count: int,
    duplicate_id_count: int,
) -> None:
    """Print summary statistics for embeddings."""
    norms = np.linalg.norm(embeddings, axis=1)
    finite_mask = np.isfinite(embeddings)
    nan_count = int(np.isnan(embeddings).sum())
    inf_count = int(np.isinf(embeddings).sum())
    zero_norm_count = int(np.count_nonzero(norms == 0.0))

    print("=" * 60)
    print("Embedding Analysis")
    print("=" * 60)
    print(f"Input file:          {input_path}")
    print(f"Total records:       {total_records:,}")
    print(f"Valid embeddings:    {embeddings.shape[0]:,}")
    print(f"Embedding dimension: {embeddings.shape[1]:,}")
    print(f"Record ids present:  {id_count:,}")
    print(f"Duplicate ids:       {duplicate_id_count:,}")
    print()
    print("Skipped records")
    print(f"  Missing embedding: {skip_counts['missing_embedding']:,}")
    print(f"  Invalid embedding: {skip_counts['invalid_embedding']:,}")
    print(f"  Wrong dimension:   {skip_counts['wrong_dimension']:,}")
    print()
    print("Value statistics")
    print(f"  Mean:              {format_float(float(np.mean(embeddings)))}")
    print(f"  Std:               {format_float(float(np.std(embeddings)))}")
    print(f"  Mean abs:          {format_float(float(np.mean(np.abs(embeddings))))}")
    print(f"  Min:               {format_float(float(np.min(embeddings)))}")
    print(f"  Max:               {format_float(float(np.max(embeddings)))}")
    print(f"  1st percentile:    {format_float(float(np.percentile(embeddings, 1)))}")
    print(f"  99th percentile:   {format_float(float(np.percentile(embeddings, 99)))}")
    print(f"  Finite values:     {int(np.count_nonzero(finite_mask)):,}/{embeddings.size:,}")
    print(f"  NaN values:        {nan_count:,}")
    print(f"  Inf values:        {inf_count:,}")
    print()
    print("Vector norm statistics")
    print(f"  Mean norm:         {format_float(float(np.mean(norms)))}")
    print(f"  Std norm:          {format_float(float(np.std(norms)))}")
    print(f"  Min norm:          {format_float(float(np.min(norms)))}")
    print(f"  Median norm:       {format_float(float(np.median(norms)))}")
    print(f"  Max norm:          {format_float(float(np.max(norms)))}")
    print(f"  Zero vectors:      {zero_norm_count:,}")
    print()

    warnings: list[str] = []
    if nan_count > 0:
        warnings.append("NaN values detected")
    if inf_count > 0:
        warnings.append("Inf values detected")
    if zero_norm_count > 0:
        warnings.append("zero-norm embeddings detected")
    if skip_counts["wrong_dimension"] > 0:
        warnings.append("inconsistent embedding dimensions detected")
    if float(np.std(embeddings)) < 1e-6:
        warnings.append("overall standard deviation is extremely small")
    if duplicate_id_count > 0:
        warnings.append("duplicate ids detected")

    print("Checks")
    if warnings:
        for warning in warnings:
            print(f"  Warning: {warning}")
    else:
        print("  Basic sanity checks look fine")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect an embeddings JSON file and report sanity statistics"
    )
    parser.add_argument(
        "input_json",
        nargs="?",
        type=Path,
        default=None,
        help="Path to embeddings JSON. If omitted, tries common filenames in the current directory.",
    )
    args = parser.parse_args()

    input_path = resolve_input_path(args.input_json)
    records = load_records(input_path)
    embeddings, skip_counts = collect_embeddings(records)
    id_count, duplicate_id_count = summarize_ids(records)
    print_summary(
        input_path=input_path,
        total_records=len(records),
        embeddings=embeddings,
        skip_counts=skip_counts,
        id_count=id_count,
        duplicate_id_count=duplicate_id_count,
    )


if __name__ == "__main__":
    main()