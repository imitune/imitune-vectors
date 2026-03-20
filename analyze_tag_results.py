#!/usr/bin/env python3
"""Inspect saved Freesound tagger outputs and compare threshold choices."""

from __future__ import annotations

import argparse
import html
import json
from collections import Counter
from pathlib import Path
from typing import Any, Optional

DEFAULT_TAG_RESULTS_JSONL = Path("freesound_tag_results.jsonl")
DEFAULT_THRESHOLDS = [0.15, 0.2, 0.25, 0.3, 0.35, 0.4]


def load_filter_labels(filter_labels_file: Optional[str] = None) -> Optional[set[str]]:
    """Load exact matched model labels to evaluate."""
    if not filter_labels_file:
        return None

    labels: set[str] = set()
    with open(filter_labels_file, "r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            labels.add(stripped)

    if not labels:
        raise ValueError(f"Filter labels file is empty: {filter_labels_file}")

    return labels


def load_tag_records(tag_results_jsonl: Path) -> list[dict[str, Any]]:
    """Load saved tagger outputs."""
    records: list[dict[str, Any]] = []
    with open(tag_results_jsonl, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))

    if not records:
        raise ValueError(f"Tag results file is empty: {tag_results_jsonl}")

    return records


def select_predictions(
    record: dict[str, Any],
    filter_labels: Optional[set[str]] = None,
) -> list[dict[str, Any]]:
    """Return the matched predictions relevant to the current label subset."""
    predictions = record.get("matched_predictions", [])
    if filter_labels is None:
        return predictions
    return [
        prediction for prediction in predictions if prediction["label"] in filter_labels
    ]


def format_url(freesound_id: Any) -> str:
    return f"https://freesound.org/s/{freesound_id}/"


def format_user_url(username: str, freesound_id: Any) -> str:
    if not username:
        return format_url(freesound_id)
    return f"https://freesound.org/people/{username}/sounds/{freesound_id}/"


def print_label_summary(
    records: list[dict[str, Any]],
    filter_labels: Optional[set[str]],
    top_labels: int,
) -> tuple[Counter[str], dict[str, float]]:
    """Print counts and max score per exact matched model label."""
    label_counts: Counter[str] = Counter()
    label_max_scores: dict[str, float] = {}

    clips_with_matches = 0
    for record in records:
        predictions = select_predictions(record, filter_labels)
        if predictions:
            clips_with_matches += 1
        for prediction in predictions:
            label = prediction["label"]
            score = float(prediction["score"])
            label_counts[label] += 1
            label_max_scores[label] = max(score, label_max_scores.get(label, 0.0))

    print(f"Total tagged clips: {len(records):,}")
    print(f"Clips with selected matched labels: {clips_with_matches:,}")
    print()
    print("Top matched labels:")
    for label, count in label_counts.most_common(top_labels):
        print(f"  {label}: {count:,} clips, max_score={label_max_scores[label]:.4f}")
    print()

    return label_counts, label_max_scores


def summarize_threshold(
    records: list[dict[str, Any]],
    threshold: float,
    filter_labels: Optional[set[str]],
) -> tuple[int, Counter[str]]:
    """Return aggregate reject counts for one threshold."""
    rejected = 0
    blocked_label_counts: Counter[str] = Counter()

    for record in records:
        blocked_predictions = [
            prediction
            for prediction in select_predictions(record, filter_labels)
            if float(prediction["score"]) >= threshold
        ]
        if not blocked_predictions:
            continue

        rejected += 1
        for prediction in blocked_predictions:
            blocked_label_counts[prediction["label"]] += 1

    return rejected, blocked_label_counts


def print_threshold_summaries(
    records: list[dict[str, Any]],
    thresholds: list[float],
    filter_labels: Optional[set[str]],
    top_labels: int,
) -> None:
    """Print clip reject counts and top labels across thresholds."""
    for threshold in thresholds:
        rejected, blocked_label_counts = summarize_threshold(
            records,
            threshold,
            filter_labels,
        )
        rejected_pct = rejected / len(records) * 100
        print(
            f"Threshold {threshold:.2f}: reject {rejected:,} / {len(records):,} "
            f"({rejected_pct:.2f}%)"
        )
        for label, count in blocked_label_counts.most_common(top_labels):
            print(f"  {label}: {count:,}")
        print()


def collect_borderline_examples(
    records: list[dict[str, Any]],
    threshold: float,
    filter_labels: Optional[set[str]],
    limit: int,
) -> tuple[
    list[tuple[float, dict[str, Any], list[dict[str, Any]]]],
    list[tuple[float, dict[str, Any], list[dict[str, Any]]]],
]:
    """Collect near-threshold kept and rejected examples for manual review."""
    kept_candidates: list[tuple[float, dict[str, Any], list[dict[str, Any]]]] = []
    rejected_candidates: list[tuple[float, dict[str, Any], list[dict[str, Any]]]] = []

    for record in records:
        predictions = select_predictions(record, filter_labels)
        if not predictions:
            continue

        max_score = max(float(prediction["score"]) for prediction in predictions)
        distance = abs(max_score - threshold)
        blocked_predictions = [
            prediction
            for prediction in predictions
            if float(prediction["score"]) >= threshold
        ]

        if blocked_predictions:
            rejected_candidates.append((distance, record, blocked_predictions))
        else:
            kept_candidates.append((distance, record, predictions))

    kept_candidates.sort(key=lambda item: item[0])
    rejected_candidates.sort(key=lambda item: item[0])
    return kept_candidates[:limit], rejected_candidates[:limit]


def print_borderline_examples(
    records: list[dict[str, Any]],
    threshold: float,
    filter_labels: Optional[set[str]],
    limit: int,
) -> None:
    """Show near-boundary kept and rejected examples for manual inspection."""
    kept_candidates, rejected_candidates = collect_borderline_examples(
        records,
        threshold,
        filter_labels,
        limit,
    )

    print(f"Borderline rejected examples near threshold {threshold:.2f}:")
    for _, record, predictions in rejected_candidates:
        top_prediction = predictions[0]
        print(
            f"  {record['freesound_id']} score={float(top_prediction['score']):.4f} "
            f"label={top_prediction['label']} url={format_url(record['freesound_id'])}"
        )
    print()

    print(f"Borderline kept examples near threshold {threshold:.2f}:")
    for _, record, predictions in kept_candidates:
        top_prediction = predictions[0]
        print(
            f"  {record['freesound_id']} score={float(top_prediction['score']):.4f} "
            f"label={top_prediction['label']} url={format_url(record['freesound_id'])}"
        )
    print()


def render_predictions(predictions: list[dict[str, Any]], limit: int = 5) -> str:
    """Render predictions as an HTML-safe short list."""
    parts = [
        f"{html.escape(prediction['label'])} ({float(prediction['score']):.3f})"
        for prediction in predictions[:limit]
    ]
    return "<br>".join(parts)


def write_html_report(
    output_path: Path,
    records: list[dict[str, Any]],
    thresholds: list[float],
    filter_labels: Optional[set[str]],
    inspect_limit: int,
    top_labels: int,
) -> None:
    """Generate an HTML review interface for threshold tuning."""
    sections: list[str] = []
    scope_label = (
        f"{len(filter_labels)} exact filter labels"
        if filter_labels is not None
        else "all saved matched labels"
    )

    for threshold in thresholds:
        rejected, blocked_label_counts = summarize_threshold(
            records,
            threshold,
            filter_labels,
        )
        kept_examples, rejected_examples = collect_borderline_examples(
            records,
            threshold,
            filter_labels,
            inspect_limit,
        )
        rejected_pct = rejected / len(records) * 100

        top_label_items = "".join(
            f"<li><strong>{html.escape(label)}</strong>: {count:,}</li>"
            for label, count in blocked_label_counts.most_common(top_labels)
        )

        def render_rows(
            items: list[tuple[float, dict[str, Any], list[dict[str, Any]]]],
        ) -> str:
            rows = []
            for distance, record, predictions in items:
                freesound_id = record["freesound_id"]
                username = record.get("username", "")
                rows.append(
                    "<tr>"
                    f"<td>{freesound_id}</td>"
                    f"<td>{distance:.6f}</td>"
                    f"<td>{html.escape(predictions[0]['label'])}</td>"
                    f"<td>{float(predictions[0]['score']):.4f}</td>"
                    f"<td>{render_predictions(predictions)}</td>"
                    f"<td>{render_predictions(record.get('top_predictions', []))}</td>"
                    f"<td><a href=\"{html.escape(format_url(freesound_id))}\" target=\"_blank\">short</a> | "
                    f"<a href=\"{html.escape(format_user_url(username, freesound_id))}\" target=\"_blank\">page</a></td>"
                    "</tr>"
                )
            return "".join(rows)

        sections.append(
            f"""
            <section class="threshold-section">
              <h2>Threshold {threshold:.2f}</h2>
              <p class="summary">Rejects <strong>{rejected:,}</strong> / {len(records):,} clips ({rejected_pct:.2f}%).</p>
              <div class="summary-grid">
                <div>
                  <h3>Top Rejection Labels</h3>
                  <ol>{top_label_items}</ol>
                </div>
                <div>
                  <h3>How To Use</h3>
                  <p>Open the clip links in a new tab and listen to the near-threshold examples on both sides. If many rejected clips sound acceptable, raise the threshold or trim the exact label list. If many kept clips still contain speech/music, lower the threshold or add exact labels.</p>
                </div>
              </div>
              <div class="tables">
                <div>
                  <h3>Borderline Rejected</h3>
                  <table>
                    <thead>
                      <tr><th>ID</th><th>|score-threshold|</th><th>Top blocked label</th><th>Score</th><th>Blocked predictions</th><th>Top predictions</th><th>Links</th></tr>
                    </thead>
                    <tbody>{render_rows(rejected_examples)}</tbody>
                  </table>
                </div>
                <div>
                  <h3>Borderline Kept</h3>
                  <table>
                    <thead>
                      <tr><th>ID</th><th>|score-threshold|</th><th>Top matched label</th><th>Score</th><th>Matched predictions</th><th>Top predictions</th><th>Links</th></tr>
                    </thead>
                    <tbody>{render_rows(kept_examples)}</tbody>
                  </table>
                </div>
              </div>
            </section>
            """
        )

    html_content = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Freesound Threshold Review</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 24px; line-height: 1.4; }}
    h1, h2, h3 {{ margin-bottom: 0.4rem; }}
    .summary-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; margin-bottom: 16px; }}
    .tables {{ display: grid; grid-template-columns: 1fr; gap: 24px; }}
    .threshold-section {{ margin-bottom: 40px; padding-bottom: 24px; border-bottom: 1px solid #ddd; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; vertical-align: top; text-align: left; }}
    th {{ background: #f5f5f5; position: sticky; top: 0; }}
    code {{ background: #f4f4f4; padding: 2px 4px; border-radius: 4px; }}
    .summary {{ font-size: 16px; }}
  </style>
</head>
<body>
  <h1>Freesound Threshold Review</h1>
  <p>Records reviewed: <strong>{len(records):,}</strong></p>
  <p>Filter scope: <strong>{html.escape(scope_label)}</strong></p>
  <p>This report shows the clips closest to each threshold from both sides, so you can listen and decide whether to tighten the class list or move the threshold.</p>
  {''.join(sections)}
</body>
</html>
"""

    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write(html_content)

    print(f"Wrote HTML review report to {output_path}")


def export_exact_labels(
    export_path: Path,
    label_counts: Counter[str],
    label_max_scores: dict[str, float],
    min_count: int,
    min_max_score: float,
) -> None:
    """Write a candidate exact-label file for embedding-time filtering."""
    selected_labels = [
        label
        for label, count in label_counts.most_common()
        if count >= min_count and label_max_scores.get(label, 0.0) >= min_max_score
    ]

    with open(export_path, "w", encoding="utf-8") as handle:
        handle.write("# Exact matched model labels for embedding-time filtering.\n")
        handle.write(
            "# Generated from analyze_tag_results.py; edit this list to taste.\n\n"
        )
        for label in selected_labels:
            handle.write(f"{label}\n")

    print(
        f"Wrote {len(selected_labels)} labels to {export_path} "
        f"(min_count={min_count}, min_max_score={min_max_score:.2f})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect saved Freesound tagger results and compare thresholds."
    )
    parser.add_argument(
        "--tag-results-jsonl",
        type=Path,
        default=DEFAULT_TAG_RESULTS_JSONL,
        help="Path to the saved tagger output JSONL.",
    )
    parser.add_argument(
        "--filter-labels-file",
        type=str,
        default=None,
        help="Optional file with exact matched labels to analyze.",
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=DEFAULT_THRESHOLDS,
        help="Thresholds to compare.",
    )
    parser.add_argument(
        "--top-labels",
        type=int,
        default=15,
        help="How many top labels to print in summaries.",
    )
    parser.add_argument(
        "--inspect-threshold",
        type=float,
        default=None,
        help="If set, print borderline kept/rejected examples near this threshold.",
    )
    parser.add_argument(
        "--inspect-limit",
        type=int,
        default=10,
        help="How many kept/rejected borderline examples to show.",
    )
    parser.add_argument(
        "--export-labels-file",
        type=Path,
        default=None,
        help="Optional output path for a candidate exact-label file.",
    )
    parser.add_argument(
        "--export-min-count",
        type=int,
        default=25,
        help="Minimum clip count for an exact label to be exported.",
    )
    parser.add_argument(
        "--export-min-max-score",
        type=float,
        default=0.25,
        help="Minimum observed max score for an exact label to be exported.",
    )
    parser.add_argument(
        "--html-report",
        type=Path,
        default=None,
        help="Optional HTML report path for listening/reviewing borderline clips.",
    )

    args = parser.parse_args()

    filter_labels = load_filter_labels(args.filter_labels_file)
    records = load_tag_records(args.tag_results_jsonl)

    label_counts, label_max_scores = print_label_summary(
        records,
        filter_labels=filter_labels,
        top_labels=args.top_labels,
    )

    print_threshold_summaries(
        records,
        thresholds=args.thresholds,
        filter_labels=filter_labels,
        top_labels=args.top_labels,
    )

    if args.inspect_threshold is not None:
        print_borderline_examples(
            records,
            threshold=args.inspect_threshold,
            filter_labels=filter_labels,
            limit=args.inspect_limit,
        )

    if args.export_labels_file is not None:
        export_exact_labels(
            args.export_labels_file,
            label_counts=label_counts,
            label_max_scores=label_max_scores,
            min_count=args.export_min_count,
            min_max_score=args.export_min_max_score,
        )

    if args.html_report is not None:
        write_html_report(
            output_path=args.html_report,
            records=records,
            thresholds=args.thresholds,
            filter_labels=filter_labels,
            inspect_limit=args.inspect_limit,
            top_labels=args.top_labels,
        )


if __name__ == "__main__":
    main()
