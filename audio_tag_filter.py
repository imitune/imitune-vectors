#!/usr/bin/env python3
"""Model-based audio tag filtering helpers for Freesound subsets."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import torch
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

DEFAULT_TAGGER_MODEL_ID = "MIT/ast-finetuned-audioset-16-16-0.442"
DEFAULT_BLOCKED_LABELS_PATH = Path(__file__).with_name("blocked_model_labels_v1.txt")
DEFAULT_TAGGER_THRESHOLD = 0.25
DEFAULT_TAGGER_BATCH_SIZE = 16
MAX_AUDIT_EXAMPLES = 100
TOP_K_PREDICTIONS = 5


def normalize_label(label: str) -> str:
    """Normalize labels so model output names match denylist terms reliably."""
    cleaned = label.strip().lower().replace("_", " ")
    cleaned = re.sub(r"[^a-z0-9\s]+", " ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def load_blocked_labels(blocked_labels_file: str | Path | None = None) -> set[str]:
    """Load normalized denylist terms for matching model output labels."""
    path = (
        Path(blocked_labels_file).expanduser()
        if blocked_labels_file is not None
        else DEFAULT_BLOCKED_LABELS_PATH
    )
    if not path.exists():
        raise FileNotFoundError(f"Blocked labels file not found: {path}")

    blocked_labels: set[str] = set()
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            normalized = normalize_label(stripped)
            if normalized:
                blocked_labels.add(normalized)

    if not blocked_labels:
        raise ValueError(f"Blocked labels file is empty: {path}")

    return blocked_labels


def _detect_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _matches_blocked_term(normalized_label: str, blocked_terms: set[str]) -> bool:
    return any(
        blocked_term in normalized_label or normalized_label in blocked_term
        for blocked_term in blocked_terms
    )


@dataclass(frozen=True)
class AudioPrediction:
    label: str
    score: float


@dataclass(frozen=True)
class AudioTagFilterResult:
    top_predictions: list[AudioPrediction]
    blocked_predictions: list[AudioPrediction]

    @property
    def is_blocked(self) -> bool:
        return bool(self.blocked_predictions)


class AudioTagFilter:
    """Run clipwise multi-label audio tagging and expose block decisions."""

    def __init__(
        self,
        model_id: str = DEFAULT_TAGGER_MODEL_ID,
        blocked_terms: set[str] | None = None,
        threshold: float = DEFAULT_TAGGER_THRESHOLD,
        batch_size: int = DEFAULT_TAGGER_BATCH_SIZE,
        device: str | None = None,
    ) -> None:
        if not 0 < threshold < 1:
            raise ValueError("Tagger threshold must be between 0 and 1")
        if batch_size <= 0:
            raise ValueError("Tagger batch size must be > 0")

        self.model_id = model_id
        self.threshold = threshold
        self.batch_size = batch_size
        self.blocked_terms = blocked_terms or set()
        self.device = torch.device(device) if device else _detect_device()

        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
        self.model = AutoModelForAudioClassification.from_pretrained(model_id)
        self.model.to(self.device)
        self.model.eval()

        self.sample_rate = int(getattr(self.feature_extractor, "sampling_rate", 16000))
        self.id2label = {
            int(index): label for index, label in self.model.config.id2label.items()
        }
        self.normalized_id2label = {
            index: normalize_label(label) for index, label in self.id2label.items()
        }
        self.blocked_model_labels = sorted(
            {
                label
                for index, label in self.id2label.items()
                if _matches_blocked_term(
                    self.normalized_id2label[index],
                    self.blocked_terms,
                )
            }
        )

        if not self.blocked_model_labels:
            raise ValueError(
                "Blocked label terms did not match any model output labels. "
                "Update the denylist or choose a different tagging model."
            )

    def predict_batch(
        self,
        waveforms: list[np.ndarray],
        sampling_rate: int,
    ) -> list[AudioTagFilterResult]:
        """Return block decisions for a batch of preprocessed waveforms."""
        if not waveforms:
            return []

        if sampling_rate != self.sample_rate:
            waveforms = [
                librosa.resample(
                    waveform,
                    orig_sr=sampling_rate,
                    target_sr=self.sample_rate,
                )
                for waveform in waveforms
            ]

        inputs = self.feature_extractor(
            waveforms,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=True,
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.inference_mode():
            logits = self.model(**inputs).logits
            probabilities = torch.sigmoid(logits).detach().cpu().numpy()

        results: list[AudioTagFilterResult] = []
        for row in probabilities:
            top_indices = np.argsort(row)[-TOP_K_PREDICTIONS:][::-1]
            top_predictions = [
                AudioPrediction(label=self.id2label[index], score=float(row[index]))
                for index in top_indices
            ]

            blocked_indices = [
                index
                for index, score in enumerate(row)
                if score >= self.threshold
                and _matches_blocked_term(
                    self.normalized_id2label[index],
                    self.blocked_terms,
                )
            ]
            blocked_indices.sort(key=lambda index: row[index], reverse=True)
            blocked_predictions = [
                AudioPrediction(label=self.id2label[index], score=float(row[index]))
                for index in blocked_indices
            ]

            results.append(
                AudioTagFilterResult(
                    top_predictions=top_predictions,
                    blocked_predictions=blocked_predictions,
                )
            )

        return results
