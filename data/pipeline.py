"""Offline dataset normalization pipeline.

This module is present so the benchmark can grow into a multi-dataset curation
workflow. The live runtime does not depend on it yet.
"""

from __future__ import annotations

import os
from typing import Callable, Dict, Iterable, List

from .schema import NormalizedSample

SUPPORTED_SOURCES: Dict[str, List[str]] = {
    "easy": ["squad", "quality", "longbench_v2"],
    "medium": ["squad_long", "quality", "longbench_v2", "novelqa", "loong"],
    "hard": ["qasper", "peerqa", "longbench_v2", "loong", "ruler"],
}

DEFAULT_SOURCES: Dict[str, List[str]] = {
    "easy": ["squad"],
    "medium": ["squad_long"],
    "hard": ["qasper"],
}


def configured_sources(task_name: str) -> List[str]:
    """Return configured or default source names for a task bucket."""
    env_key = f"OPENENV_{task_name.upper()}_SOURCES"
    raw = os.getenv(env_key, "")
    if not raw.strip():
        return list(DEFAULT_SOURCES.get(task_name, []))
    allowed = set(SUPPORTED_SOURCES.get(task_name, []))
    selected = [
        part.strip().lower().replace("-", "_")
        for part in raw.split(",")
        if part.strip()
    ]
    return [name for name in selected if name in allowed] or list(DEFAULT_SOURCES.get(task_name, []))


def fallback_to_normalized_samples(
    fallback_samples: Iterable[dict],
    *,
    infer_category: Callable[[str], str],
    fallback_source_type: str,
) -> List[NormalizedSample]:
    """Normalize the repo's existing hardcoded fallback samples."""
    normalized: List[NormalizedSample] = []
    for item in fallback_samples:
        question = item["question"]
        normalized.append(
            NormalizedSample(
                context=item["context"],
                question=question,
                answer_list=list(item["answer_list"]),
                source_dataset="local_fallback",
                source_type=fallback_source_type,
                category=infer_category(question),
                metadata={"pipeline_origin": "fallback_only"},
            )
        )
    return normalized


def load_task_samples(
    *,
    task_name: str,
    infer_category: Callable[[str], str],
    fallback_samples: Iterable[dict],
    fallback_source_type: str,
) -> List[dict]:
    """Return normalized samples for offline analysis or future task wiring.

    For now this intentionally returns the repo's deterministic fallback-backed
    representation so the benchmark runtime remains stable.
    """
    return [
        {
            "context": sample.context,
            "question": sample.question,
            "answer": sample.answer,
            "answer_list": sample.answer_list,
            "category": sample.category,
            "source_type": sample.source_type,
            "source_dataset": sample.source_dataset,
            "metadata": sample.metadata,
        }
        for sample in fallback_to_normalized_samples(
            fallback_samples,
            infer_category=infer_category,
            fallback_source_type=fallback_source_type,
        )
    ]
