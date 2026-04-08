"""Difficulty labeling pipeline for offline benchmark curation.

This module is intentionally not wired into the runtime task loaders yet.
It exists so we can curate datasets with either heuristic or LLM-based labels
without introducing deployment risk.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Iterable, List, Literal

from openai import OpenAI

from .schema import NormalizedSample

DifficultyLabel = Literal["easy", "medium", "hard"]


def _env_enabled(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def labeling_pipeline_enabled() -> bool:
    return _env_enabled("OPENENV_ENABLE_LABELING_PIPELINE")


def llm_labeler_enabled() -> bool:
    return _env_enabled("OPENENV_USE_LLM_LABELER")


def heuristic_label(sample: NormalizedSample) -> DifficultyLabel:
    context_len = len(sample.context)
    source_type = sample.source_type.lower()
    source_dataset = sample.source_dataset.lower()

    if (
        context_len >= 2200
        or "scientific" in source_type
        or "peer" in source_type
        or "ruler" in source_dataset
    ):
        return "hard"
    if (
        context_len >= 900
        or "long" in source_type
        or "multi" in source_type
        or "quality" in source_dataset
    ):
        return "medium"
    return "easy"


@lru_cache(maxsize=1)
def _labeler_client() -> OpenAI:
    return OpenAI(
        base_url=os.getenv("API_BASE_URL", "https://router.huggingface.co/v1"),
        api_key=os.getenv("HF_TOKEN") or os.getenv("API_KEY") or "EMPTY",
    )


def llm_label(sample: NormalizedSample) -> DifficultyLabel:
    client = _labeler_client()
    model = os.getenv("OPENENV_LABELER_MODEL") or os.getenv(
        "MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct"
    )
    prompt = (
        "Assign one difficulty label to this long-context QA sample: easy, medium, or hard.\n"
        "Consider context length, technicality, number of answer-critical facts, and whether\n"
        "the sample would challenge a frontier model under summarization-based compression.\n\n"
        f"Source dataset: {sample.source_dataset}\n"
        f"Source type: {sample.source_type}\n"
        f"Category: {sample.category}\n"
        f"Context length: {len(sample.context)}\n"
        f"Question: {sample.question}\n"
        f"Context preview: {sample.context[:900]}\n\n"
        'Return JSON only: {"difficulty": "easy|medium|hard"}'
    )
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a benchmark curator. Return strict JSON only."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=60,
        stream=False,
        response_format={"type": "json_object"},
    )
    payload = json.loads(response.choices[0].message.content or "{}")
    difficulty = payload.get("difficulty")
    if difficulty not in {"easy", "medium", "hard"}:
        return heuristic_label(sample)
    return difficulty


def label_sample(sample: NormalizedSample) -> DifficultyLabel:
    if llm_labeler_enabled():
        try:
            return llm_label(sample)
        except Exception:
            return heuristic_label(sample)
    return heuristic_label(sample)


def label_samples(samples: Iterable[NormalizedSample]) -> List[dict]:
    """Return labeled samples for offline inspection or artifact generation."""
    labeled: List[dict] = []
    for sample in samples:
        labeled.append(
            {
                "context": sample.context,
                "question": sample.question,
                "answer_list": list(sample.answer_list),
                "source_dataset": sample.source_dataset,
                "source_type": sample.source_type,
                "category": sample.category,
                "metadata": {**sample.metadata, "difficulty_label": label_sample(sample)},
            }
        )
    return labeled
