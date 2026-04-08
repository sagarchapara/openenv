"""Optional LLM-based grading for summary quality.

This grader is disabled by default and only used when ``USE_LLM_GRADER=true``.
It is designed as an auxiliary signal on top of deterministic answer grading.
"""

from __future__ import annotations

import json
import os
from typing import List, Optional

from openai import OpenAI


def llm_grader_enabled() -> bool:
    return os.getenv("USE_LLM_GRADER", "").strip().lower() in {"1", "true", "yes", "on"}


def _build_client() -> OpenAI:
    api_base_url = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
    api_key = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or "EMPTY"
    return OpenAI(base_url=api_base_url, api_key=api_key)


def grade_summary_quality(
    *,
    summary: Optional[str],
    question: Optional[str],
    predicted: str,
    ground_truth_list: List[str],
    task_name: str,
) -> Optional[float]:
    """Return an LLM-judged summary quality score in [0.0, 1.0], or None on failure."""
    if not llm_grader_enabled() or not summary:
        return None

    model = os.getenv("LLM_GRADER_MODEL") or os.getenv(
        "MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct"
    )
    client = _build_client()

    system_prompt = (
        "You are grading the quality of a compressed working-memory summary for a downstream "
        "question-answering agent. Score only the summary quality, not whether the final answer "
        "happened to be correct. Reward summaries that preserve critical facts, quantities, and "
        "entities needed for answering the question. Penalize summaries that omit key details, "
        "hallucinate unsupported claims, or are unnecessarily verbose. Return only compact JSON."
    )

    user_prompt = (
        f"Task: {task_name}\n"
        f"Question: {question or 'N/A'}\n"
        f"Accepted answers: {ground_truth_list}\n"
        f"Model final answer: {predicted}\n"
        f"Summary to grade:\n{summary}\n\n"
        'Return JSON of the form {"score": <float 0.0-1.0>, "reason": "<short reason>"}.'
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=120,
            stream=False,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content or "{}"
        payload = json.loads(content)
        score = float(payload.get("score"))
        return min(max(score, 0.0), 1.0)
    except Exception:
        return None
