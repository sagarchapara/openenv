"""Canonical sample schema used by the offline data pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass(slots=True)
class NormalizedSample:
    """Unified sample representation across datasets and domains."""

    context: str
    question: str
    answer_list: List[str]
    source_dataset: str
    source_type: str
    category: str = "general"
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def answer(self) -> str:
        return self.answer_list[0] if self.answer_list else ""
