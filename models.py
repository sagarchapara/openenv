"""Pydantic models for the Long-Context Summarization environment."""
from typing import Optional, List, Dict, Any
from pydantic import Field

from openenv.core.env_server.types import Action, Observation, State


class SummarizationAction(Action):
    """Action containing the model's text response (summary or answer)."""

    response: str = Field(..., description="The model's text response")


class SummarizationObservation(Observation):
    """Observation containing conversation messages and episode metadata."""

    messages: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Conversation messages in OpenAI chat format",
    )
    step_type: str = Field(
        default="summarize",
        description="Current step: 'summarize', 'update_summary', 'answer', or 'done'",
    )
    task_name: str = Field(default="easy", description="Task difficulty: easy/medium/hard")
    context_length: int = Field(default=0, description="Total context length in characters")
    truncation_ratio: float = Field(
        default=0.7, description="Fraction of context shown to the model"
    )


class SummarizationState(State):
    """Internal state tracking episode metadata."""

    task_name: str = Field(default="easy")
    step_type: str = Field(default="summarize")
    context_length: int = Field(default=0)
    question: Optional[str] = Field(default=None)
