"""Abstract base class for summarization tasks."""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class BaseSummarizationTask(ABC):
    """Base class for all summarization tasks.

    Each task loads a dataset, samples an example, and constructs
    the episode data (truncated context, question, ground truth answers).
    """

    name: str = "base"
    max_steps: int = 2  # default: summarize + answer

    @abstractmethod
    def get_sample(self, seed: Optional[int] = None) -> Dict[str, Any]:
        """Return a single episode sample.

        Returns a dict with:
          - context: str              Full context text
          - truncated_context: str    Visible portion of context
          - truncation_ratio: float   Fraction shown (e.g. 0.7)
          - question: str             The question to answer
          - answer: str               Primary ground-truth answer
          - answer_list: list[str]    All valid answers (for F1 scoring)
        """

    def get_system_prompt(self) -> str:
        return (
            "You are an expert at analyzing and summarizing long documents. "
            "Your goal is to create concise but information-dense summaries "
            "that preserve all key facts needed to answer questions about the document."
        )

    def get_summarize_prompt(self, truncated_context: str, truncation_ratio: float) -> str:
        pct = int(truncation_ratio * 100)
        return (
            f"Here is a document excerpt (you are seeing approximately {pct}% of the full text):\n\n"
            f"{truncated_context}\n\n"
            "Please provide a concise summary of the key information in this excerpt. "
            "Focus on specific facts, names, dates, quantities, and relationships "
            "that might be needed to answer questions about this document."
        )

    def get_answer_prompt(self, question: str) -> str:
        return (
            f"Based on your summary of the document, please answer the following question:\n\n"
            f"Question: {question}\n\n"
            "Provide a direct and concise answer. "
            "If the answer is a specific name, number, or phrase, give just that."
        )
