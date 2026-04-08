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

    def infer_category(self, question: str) -> str:
        """Infer a coarse document category for richer benchmark metadata."""
        q = question.lower()
        if any(word in q for word in ["who", "born", "war", "empire", "king", "queen"]):
            return "history"
        if any(word in q for word in ["city", "country", "river", "mountain", "where"]):
            return "geography"
        if any(word in q for word in ["process", "chemical", "cell", "atom", "science"]):
            return "science"
        if any(word in q for word in ["programming", "software", "language", "python", "code"]):
            return "software"
        return "general"

    @abstractmethod
    def get_sample(self, seed: Optional[int] = None) -> Dict[str, Any]:
        """Return a single episode sample.

        Returns a dict with:
          - context: str              Full context text
          - truncated_context: str    Visible portion of context
          - truncation_ratio: float   Fraction shown (e.g. 0.7)
          - category: str             Coarse domain/category
          - source_type: str          Source style (encyclopedic, report, paper, etc.)
          - question: str             The question to answer
          - answer: str               Primary ground-truth answer
          - answer_list: list[str]    All valid answers (for F1 scoring)
        """

    def get_system_prompt(self) -> str:
        return (
            "You are preparing a compact working memory for another assistant that will "
            "not get to read the original document. Summaries must be concise, faithful, "
            "and dense with facts that are likely to matter for later question answering."
        )

    def get_summarize_prompt(self, truncated_context: str, truncation_ratio: float) -> str:
        pct = int(truncation_ratio * 100)
        return (
            f"Here is a document excerpt (you are seeing approximately {pct}% of the full text):\n\n"
            f"{truncated_context}\n\n"
            "Write a compact summary for downstream use. Preserve concrete details such as "
            "names, dates, quantities, entities, causal links, and definitions that a later "
            "assistant might need in order to answer factual questions without the source text."
        )

    def get_answer_prompt(self, question: str) -> str:
        return (
            f"Based on your summary of the document, please answer the following question:\n\n"
            f"Question: {question}\n\n"
            "Answer using only information that was preserved in the summary. "
            "Provide a direct, concise answer. If the answer is a specific name, number, or phrase, give just that."
        )
