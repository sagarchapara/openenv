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

    # Professional personas to vary the instructions
    PERSONAS = {
        "analyst": (
            "You are a Senior Research Analyst. Your goal is to create concise, "
            "fact-heavy briefings that prioritize precision and numerical accuracy."
        ),
        "editor": (
            "You are a Technical Editor. Your goal is to produce information-dense "
            "summaries that are clear, structured, and preserve the document's logical flow."
        ),
        "archivist": (
            "You are a Legal Archivist. Your goal is to document every specific name, "
            "date, and claim with maximum fidelity for historical record-keeping."
        ),
    }

    @abstractmethod
    def get_sample(self, seed: Optional[int] = None) -> Dict[str, Any]:
        """Return a single episode sample.

        Returns a dict with:
          - context: str              Full context text
          - truncated_context: str    Visible portion of context
          - truncation_ratio: float   Fraction shown (e.g. 0.7)
          - category: str             (Optional) Document category (e.g. 'History')
          - question: str             The question to answer
          - answer: str               Primary ground-truth answer
          - answer_list: list[str]    All valid answers (for F1 scoring)
        """

    def get_system_prompt(self, persona: str = "analyst") -> str:
        prompt_prefix = self.PERSONAS.get(persona, self.PERSONAS["analyst"])
        return (
            f"{prompt_prefix} "
            "Your summaries must be information-dense and preserve all key facts "
            "needed to answer subsequent factual questions about the document."
        )

    def get_summarize_prompt(self, truncated_context: str, truncation_ratio: float) -> str:
        pct = int(truncation_ratio * 100)
        return (
            f"Here is a document segment (representing {pct}% of the full text):\n\n"
            f"{truncated_context}\n\n"
            "Please provide a dense summary of the key information. Extract all "
            "specific details (names, dates, metrics, causal chains) essential "
            "for accurate factual question answering."
        )

    def get_answer_prompt(self, question: str) -> str:
        return (
            f"Based on your summary of the document, please answer the following question:\n\n"
            f"Question: {question}\n\n"
            "Provide a direct and concise answer. "
            "If the answer is a specific name, number, or phrase, give just that."
        )
