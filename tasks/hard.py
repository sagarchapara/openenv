"""Hard task: Long scientific paper QA using QASPER dataset.

Context: 3000–10000 characters, split into TWO chunks.
Episode: 3 steps — summarize chunk 1 → update summary with chunk 2 → answer.
This mirrors the chained-summarization approach described in Cursor's blog:
each summary is carried forward and updated, testing whether key information
survives multiple compression rounds.

Grading: token-level F1 score.
"""
import random
import logging
from typing import Dict, Any, List, Optional

from .base import BaseSummarizationTask

logger = logging.getLogger(__name__)

FALLBACK_SAMPLES: List[Dict[str, Any]] = [
    {
        "context": (
            "Abstract: This paper presents a comprehensive study of transformer-based language "
            "models and their ability to handle long-context understanding tasks. We evaluate "
            "several state-of-the-art models on the SCROLLS benchmark, which comprises seven "
            "challenging long-document tasks requiring understanding of full books, scientific "
            "papers, and meeting transcripts.\n\n"
            "Introduction: The ability to process and understand long documents remains one of "
            "the fundamental challenges in natural language processing. While transformer models "
            "have achieved remarkable performance on many NLP benchmarks, their performance "
            "degrades significantly as context length increases beyond their training window. "
            "This degradation is particularly pronounced for tasks requiring integration of "
            "information from distant parts of a document.\n\n"
            "Our experiments show that models trained with extended context windows achieve "
            "significantly better performance on long-document tasks compared to models with "
            "standard 2048-token windows. Specifically, we find that a model with 16K token "
            "context achieves 43.2% F1 on the SCROLLS benchmark, compared to 31.8% for the "
            "same model with a 2048-token context window.\n\n"
            "Methods: We fine-tune the base model using a combination of summarization and "
            "question-answering objectives on documents up to 16,384 tokens. During training, "
            "we apply a sliding window attention mechanism to handle documents that exceed "
            "the model's context window. The sliding window has a stride of 512 tokens and "
            "a window size of 4096 tokens, allowing the model to process arbitrarily long documents.\n\n"
            "Results: Our model achieves state-of-the-art performance on 5 out of 7 SCROLLS tasks. "
            "On the NarrativeQA task, which requires answering questions about full novels, "
            "our model achieves 28.4% F1, a 6.2 point improvement over the previous best. "
            "On GovReport, a summarization task using long government reports, our model achieves "
            "a ROUGE-L score of 34.8, compared to 29.3 for the baseline."
        ),
        "question": "What F1 score did the model with 16K token context achieve on SCROLLS?",
        "answer_list": ["43.2%", "43.2% F1", "43.2"],
    },
    {
        "context": (
            "Abstract: We study the problem of efficient training of large language models (LLMs) "
            "with limited computational resources. Our main contribution is a novel gradient "
            "checkpointing strategy that reduces memory consumption by 60% while adding only "
            "15% overhead to training time. We validate our approach on models ranging from "
            "1B to 70B parameters.\n\n"
            "Background: Training LLMs requires enormous computational resources. A 70B parameter "
            "model requires approximately 140GB of memory just to store the parameters in 16-bit "
            "precision, before accounting for optimizer states and activations. Gradient "
            "checkpointing, first introduced by Chen et al. (2016), trades computation for memory "
            "by recomputing activations during the backward pass instead of storing them.\n\n"
            "Our Approach: We propose Selective Gradient Checkpointing (SGC), which identifies "
            "the optimal subset of layers to checkpoint based on their memory-to-computation ratio. "
            "Unlike standard checkpointing which applies uniformly to all layers, SGC focuses "
            "computational overhead on layers where memory savings are greatest.\n\n"
            "Experiments: We evaluate SGC on GPT-3 style architectures with 1B, 7B, 13B, 30B, "
            "and 70B parameters. For the 70B model, standard training requires 1120GB of GPU memory "
            "across 8 A100s. Our SGC approach reduces this to 448GB, enabling training on just "
            "4 A100 GPUs. The training throughput with SGC is 2,150 tokens/second compared to "
            "2,450 tokens/second without checkpointing, representing only a 12% slowdown.\n\n"
            "Conclusion: SGC enables training of 70B parameter models on 4x fewer GPUs compared "
            "to standard approaches. This democratizes access to large-scale model training and "
            "opens the door for smaller research groups to train frontier-scale models."
        ),
        "question": "By what percentage does SGC reduce memory consumption?",
        "answer_list": ["60%", "60 percent"],
    },
    {
        "context": (
            "Abstract: This work presents MedLLM, a large language model specifically pre-trained "
            "on medical literature, clinical notes, and biomedical databases. MedLLM achieves "
            "state-of-the-art performance on medical question answering benchmarks, including "
            "MedQA, MedMCQA, and PubMedQA. The model was trained on 200 billion tokens of "
            "medical text and contains 7 billion parameters.\n\n"
            "Introduction: Medical knowledge is highly specialized and constantly evolving. "
            "General-purpose language models, despite their impressive capabilities, often "
            "struggle with nuanced medical questions that require deep domain expertise. "
            "Clinical decision support systems require not just factual recall but also "
            "the ability to reason about patient symptoms, treatment options, and drug interactions.\n\n"
            "Training Data: MedLLM was pre-trained on a curated corpus of 200B tokens including: "
            "PubMed abstracts (45B tokens), full-text medical journals (80B tokens), clinical "
            "guidelines from major medical associations (15B tokens), de-identified clinical "
            "notes (40B tokens), and biomedical knowledge bases including DrugBank and OMIM (20B tokens).\n\n"
            "Evaluation: On MedQA (USMLE-style questions), MedLLM achieves 72.4% accuracy, "
            "compared to 57.6% for GPT-3.5 and 67.2% for Med-PaLM. On PubMedQA, which requires "
            "reading scientific papers and answering yes/no/maybe questions, MedLLM achieves "
            "78.2% accuracy. The model particularly excels at questions requiring synthesis "
            "of information across multiple medical domains."
        ),
        "question": "How many tokens was MedLLM trained on?",
        "answer_list": ["200 billion tokens", "200B tokens", "200 billion"],
    },
]

TRUNCATION_RATIO = 0.55  # Show 55% total; split into two ~27.5% chunks


class HardTask(BaseSummarizationTask):
    """Long scientific paper QA with chained summarization (3 steps)."""

    name = "hard"
    max_steps = 3  # summarize chunk1 → update summary with chunk2 → answer

    def __init__(self):
        self._samples: List[Dict[str, Any]] = []
        self._load_dataset()

    def _load_dataset(self):
        """Load QASPER samples with long paper contexts."""
        try:
            from datasets import load_dataset

            logger.info("Loading QASPER dataset for hard task...")
            ds = load_dataset("allenai/qasper", split="validation", trust_remote_code=True)

            for item in ds:
                try:
                    context = self._build_context(item)
                    if len(context) < 2000:
                        continue

                    for qa in item.get("qas", []):
                        question = qa.get("question", "").strip()
                        if not question:
                            continue

                        answer_list = self._extract_answers(qa)
                        if not answer_list:
                            continue

                        # Ensure answer is in the first 55% of context
                        cutoff = int(len(context) * TRUNCATION_RATIO)
                        answer_in_cutoff = any(
                            ans.lower() in context[:cutoff].lower() for ans in answer_list
                        )
                        if not answer_in_cutoff:
                            continue

                        self._samples.append(
                            {
                                "context": context,
                                "question": question,
                                "answer_list": answer_list,
                            }
                        )
                        break  # one QA pair per paper

                except Exception:
                    continue

                if len(self._samples) >= 200:
                    break

            logger.info(f"Hard task: loaded {len(self._samples)} QASPER samples")
        except Exception as e:
            logger.warning(f"Could not load QASPER dataset: {e}. Using fallback samples.")

        if not self._samples:
            self._samples = FALLBACK_SAMPLES

    def _build_context(self, item: Dict[str, Any]) -> str:
        """Concatenate abstract and main sections of a QASPER paper."""
        parts = []
        abstract = item.get("abstract", "").strip()
        if abstract:
            parts.append(f"Abstract: {abstract}")

        full_text = item.get("full_text", {})
        section_names = full_text.get("section_name", [])
        paragraphs = full_text.get("paragraphs", [])

        for sec_name, paras in zip(section_names, paragraphs):
            if not paras:
                continue
            sec_text = "\n".join(p for p in paras if p.strip())
            if sec_text:
                parts.append(f"\n{sec_name}:\n{sec_text}")

        return "\n\n".join(parts)

    def _extract_answers(self, qa: Dict[str, Any]) -> List[str]:
        """Extract answer strings from QASPER QA entry."""
        answers = []
        for ann in qa.get("answers", []):
            if ann.get("unanswerable", False):
                continue
            ann_type = ann.get("annotation_type", "")
            if ann_type == "extractive":
                spans = ann.get("answer", {}).get("extractive_spans", [])
                answers.extend([s.strip() for s in spans if s.strip()])
            elif ann_type == "abstractive":
                free = ann.get("answer", {}).get("free_response", "").strip()
                if free:
                    answers.append(free)
            elif ann_type == "boolean":
                yes_no = ann.get("answer", {}).get("yes_no")
                if yes_no is not None:
                    answers.append("yes" if yes_no else "no")
        return answers

    def get_sample(self, seed: Optional[int] = None) -> Dict[str, Any]:
        rng = random.Random(seed)
        item = rng.choice(self._samples)

        context = item["context"]
        total_len = len(context)

        # Split into two roughly equal chunks (each ~27.5% of full context)
        mid = total_len // 2
        chunk1 = context[:mid]
        chunk2 = context[mid : int(total_len * TRUNCATION_RATIO)]

        return {
            "context": context,
            "truncated_context": context[: int(total_len * TRUNCATION_RATIO)],
            "chunk1": chunk1,
            "chunk2": chunk2,
            "truncation_ratio": TRUNCATION_RATIO,
            "question": item["question"],
            "answer": item["answer_list"][0],
            "answer_list": item["answer_list"],
        }

    def get_summarize_prompt(self, truncated_context: str, truncation_ratio: float) -> str:
        """Prompt for the first chunk (chunk1)."""
        return (
            f"Here is the first section of a long scientific document:\n\n"
            f"{truncated_context}\n\n"
            "Please summarize the key information from this section. "
            "Focus on: research goals, methods, key findings, specific numbers/metrics, "
            "and any named entities. Keep your summary under 250 words."
        )

    def get_update_summary_prompt(self, chunk2: str) -> str:
        """Prompt for updating the summary with the second chunk."""
        return (
            f"Here is the next section of the same document:\n\n"
            f"{chunk2}\n\n"
            "Please update your previous summary to incorporate the key information "
            "from this section as well. Keep your combined summary under 300 words, "
            "preserving all important facts from both sections."
        )

    def get_answer_prompt(self, question: str) -> str:
        return (
            f"Based on your summary of the document, answer this question:\n\n"
            f"Question: {question}\n\n"
            "Give a direct, specific answer. If it's a number or name, provide exactly that."
        )
