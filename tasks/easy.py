"""Easy task: SQuAD v1 short passages with factual QA.

Context: 300–700 characters, truncated to 70%.
The answer always appears in the truncated portion.
Episode: 2 steps (summarize → answer).
"""
import random
import logging
from typing import Dict, Any, List, Optional

from .base import BaseSummarizationTask

logger = logging.getLogger(__name__)

# Hardcoded fallback samples (used if datasets library is unavailable)
FALLBACK_SAMPLES: List[Dict[str, Any]] = [
    {
        "context": (
            "The Amazon rainforest, also known in English as Amazonia, is a moist broadleaf "
            "tropical rainforest in the Amazon biome that covers most of the Amazon basin of "
            "South America. This basin encompasses 7,000,000 km2 (2,700,000 sq mi), of which "
            "5,500,000 km2 (2,100,000 sq mi) are covered by the rainforest. This region includes "
            "territory belonging to nine nations and 3,344 formally acknowledged indigenous territories."
        ),
        "question": "How many nations have territory in the Amazon rainforest region?",
        "answer_list": ["nine", "9"],
    },
    {
        "context": (
            "The Great Wall of China is a series of fortifications that were built across the "
            "historical northern borders of ancient Chinese states and Imperial China as protection "
            "against various nomadic groups from the Eurasian Steppe. Several walls were being built "
            "as early as the 7th century BCE by ancient Chinese states. Selective stretches were "
            "later joined together by Qin Shi Huang (220–206 BCE), the first emperor of China."
        ),
        "question": "Who was the first emperor of China who joined the wall sections?",
        "answer_list": ["Qin Shi Huang"],
    },
    {
        "context": (
            "Photosynthesis is a process used by plants and other organisms to convert light energy "
            "into chemical energy that, through cellular respiration, can later be released to fuel "
            "the organism's activities. Some of this chemical energy is stored in carbohydrate "
            "molecules, such as sugars and starches, which are synthesized from carbon dioxide and "
            "water – hence the name photosynthesis, from the Greek phōs (light), and synthesis "
            "(putting together)."
        ),
        "question": "What molecules are synthesized from carbon dioxide and water in photosynthesis?",
        "answer_list": ["sugars and starches", "carbohydrate molecules", "carbohydrates"],
    },
    {
        "context": (
            "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. "
            "It is named after the engineer Gustave Eiffel, whose company designed and built the tower "
            "from 1887 to 1889 as the centerpiece of the 1889 World's Fair. Although initially "
            "criticised by some of France's leading artists and intellectuals for its design, it has "
            "become a global cultural icon of France and one of the most recognisable structures in the world."
        ),
        "question": "After whom is the Eiffel Tower named?",
        "answer_list": ["Gustave Eiffel", "the engineer Gustave Eiffel"],
    },
    {
        "context": (
            "DNA, or deoxyribonucleic acid, is a polymer composed of two polynucleotide chains that "
            "coil around each other to form a double helix. The polymer carries genetic instructions "
            "for the development, functioning, growth and reproduction of all known organisms and many "
            "viruses. DNA and ribonucleic acid (RNA) are nucleic acids. Alongside proteins, lipids and "
            "complex carbohydrates (polysaccharides), nucleic acids are one of the four major types of "
            "macromolecules that are essential for all known forms of life."
        ),
        "question": "What shape do the two polynucleotide chains form in DNA?",
        "answer_list": ["double helix", "a double helix"],
    },
    {
        "context": (
            "The Python programming language was conceived in the late 1980s by Guido van Rossum at "
            "Centrum Wiskunde & Informatica in the Netherlands as a successor to the ABC language. "
            "Python 2.0, released in 2000, introduced new features such as list comprehensions and "
            "a garbage collection system capable of collecting reference cycles. Python 3.0, released "
            "in 2008, was a major revision not completely backward-compatible with earlier versions."
        ),
        "question": "Who created the Python programming language?",
        "answer_list": ["Guido van Rossum"],
    },
    {
        "context": (
            "The Moon is Earth's only natural satellite. It orbits at an average distance of 384,400 km "
            "(238,900 mi), or about 30 times Earth's diameter. The Moon's gravitational influence "
            "is the main driver of Earth's tides and very slowly lengthening Earth's day. "
            "The Moon's current orbital distance makes it appear nearly the same size in the sky as "
            "the Sun, allowing it to cover the Sun almost precisely in total solar eclipses."
        ),
        "question": "What is the average orbital distance of the Moon from Earth?",
        "answer_list": ["384,400 km", "384,400 km (238,900 mi)", "238,900 mi"],
    },
    {
        "context": (
            "The Roman Empire was the post-Republican state of ancient Rome. It included large "
            "territorial holdings around the Mediterranean Sea in Europe, North Africa, and Western "
            "Asia. It was ruled by emperors. Julius Caesar's adopted son Augustus became the first "
            "Roman emperor in 27 BCE. The Roman Empire lasted until 476 CE when Odoacer deposed "
            "the last emperor, Romulus Augustulus."
        ),
        "question": "Who became the first Roman emperor?",
        "answer_list": ["Augustus", "Julius Caesar's adopted son Augustus"],
    },
]

TRUNCATION_RATIO = 0.70  # Show 70% of context


class EasyTask(BaseSummarizationTask):
    """Short-context factual QA task using SQuAD v1."""

    name = "easy"
    max_steps = 2

    def __init__(self):
        self._samples: List[Dict[str, Any]] = []
        self._load_dataset()

    def _load_dataset(self):
        """Load SQuAD samples where the answer is in the first 70% of context."""
        try:
            from datasets import load_dataset

            logger.info("Loading SQuAD dataset for easy task...")
            ds = load_dataset("rajpurkar/squad", split="validation", trust_remote_code=False)

            target_min, target_max = 300, 900  # chars
            for item in ds:
                context: str = item["context"]
                if not (target_min <= len(context) <= target_max):
                    continue

                answers = item["answers"]["text"]
                answer_starts = item["answers"]["answer_start"]
                cutoff = int(len(context) * TRUNCATION_RATIO)

                # Only include examples where ALL answer spans are before cutoff
                if not answers or not all(s < cutoff for s in answer_starts):
                    continue

                self._samples.append(
                    {
                        "context": context,
                        "question": item["question"],
                        "answer_list": list(set(answers)),
                    }
                )

                if len(self._samples) >= 500:
                    break

            logger.info(f"Easy task: loaded {len(self._samples)} SQuAD samples")
        except Exception as e:
            logger.warning(f"Could not load SQuAD dataset: {e}. Using fallback samples.")

        if not self._samples:
            self._samples = FALLBACK_SAMPLES

    def get_sample(self, seed: Optional[int] = None) -> Dict[str, Any]:
        rng = random.Random(seed)
        item = rng.choice(self._samples)

        context = item["context"]
        
        # Dynamic truncation (65% to 75%)
        ratio = rng.uniform(0.65, 0.75)
        cutoff = int(len(context) * ratio)

        # Basic categorization
        q = item["question"].lower()
        if any(w in q for w in ["who", "born", "king", "queen", "empire", "war"]):
            cat = "History"
        elif any(w in q for w in ["what is", "process", "science", "chemical", "atom", "cell"]):
            cat = "Science"
        elif any(w in q for w in ["where", "city", "country", "river", "mountain"]):
            cat = "Geography"
        else:
            cat = "General"

        return {
            "context": context,
            "truncated_context": context[:cutoff],
            "truncation_ratio": ratio,
            "category": cat,
            "question": item["question"],
            "answer": item["answer_list"][0],
            "answer_list": item["answer_list"],
        }
