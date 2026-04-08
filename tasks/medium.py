"""Medium task: SQuAD v1 longer passages with multiple-hop reasoning.

Context: 800–2000 characters, truncated to 65%.
The answer is always within the truncated portion.
Episode: 2 steps (summarize → answer).
Grading: token-level F1 score (same as SQuAD official eval).
"""
import random
import logging
from typing import Dict, Any, List, Optional

from .base import BaseSummarizationTask

logger = logging.getLogger(__name__)

FALLBACK_SAMPLES: List[Dict[str, Any]] = [
    {
        "context": (
            "The Byzantine Empire, also referred to as the Eastern Roman Empire or Byzantium, "
            "was the continuation of the Roman Empire primarily in its eastern provinces during "
            "Late Antiquity and the Middle Ages, when its capital city was Constantinople. It "
            "survived the fragmentation and fall of the Western Roman Empire in the 5th century "
            "AD and continued to exist for an additional thousand years until the fall of "
            "Constantinople to the Ottoman Empire in 1453. During most of its existence, the "
            "empire was the most powerful economic, cultural, and military force in Europe. "
            "Both the See of Constantinople and the Ecumenical Patriarchate, which are Christian "
            "institutions, trace their origins to the foundation of Constantinople by Constantine "
            "the Great in 330 AD. The empire's rich history, blending Greek, Roman, and Christian "
            "traditions, produced important developments in art, architecture, and philosophy "
            "that continue to influence Eastern Europe to this day."
        ),
        "question": "In what year did the Byzantine Empire fall to the Ottoman Empire?",
        "answer_list": ["1453"],
    },
    {
        "context": (
            "The Industrial Revolution was the transition to new manufacturing processes in Great "
            "Britain, continental Europe, and the United States, from about 1760 to sometime between "
            "1820 and 1840. This transition included going from hand production methods to machines; "
            "new chemical manufacturing and iron production processes; the increasing use of steam "
            "power and water power; the development of machine tools; and the rise of the mechanised "
            "factory system. Output greatly increased, and a result was an unprecedented rise in "
            "population and the rate of population growth. The textile industry was the first to use "
            "modern production methods, and textiles became the dominant industry in terms of "
            "employment, value of output, and capital invested. Cotton was the leading textile of "
            "the Industrial Revolution and assumed its dominant role because cotton could be cultivated "
            "at scale in warm climates outside Europe, especially in what became the southern United States."
        ),
        "question": "Which industry was the first to use modern production methods during the Industrial Revolution?",
        "answer_list": ["textile industry", "The textile industry", "textiles"],
    },
    {
        "context": (
            "Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to "
            "intelligence of humans and other animals. Example tasks in which this is done include "
            "speech recognition, computer vision, translation between (natural) languages, as well as "
            "other mappings of inputs. AI applications include advanced web search engines (e.g., "
            "Google Search), recommendation systems (used by YouTube, Amazon, and Netflix), "
            "understanding human speech (such as Siri and Alexa), self-driving cars (e.g., Waymo), "
            "generative or creative tools (ChatGPT and AI art), automated decision-making, and "
            "competing at the highest level in strategic game systems (such as chess and Go). As "
            "machines become increasingly capable, tasks considered to require 'intelligence' are "
            "often removed from the definition of AI, a phenomenon known as the AI effect. For "
            "instance, optical character recognition is frequently excluded from things considered "
            "to be AI, having become a routine technology. Artificial intelligence was founded as "
            "an academic discipline in 1956, and in the years since it has experienced several waves "
            "of optimism, followed by disappointment and the loss of funding (known as an 'AI winter'), "
            "followed by new approaches, success, and renewed funding."
        ),
        "question": "In what year was artificial intelligence founded as an academic discipline?",
        "answer_list": ["1956"],
    },
    {
        "context": (
            "The human brain is the command center for the human nervous system. It receives signals "
            "from the body's sensory organs and outputs information to the muscles. The human brain "
            "has the same basic structure as other mammal brains, but is larger in relation to body "
            "size than any other brains. The cerebral cortex is the outer layer of the brain and is "
            "responsible for most higher-order functions, including consciousness, memory, reasoning, "
            "and language. It is divided into four lobes: the frontal lobe, parietal lobe, temporal "
            "lobe, and occipital lobe. The brain communicates with the rest of the body through the "
            "spinal cord and a network of nerves, which together form the peripheral nervous system. "
            "The average adult human brain weighs about 1.4 kilograms (3 lb) and contains approximately "
            "86 billion neurons. These neurons are connected by trillions of synaptic connections, "
            "making the brain one of the most complex structures in the known universe."
        ),
        "question": "What is the outer layer of the brain called?",
        "answer_list": ["cerebral cortex", "The cerebral cortex"],
    },
    {
        "context": (
            "Climate change refers to long-term shifts in temperatures and weather patterns. Such "
            "shifts can be natural, due to changes in the sun's activity or large volcanic eruptions. "
            "But since the 1800s, human activities have been the main driver of climate change, "
            "primarily due to the burning of fossil fuels like coal, oil and gas. Burning fossil fuels "
            "generates greenhouse gas emissions that act like a blanket wrapped around the Earth, "
            "trapping the sun's heat and raising temperatures. The main greenhouse gases that are "
            "causing climate change include carbon dioxide and methane. These come from using gasoline "
            "for driving a car or coal for heating a building, for example. Clearing land and forests "
            "can also release carbon dioxide. Agriculture, oil and gas operations are major sources of "
            "methane emissions. Energy, industry, transport, buildings, agriculture and land use are "
            "among the main sectors causing greenhouse gas emissions. Between 1880 and 2012, the "
            "average global temperature increased by 0.85 degrees Celsius."
        ),
        "question": "What has been the main driver of climate change since the 1800s?",
        "answer_list": ["human activities", "burning of fossil fuels", "fossil fuels"],
    },
]

TRUNCATION_RATIO = 0.65  # Show 65% of context (harder than easy)


class MediumTask(BaseSummarizationTask):
    """Medium-length context factual QA task using longer SQuAD passages."""

    name = "medium"
    max_steps = 2

    def __init__(self):
        self._samples: List[Dict[str, Any]] = []
        self._load_dataset()

    def _load_dataset(self):
        """Load SQuAD samples with longer passages."""
        try:
            from datasets import load_dataset

            logger.info("Loading SQuAD dataset for medium task...")
            ds = load_dataset("rajpurkar/squad", split="validation", trust_remote_code=False)

            target_min, target_max = 900, 2500  # chars
            seen_contexts = set()

            for item in ds:
                context: str = item["context"]
                if not (target_min <= len(context) <= target_max):
                    continue
                if context in seen_contexts:
                    continue
                seen_contexts.add(context)

                answers = item["answers"]["text"]
                answer_starts = item["answers"]["answer_start"]
                cutoff = int(len(context) * TRUNCATION_RATIO)

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

            logger.info(f"Medium task: loaded {len(self._samples)} SQuAD samples")
        except Exception as e:
            logger.warning(f"Could not load SQuAD dataset for medium: {e}. Using fallback.")

        if not self._samples:
            self._samples = FALLBACK_SAMPLES

    def get_sample(self, seed: Optional[int] = None) -> Dict[str, Any]:
        rng = random.Random(seed)
        item = rng.choice(self._samples)

        context = item["context"]
        cutoff = int(len(context) * TRUNCATION_RATIO)
        category = self.infer_category(item["question"])

        return {
            "context": context,
            "truncated_context": context[:cutoff],
            "truncation_ratio": TRUNCATION_RATIO,
            "category": category,
            "source_type": "long_form_reference",
            "question": item["question"],
            "answer": item["answer_list"][0],
            "answer_list": item["answer_list"],
        }

    def get_summarize_prompt(self, truncated_context: str, truncation_ratio: float) -> str:
        pct = int(truncation_ratio * 100)
        return (
            f"Here is a document excerpt ({pct}% of the full text):\n\n"
            f"{truncated_context}\n\n"
            "Produce a retrieval-safe summary for a downstream assistant. Preserve "
            "specific names, dates, numbers, causal relationships, and claims that "
            "are likely to be queried later. Keep the summary under 200 words while "
            "retaining the details needed for factual QA."
        )
