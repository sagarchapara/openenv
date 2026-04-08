import unittest

from data.labeling import heuristic_label, label_samples
from data.schema import NormalizedSample


class LabelingPipelineSmokeTest(unittest.TestCase):
    def test_heuristic_label_short_sample_is_easy(self) -> None:
        sample = NormalizedSample(
            context="A short encyclopedic paragraph about a city and its location.",
            question="Where is the city located?",
            answer_list=["Warsaw"],
            source_dataset="local_fallback",
            source_type="encyclopedic_passage",
            category="geography",
        )
        self.assertEqual(heuristic_label(sample), "easy")

    def test_label_samples_attaches_difficulty_label(self) -> None:
        sample = NormalizedSample(
            context="A" * 3000,
            question="What percentage improvement was reported?",
            answer_list=["60%"],
            source_dataset="allenai/qasper",
            source_type="scientific_paper",
            category="scientific_research",
        )
        labeled = label_samples([sample])
        self.assertEqual(len(labeled), 1)
        self.assertIn("difficulty_label", labeled[0]["metadata"])


if __name__ == "__main__":
    unittest.main()
