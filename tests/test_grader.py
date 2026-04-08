import unittest

from graders.qa_grader import compute_reward, normalize_answer


class GraderSmokeTest(unittest.TestCase):
    def test_numeric_normalization_handles_commas_and_percent(self) -> None:
        self.assertEqual(normalize_answer("384,400 km"), normalize_answer("384400 km"))
        self.assertEqual(normalize_answer("60 percent"), normalize_answer("60%"))

    def test_reward_is_capped_and_bounded(self) -> None:
        reward = compute_reward(
            predicted="60 percent",
            ground_truth_list=["60%"],
            summary="Short summary with the key metric.",
            task_name="hard",
        )
        self.assertGreaterEqual(reward, 0.0)
        self.assertLessEqual(reward, 1.0)
        self.assertGreater(reward, 0.5)


if __name__ == "__main__":
    unittest.main()
