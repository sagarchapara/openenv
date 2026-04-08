import unittest

from server.environment import SummarizationEnvironment
from models import SummarizationAction


class EnvironmentSmokeTest(unittest.TestCase):
    def test_easy_episode_reaches_terminal_state_with_bounded_reward(self) -> None:
        env = SummarizationEnvironment()

        obs = env.reset(task_name="easy", seed=0)
        self.assertEqual(obs.step_type, "summarize")
        self.assertFalse(obs.done)
        self.assertIsNotNone(obs.category)
        self.assertIsNotNone(obs.source_type)

        obs = env.step(SummarizationAction(response="Compact factual summary."))
        self.assertEqual(obs.step_type, "answer")
        self.assertFalse(obs.done)

        answer = env.state.question or ""
        obs = env.step(SummarizationAction(response=answer))
        self.assertTrue(obs.done)
        self.assertGreaterEqual(obs.reward, 0.0)
        self.assertLessEqual(obs.reward, 1.0)

    def test_hard_episode_uses_update_summary_stage(self) -> None:
        env = SummarizationEnvironment()

        obs = env.reset(task_name="hard", seed=0)
        self.assertEqual(obs.step_type, "summarize")
        self.assertEqual(obs.source_type, "scientific_paper")

        obs = env.step(SummarizationAction(response="Initial summary."))
        self.assertEqual(obs.step_type, "update_summary")

        obs = env.step(SummarizationAction(response="Updated combined summary."))
        self.assertEqual(obs.step_type, "answer")


if __name__ == "__main__":
    unittest.main()
