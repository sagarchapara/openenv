"""Main SummarizationEnvironment — implements the OpenEnv Environment interface.

Episode flow per task:
  easy   (2 steps): truncated_context → summarize → question → answer
  medium (2 steps): longer truncated_context → summarize → question → answer
  hard   (3 steps): chunk1 → summarize → chunk2 → update_summary → question → answer

Reward: token-level F1 score, with a small conciseness bonus for compact summaries.
"""
import random
import sys
import os
import logging
from typing import Optional, List, Dict, Any

# Allow imports from project root when running from server/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openenv.core.env_server import Environment

from models import SummarizationAction, SummarizationObservation, SummarizationState
from tasks import get_task
from tasks.hard import HardTask
from graders import compute_reward

logger = logging.getLogger(__name__)


class SummarizationEnvironment(Environment):
    """RL environment for evaluating long-context summarization.

    The agent must condense a truncated document into a summary, then use
    that summary to answer a question about the original content. The reward
    signal trains the model to write summaries that preserve answer-critical
    information.
    """

    SUPPORTS_CONCURRENT_SESSIONS = False

    def __init__(self):
        logger.info("Initialising SummarizationEnvironment...")
        self._tasks: Dict[str, Any] = {}
        self._reset_episode_state()
        logger.info("Environment ready.")

    # ------------------------------------------------------------------
    # Internal episode state
    # ------------------------------------------------------------------

    def _reset_episode_state(self):
        self._episode_id: Optional[str] = None
        self._step_count: int = 0
        self._task_name: str = "easy"
        self._step_type: str = "summarize"
        self._messages: List[Dict[str, str]] = []
        self._ground_truth_list: List[str] = []
        self._summary: Optional[str] = None
        self._question: Optional[str] = None
        self._context_length: int = 0
        self._truncation_ratio: float = 0.7
        # Hard task only: second chunk shown after first summary
        self._hard_chunk2: Optional[str] = None

    def _get_task(self, task_name: str):
        """Lazily initialize tasks so app startup stays fast on Spaces."""
        task = self._tasks.get(task_name)
        if task is None:
            logger.info("Loading task '%s'...", task_name)
            task = get_task(task_name)
            self._tasks[task_name] = task
        return task

    # ------------------------------------------------------------------
    # OpenEnv API
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_name: Optional[str] = None,
        **kwargs,
    ) -> SummarizationObservation:
        """Start a new episode.

        Task selection priority:
          1. ``task_name`` kwarg (passed as extra field in ResetRequest)
          2. ``seed`` — seed % 3 maps to easy/medium/hard
          3. random choice
        """
        self._reset_episode_state()

        # Determine task
        if task_name is None:
            if seed is not None:
                names = ["easy", "medium", "hard"]
                task_name = names[seed % len(names)]
            else:
                task_name = random.choice(["easy", "medium", "hard"])

        self._task_name = task_name
        self._episode_id = episode_id or f"ep_{random.randint(10000, 99999)}"

        rng_seed = seed
        task = self._get_task(task_name)
        sample = task.get_sample(seed=rng_seed)

        # Store episode data
        self._question = sample["question"]
        self._ground_truth_list = sample["answer_list"]
        self._context_length = len(sample["context"])
        self._truncation_ratio = sample["truncation_ratio"]

        # Hard task: store second chunk for step 2
        if task_name == "hard" and "chunk2" in sample:
            self._hard_chunk2 = sample["chunk2"]
            first_chunk = sample["chunk1"]
        else:
            self._hard_chunk2 = None
            first_chunk = sample["truncated_context"]

        # Build initial conversation
        system_msg = {"role": "system", "content": task.get_system_prompt()}
        user_msg = {
            "role": "user",
            "content": task.get_summarize_prompt(first_chunk, self._truncation_ratio),
        }
        self._messages = [system_msg, user_msg]
        self._step_type = "summarize"

        return self._make_observation(done=False, reward=None)

    def step(self, action: SummarizationAction) -> SummarizationObservation:
        """Process one agent action and return the next observation."""
        self._step_count += 1
        response = action.response.strip()

        # Append model response to conversation history
        self._messages.append({"role": "assistant", "content": response})

        task = self._get_task(self._task_name)

        # ── Summarize step ─────────────────────────────────────────────
        if self._step_type == "summarize":
            self._summary = response

            if self._task_name == "hard" and self._hard_chunk2 is not None:
                # Hard task: move to update_summary step with second chunk
                assert isinstance(task, HardTask)
                next_msg = {
                    "role": "user",
                    "content": task.get_update_summary_prompt(self._hard_chunk2),
                }
                self._messages.append(next_msg)
                self._step_type = "update_summary"
                self._hard_chunk2 = None  # consumed
                return self._make_observation(done=False, reward=None)

            # Easy / medium: move directly to answer step
            self._step_type = "answer"
            self._messages.append(
                {"role": "user", "content": task.get_answer_prompt(self._question)}
            )
            return self._make_observation(done=False, reward=None)

        # ── Update-summary step (hard task only) ───────────────────────
        if self._step_type == "update_summary":
            self._summary = response  # updated combined summary
            self._step_type = "answer"
            assert isinstance(task, HardTask)
            self._messages.append(
                {"role": "user", "content": task.get_answer_prompt(self._question)}
            )
            return self._make_observation(done=False, reward=None)

        # ── Answer step ────────────────────────────────────────────────
        if self._step_type == "answer":
            reward = compute_reward(
                predicted=response,
                ground_truth_list=self._ground_truth_list,
                summary=self._summary,
                task_name=self._task_name,
            )
            self._step_type = "done"
            return self._make_observation(done=True, reward=reward)

        # Fallback: episode already done
        return self._make_observation(done=True, reward=0.0)

    @property
    def state(self) -> SummarizationState:
        return SummarizationState(
            episode_id=self._episode_id,
            step_count=self._step_count,
            task_name=self._task_name,
            step_type=self._step_type,
            context_length=self._context_length,
            question=self._question,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_observation(
        self, done: bool, reward: Optional[float]
    ) -> SummarizationObservation:
        return SummarizationObservation(
            done=done,
            reward=reward,
            messages=list(self._messages),  # copy
            step_type=self._step_type,
            task_name=self._task_name,
            context_length=self._context_length,
            truncation_ratio=self._truncation_ratio,
        )

    def metadata(self) -> Dict[str, Any]:
        return {
            "name": "Long-Context Summarization",
            "description": (
                "An RL environment that trains models to compress long documents into "
                "compact summaries, evaluated by their ability to answer questions from "
                "those summaries. Inspired by Cursor's self-summarization approach."
            ),
            "version": "1.0.0",
            "tasks": ["easy", "medium", "hard"],
            "action_space": "Text (summary or answer)",
            "reward_range": [0.0, 1.0],
        }
