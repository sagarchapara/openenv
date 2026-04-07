"""HTTP client for the Long-Context Summarization environment.

Usage:
    from client import SummarizationClient

    client = SummarizationClient(base_url="http://localhost:7860")

    obs = client.reset(task_name="easy", seed=42)
    while not obs.done:
        action_text = my_llm(obs.messages)
        obs = client.step(action_text)
    print(f"Final reward: {obs.reward}")
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import requests
from typing import Optional
from models import SummarizationObservation, SummarizationState


class SummarizationClient:
    """Thin HTTP wrapper around the summarization environment REST API."""

    def __init__(self, base_url: str = "http://localhost:7860", timeout: int = 60):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def reset(
        self,
        task_name: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> SummarizationObservation:
        """Reset the environment and return the initial observation."""
        payload: dict = {}
        if task_name is not None:
            payload["task_name"] = task_name
        if seed is not None:
            payload["seed"] = seed

        resp = requests.post(
            f"{self.base_url}/reset", json=payload, timeout=self.timeout
        )
        resp.raise_for_status()
        return self._parse_response(resp.json())

    def step(self, response: str) -> SummarizationObservation:
        """Send an action (text response) and return the next observation."""
        payload = {"action": {"response": response}}
        resp = requests.post(
            f"{self.base_url}/step", json=payload, timeout=self.timeout
        )
        resp.raise_for_status()
        return self._parse_response(resp.json())

    def state(self) -> SummarizationState:
        """Return current episode metadata."""
        resp = requests.get(f"{self.base_url}/state", timeout=self.timeout)
        resp.raise_for_status()
        return SummarizationState(**resp.json())

    def health(self) -> dict:
        resp = requests.get(f"{self.base_url}/health", timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _parse_response(self, data: dict) -> SummarizationObservation:
        """Parse /reset or /step JSON response into a typed observation."""
        obs_data: dict = data.get("observation", data)

        # Top-level reward/done may override what's inside obs_data
        if "reward" in data and data["reward"] is not None:
            obs_data = dict(obs_data)
            obs_data["reward"] = data["reward"]
        if "done" in data:
            obs_data = dict(obs_data)
            obs_data["done"] = data["done"]

        return SummarizationObservation(**obs_data)
