"""Submission baseline for the Long-Context Summarization environment.

This script follows the hackathon logging contract exactly:
  [START] task=<task_name> env=<benchmark> model=<model_name>
  [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

Supported execution modes:
  1. Connect to a running environment via ENV_URL
  2. Start the environment from a local Docker image via LOCAL_IMAGE_NAME / IMAGE_NAME
"""

from __future__ import annotations

import os
import re
import sys
from typing import Any, List, Optional, Tuple

import requests
from openai import OpenAI
from openenv.core.containers.runtime.providers import LocalDockerProvider

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")

ENV_URL = os.getenv("ENV_URL", "").strip()
LOCAL_IMAGE_NAME = (
    os.getenv("LOCAL_IMAGE_NAME") or os.getenv("IMAGE_NAME") or ""
).strip()

BENCHMARK = os.getenv("OPENENV_BENCHMARK", "long-context-summarization")
TASK_NAME = os.getenv("TASK_NAME", "").strip()
TASKS = [TASK_NAME] if TASK_NAME else ["easy", "medium", "hard"]

TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
MAX_TOKENS_SUMMARY = int(os.getenv("MAX_TOKENS_SUMMARY", "220"))
MAX_TOKENS_ANSWER = int(os.getenv("MAX_TOKENS_ANSWER", "80"))
SUCCESS_SCORE_THRESHOLD = float(os.getenv("SUCCESS_SCORE_THRESHOLD", "0.30"))


def _print_stderr(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _flatten_log_value(value: Any, *, limit: int = 160) -> str:
    text = str(value if value is not None else "null")
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit] if len(text) > limit else text


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int, action: str, reward: float, done: bool, error: Optional[str]
) -> None:
    error_value = _flatten_log_value(error) if error else "null"
    print(
        f"[STEP] step={step} action={_flatten_log_value(action)} "
        f"reward={reward:.2f} done={str(done).lower()} error={error_value}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{reward:.2f}" for reward in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def extract_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append((item.get("text") or "").strip())
        return "\n".join(part for part in parts if part).strip()
    return ""


def normalize_action(text: str, step_type: str) -> str:
    cleaned = (text or "").strip()
    if cleaned:
        return cleaned
    if step_type == "answer":
        return "I do not know."
    return "No summary available."


def max_tokens_for_step(step_type: str) -> int:
    return MAX_TOKENS_ANSWER if step_type == "answer" else MAX_TOKENS_SUMMARY


def generate_action(client: OpenAI, messages: List[dict[str, str]], step_type: str) -> str:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=max_tokens_for_step(step_type),
        stream=False,
    )
    text = extract_text(response.choices[0].message.content)
    return normalize_action(text, step_type)


def connect_environment() -> Tuple[str, Optional[LocalDockerProvider]]:
    if ENV_URL:
        return ENV_URL.rstrip("/"), None
    if LOCAL_IMAGE_NAME:
        provider = LocalDockerProvider()
        base_url = provider.start_container(LOCAL_IMAGE_NAME)
        provider.wait_for_ready(base_url, timeout_s=60.0)
        return base_url.rstrip("/"), provider
    raise RuntimeError("Set either ENV_URL or LOCAL_IMAGE_NAME (or IMAGE_NAME).")


def env_reset(base_url: str, task_name: str) -> dict:
    response = requests.post(
        f"{base_url}/reset",
        json={"task_name": task_name},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def env_step(base_url: str, action: str) -> dict:
    response = requests.post(
        f"{base_url}/step",
        json={"action": {"response": action}},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def run_task(base_url: str, client: OpenAI, task_name: str) -> float:
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = env_reset(base_url, task_name)

        while not result.get("done", False):
            steps_taken += 1
            observation = result.get("observation", {})
            step_type = observation.get("step_type", "summarize")
            messages = observation.get("messages", [])

            action = generate_action(client, messages, step_type)
            result = env_step(base_url, action)

            reward = float(result.get("reward") or 0.0)
            rewards.append(reward)

            log_step(
                step=steps_taken,
                action=action,
                reward=reward,
                done=result.get("done", False),
                error=None,
            )

        score = float(result.get("reward") or 0.0)
        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD
        return score
    except Exception as exc:
        log_step(
            step=max(steps_taken, 1),
            action="runtime_error",
            reward=0.0,
            done=True,
            error=str(exc),
        )
        return 0.0
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


def main() -> None:
    if not HF_TOKEN:
        _print_stderr("HF_TOKEN is not set; authenticated LLM calls may fail.")

    llm_client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN or "EMPTY")
    base_url, provider = connect_environment()

    try:
        for task_name in TASKS:
            run_task(base_url, llm_client, task_name)
    finally:
        try:
            if provider is not None:
                provider.stop_container()
        except Exception:
            pass


if __name__ == "__main__":
    main()
