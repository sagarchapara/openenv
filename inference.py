"""Baseline inference script for the Long-Context Summarization environment.

Structured logging follows the hackathon specification:
  [START] {...}   — episode begins
  [STEP]  {...}   — each environment step
  [END]   {...}   — episode concludes with final reward

Required environment variables:
  API_BASE_URL  — LLM API endpoint (OpenAI-compatible)
  MODEL_NAME    — model identifier
  HF_TOKEN      — API key / HuggingFace token

Optional:
  ENV_URL       — environment server URL (default: http://localhost:7860)

Usage:
  export API_BASE_URL="https://api-inference.huggingface.co/v1/"
  export MODEL_NAME="meta-llama/Llama-3.2-3B-Instruct"
  export HF_TOKEN="hf_..."
  python inference.py
"""

import json
import logging
import os
import sys
import time
from typing import Optional

import requests
from openai import OpenAI

logging.basicConfig(level=logging.WARNING)

# ── Configuration ──────────────────────────────────────────────────────────────

API_BASE_URL: str = os.environ.get(
    "API_BASE_URL", "https://api-inference.huggingface.co/v1/"
)
MODEL_NAME: str = os.environ.get(
    "MODEL_NAME", "meta-llama/Llama-3.2-3B-Instruct"
)
HF_TOKEN: str = os.environ.get("HF_TOKEN", "")
ENV_URL: str = os.environ.get("ENV_URL", "http://localhost:7860").rstrip("/")

MAX_TOKENS_SUMMARY = 400
MAX_TOKENS_ANSWER = 150
TEMPERATURE = 0.3
EPISODES_PER_TASK = 3  # run multiple episodes per task for stable avg reward
MAX_STEPS_PER_EPISODE = 30  # safety guard to avoid infinite loops if env never returns done=true
DEBUG_LLM = os.environ.get("DEBUG_LLM", "0") == "1"
DEBUG_ENV = os.environ.get("DEBUG_ENV", "0") == "1"

# ── Logging helpers ────────────────────────────────────────────────────────────


def log_start(task: str, episode_id: str, episode_num: int):
    print(
        f"[START] {json.dumps({'task': task, 'episode_id': episode_id, 'episode_num': episode_num})}",
        flush=True,
    )


def log_step(step: int, step_type: str, action_preview: str, reward: Optional[float]):
    print(
        f"[STEP] {json.dumps({'step': step, 'type': step_type, 'action_preview': action_preview[:120], 'reward': reward})}",
        flush=True,
    )


def log_end(task: str, episode_id: str, total_steps: int, final_reward: float):
    print(
        f"[END] {json.dumps({'task': task, 'episode_id': episode_id, 'total_steps': total_steps, 'final_reward': round(final_reward, 4), 'success': final_reward > 0.3})}",
        flush=True,
    )


def log_error(task: str, error: str):
    print(f"[ERROR] {json.dumps({'task': task, 'error': error})}", flush=True)


def log_debug(tag: str, payload: dict):
    print(f"[DEBUG] {json.dumps({'tag': tag, **payload}, default=str)}", flush=True)


# ── Environment helpers ────────────────────────────────────────────────────────


def env_reset(task_name: str, seed: Optional[int] = None) -> dict:
    payload = {"task_name": task_name}
    if seed is not None:
        payload["seed"] = seed
    resp = requests.post(f"{ENV_URL}/reset", json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()


def env_step(action_text: str) -> dict:
    resp = requests.post(
        f"{ENV_URL}/step",
        json={"action": {"response": action_text}},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()


def parse_obs(data: dict) -> dict:
    """Normalise /reset or /step response into a flat observation dict."""
    obs = dict(data.get("observation", data))
    if "reward" in data and data["reward"] is not None:
        obs["reward"] = data["reward"]
    if "done" in data:
        obs["done"] = data["done"]
    return obs


# ── LLM call ──────────────────────────────────────────────────────────────────


def extract_text_from_content(content) -> str:
    """Handle providers that return either plain text or structured content parts."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append((item.get("text") or "").strip())
        return "\n".join(p for p in parts if p).strip()
    return ""


def call_llm(client: OpenAI, messages: list, step_type: str) -> str:
    max_tokens = MAX_TOKENS_SUMMARY if step_type == "summarize" else MAX_TOKENS_ANSWER
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        max_tokens=max_tokens,
        temperature=TEMPERATURE,
    )
    choice = response.choices[0]
    content = choice.message.content
    text = extract_text_from_content(content)

    if DEBUG_LLM:
        raw_msg = {}
        if hasattr(choice.message, "model_dump"):
            raw_msg = choice.message.model_dump()
        log_debug(
            "llm_response",
            {
                "step_type": step_type,
                "finish_reason": getattr(choice, "finish_reason", None),
                "content_type": type(content).__name__,
                "content_preview": str(content)[:240],
                "parsed_text_preview": text[:240],
                "raw_message": raw_msg,
            },
        )

    return text


def normalize_action_text(action_text: str, step_type: str) -> str:
    """Ensure we always send usable text to the environment."""
    text = (action_text or "").strip()
    if text.lower() in {"", "none", "null", "n/a"}:
        if step_type == "answer":
            return "I don't know."
        return "No summary available."
    return text


# ── Episode runner ─────────────────────────────────────────────────────────────


def run_episode(client: OpenAI, task_name: str, seed: int) -> float:
    """Run a single episode and return the final reward (0.0 on failure)."""
    try:
        raw = env_reset(task_name, seed=seed)
        obs = parse_obs(raw)
        episode_id = obs.get("metadata", {}).get("episode_id", f"{task_name}-{seed}")

        log_start(task_name, episode_id, seed)

        step_num = 0
        final_reward = 0.0

        while not obs.get("done", False):
            if step_num >= MAX_STEPS_PER_EPISODE:
                raise RuntimeError(
                    f"Episode exceeded max steps ({MAX_STEPS_PER_EPISODE}) without done=true"
                )
            step_num += 1
            step_type = obs.get("step_type", "summarize")
            messages = obs.get("messages", [])

            action_text = call_llm(client, messages, step_type)
            action_text = normalize_action_text(action_text, step_type)

            raw = env_step(action_text)
            next_obs = parse_obs(raw)

            if DEBUG_ENV:
                log_debug(
                    "env_step",
                    {
                        "step_num": step_num,
                        "step_type": step_type,
                        "action_preview": action_text[:120],
                        "raw_done": raw.get("done"),
                        "raw_reward": raw.get("reward"),
                        "obs_done": next_obs.get("done"),
                        "obs_reward": next_obs.get("reward"),
                        "next_step_type": next_obs.get("step_type"),
                    },
                )

            log_step(step_num, step_type, action_text, next_obs.get("reward"))
            obs = next_obs

        final_reward = obs.get("reward") or 0.0
        episode_id_end = obs.get("metadata", {}).get("episode_id", episode_id)
        log_end(task_name, episode_id_end, step_num, final_reward)
        return final_reward

    except Exception as exc:
        log_error(task_name, str(exc))
        return 0.0


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    if not HF_TOKEN:
        print(
            "[WARN] HF_TOKEN not set — LLM calls may fail if the endpoint requires auth.",
            file=sys.stderr,
        )

    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN or "EMPTY")

    # Verify server is up
    try:
        resp = requests.get(f"{ENV_URL}/health", timeout=10)
        resp.raise_for_status()
    except Exception as e:
        print(f"[ERROR] Cannot reach environment at {ENV_URL}: {e}", file=sys.stderr)
        sys.exit(1)

    tasks = ["easy", "medium", "hard"]
    all_rewards: dict[str, list[float]] = {t: [] for t in tasks}

    for task_name in tasks:
        for ep in range(EPISODES_PER_TASK):
            seed = ep  # seeds 0,1,2 → reproducible but varied episodes
            reward = run_episode(client, task_name, seed=seed)
            all_rewards[task_name].append(reward)
            time.sleep(0.5)  # brief pause between episodes

    # Final summary
    summary = {
        task: {
            "rewards": rewards,
            "avg_reward": round(sum(rewards) / len(rewards), 4) if rewards else 0.0,
        }
        for task, rewards in all_rewards.items()
    }
    overall_avg = sum(
        summary[t]["avg_reward"] for t in tasks
    ) / len(tasks)
    summary["overall_avg"] = round(overall_avg, 4)

    print(f"[SUMMARY] {json.dumps(summary)}", flush=True)


if __name__ == "__main__":
    main()
