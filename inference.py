import asyncio
import json
import logging
import os
import sys
from typing import List, Optional

from openai import OpenAI

# Required environment variables
API_BASE_URL: str = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME: str = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
# Support multiple common env var names for the key
API_KEY: str = os.environ.get("HF_TOKEN") or os.environ.get("OPENAI_API_KEY") or ""

# Environment connection config
ENV_URL: str = os.environ.get("ENV_URL", "http://localhost:7860")

# Local client import
from client import SummarizationClient

logging.basicConfig(level=logging.WARNING)

MAX_TOKENS = 400
TEMPERATURE = 0.3
MAX_STEPS = 5

SYSTEM_PROMPT = "You are a helpful assistant specialized in reading long texts, summarizing them concisely, and answering questions clearly."

BENCHMARK = "Long-Context-Summarization"


def log_start(task: str, env: str, model: str):
    print(
        f"[START] {json.dumps({'task': task, 'env': env, 'model': model})}",
        flush=True,
    )


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]):
    print(
        f"[STEP] {json.dumps({'step': step, 'action': action, 'reward': reward, 'done': done, 'error': error})}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]):
    print(
        f"[END] {json.dumps({'success': success, 'steps': steps, 'score': score, 'rewards': rewards})}",
        flush=True,
    )


def get_model_message(client: OpenAI, messages: List[dict]) -> str:
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        return text if text else "hello"
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return "hello"


async def run_episode(client: OpenAI, env_client: SummarizationClient, task_name: str, seed: int) -> float:
    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        obs = env_client.reset(task_name=task_name, seed=seed)
        last_reward = 0.0

        for step in range(1, MAX_STEPS + 1):
            if obs.done:
                break

            message = get_model_message(client, obs.messages)

            # Important: step using our client
            obs = env_client.step(response=message)

            reward = obs.reward if obs.reward is not None else 0.0
            done = obs.done
            error = None

            rewards.append(reward)
            steps_taken = step
            last_reward = reward

            log_step(step=step, action=message, reward=reward, done=done, error=error)

            history.append(f"Step {step}: {message[:50]!r}... -> reward {reward:+.2f}")

            if done:
                break

        score = rewards[-1] if rewards else 0.0
        score = min(max(score, 0.0), 1.0)  # clamp to [0, 1]
        
        # Determine success visually like the hackathon expected
        success = score >= 0.3

    except Exception as e:
        print(f"[DEBUG] Episode error: {e}", flush=True)
        # Assuming failure
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


async def main() -> None:
    if not API_KEY:
        print("[WARN] HF_TOKEN not set — LLM calls may fail.", file=sys.stderr)

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY or "EMPTY")
    env_client = SummarizationClient(base_url=ENV_URL)

    # Verify server is up
    try:
        env_client.health()
    except Exception as e:
        print(f"[ERROR] Cannot reach environment at {ENV_URL}: {e}", file=sys.stderr)
        sys.exit(1)

    tasks = ["easy", "medium", "hard"]
    
    for task_name in tasks:
        # Run exactly 3 episodes per task as per typical baselines
        for ep_num in range(3):
            seed = ep_num
            await run_episode(client, env_client, task_name, seed)
            await asyncio.sleep(0.5)


if __name__ == "__main__":
    asyncio.run(main())
