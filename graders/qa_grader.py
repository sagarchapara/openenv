"""Reward computation for the Long-Context Summarization environment.

Uses token-level F1 score (same metric as the SQuAD official evaluation)
for open-ended QA, and exact match for multiple-choice tasks.
"""
import os
import re
import string
from collections import Counter
from typing import List, Optional

from .llm_grader import grade_summary_quality


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


ANSWER_REWARD_THRESHOLD = _env_float("GRADER_ANSWER_REWARD_THRESHOLD", 0.5)
SHORT_SUMMARY_WORD_LIMIT = _env_int("GRADER_SHORT_SUMMARY_WORD_LIMIT", 150)
LONG_SUMMARY_WORD_LIMIT = _env_int("GRADER_LONG_SUMMARY_WORD_LIMIT", 300)
SHORT_SUMMARY_BONUS = _env_float("GRADER_SHORT_SUMMARY_BONUS", 0.05)
LONG_SUMMARY_BONUS = _env_float("GRADER_LONG_SUMMARY_BONUS", 0.02)

ANSWER_WEIGHT = _env_float("GRADER_ANSWER_WEIGHT", 0.75)
LLM_SUMMARY_WEIGHT = _env_float("GRADER_LLM_SUMMARY_WEIGHT", 0.20)
CONCISENESS_WEIGHT = _env_float("GRADER_CONCISENESS_WEIGHT", 0.05)
LLM_BLEND_MIN_ANSWER_REWARD = _env_float("GRADER_LLM_BLEND_MIN_ANSWER_REWARD", 0.5)


def compute_conciseness_bonus(summary: Optional[str], answer_reward: float) -> float:
    """Compute the deterministic conciseness bonus."""
    if summary is None or answer_reward <= ANSWER_REWARD_THRESHOLD:
        return 0.0

    summary_word_count = len(summary.split())
    if summary_word_count <= SHORT_SUMMARY_WORD_LIMIT:
        return SHORT_SUMMARY_BONUS
    if summary_word_count <= LONG_SUMMARY_WORD_LIMIT:
        return LONG_SUMMARY_BONUS
    return 0.0


def canonicalize_numeric_text(text: str) -> str:
    """Normalize common numeric surface forms used in short factual answers."""
    normalized = text.lower().strip()
    normalized = normalized.replace("percent", "%")
    normalized = re.sub(r"\bper cent\b", "%", normalized)
    normalized = re.sub(r"(\d)\s+%", r"\1%", normalized)
    normalized = re.sub(r"(?<=\d),(?=\d)", "", normalized)
    return normalized


def normalize_answer(s: str) -> str:
    """Lowercase, remove punctuation, articles, and extra whitespace."""
    s = canonicalize_numeric_text(s)

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def get_tokens(s: str) -> List[str]:
    if not s:
        return []
    return normalize_answer(s).split()


def compute_f1(prediction: str, ground_truth: str) -> float:
    """Compute token-level F1 between prediction and ground truth."""
    pred_tokens = get_tokens(prediction)
    gt_tokens = get_tokens(ground_truth)

    if not pred_tokens and not gt_tokens:
        return 1.0
    if not pred_tokens or not gt_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def compute_exact_match(prediction: str, ground_truth: str) -> float:
    """Compute exact match after normalization (0.0 or 1.0)."""
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def best_f1_against_list(prediction: str, ground_truth_list: List[str]) -> float:
    """Return the best F1 score against any ground truth in the list."""
    if not ground_truth_list:
        return 0.0
    return max(compute_f1(prediction, gt) for gt in ground_truth_list)


def best_exact_match_against_list(prediction: str, ground_truth_list: List[str]) -> float:
    """Return 1.0 if prediction exactly matches any ground truth."""
    if not ground_truth_list:
        return 0.0
    return max(compute_exact_match(prediction, gt) for gt in ground_truth_list)


def compute_reward(
    predicted: str,
    ground_truth_list: List[str],
    summary: Optional[str],
    task_name: str,
    question: Optional[str] = None,
) -> float:
    """Compute final reward for an episode.

    For all tasks: token-level F1 score (0.0–1.0).

    A small conciseness bonus (+0.05) is applied when the model answers
    correctly AND the summary is reasonably compact (<= 300 words).
    This encourages learning to summarize efficiently.
    """
    # Extractive/free-form QA: token-level F1
    answer_reward = best_f1_against_list(predicted, ground_truth_list)

    # Conciseness bonus: reward shorter summaries when the answer is good enough
    conciseness_bonus = compute_conciseness_bonus(summary, answer_reward)

    llm_summary_score = grade_summary_quality(
        summary=summary,
        question=question,
        predicted=predicted,
        ground_truth_list=ground_truth_list,
        task_name=task_name,
    )
    if llm_summary_score is None or answer_reward < LLM_BLEND_MIN_ANSWER_REWARD:
        return min(1.0, answer_reward + conciseness_bonus)

    blended_reward = (
        (ANSWER_WEIGHT * answer_reward)
        + (LLM_SUMMARY_WEIGHT * llm_summary_score)
        + (CONCISENESS_WEIGHT * (1.0 if conciseness_bonus > 0 else 0.0))
    )
    return min(1.0, max(0.0, blended_reward))
