"""Reward computation for the Long-Context Summarization environment.

Uses token-level F1 score (same metric as the SQuAD official evaluation)
for open-ended QA, and exact match for multiple-choice tasks.
"""
import re
import string
from collections import Counter
from typing import List, Optional


def normalize_answer(s: str) -> str:
    """Lowercase, remove punctuation, articles, and extra whitespace."""

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
) -> float:
    """Compute final reward for an episode.

    For all tasks: token-level F1 score (0.0–1.0).

    A small conciseness bonus (+0.05) is applied when the model answers
    correctly AND the summary is reasonably compact (<= 300 words).
    This encourages learning to summarize efficiently.
    """
    # Extractive/free-form QA: token-level F1
    base_reward = best_f1_against_list(predicted, ground_truth_list)

    # Conciseness bonus: reward shorter summaries when the answer is correct
    conciseness_bonus = 0.0
    if base_reward > 0.5 and summary is not None:
        summary_word_count = len(summary.split())
        if summary_word_count <= 150:
            conciseness_bonus = 0.05
        elif summary_word_count <= 300:
            conciseness_bonus = 0.02

    return min(1.0, base_reward + conciseness_bonus)
