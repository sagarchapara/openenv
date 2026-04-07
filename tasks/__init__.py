from .easy import EasyTask
from .medium import MediumTask
from .hard import HardTask
from .base import BaseSummarizationTask

TASK_REGISTRY = {
    "easy": EasyTask,
    "medium": MediumTask,
    "hard": HardTask,
}


def get_task(name: str) -> BaseSummarizationTask:
    if name not in TASK_REGISTRY:
        raise ValueError(f"Unknown task '{name}'. Choose from: {list(TASK_REGISTRY)}")
    return TASK_REGISTRY[name]()


__all__ = ["EasyTask", "MediumTask", "HardTask", "get_task", "TASK_REGISTRY"]
