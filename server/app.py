"""FastAPI application entry point for the Summarization environment."""
import sys
import os

# Ensure project root is on the path so server/environment.py can import models, tasks, etc.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openenv.core.env_server import create_fastapi_app
from server.environment import SummarizationEnvironment
from models import SummarizationAction, SummarizationObservation

# Keep one environment instance for HTTP requests so /reset and /step share state.
_ENV = SummarizationEnvironment()


def _env_factory() -> SummarizationEnvironment:
    return _ENV


app = create_fastapi_app(
    _env_factory,
    action_cls=SummarizationAction,
    observation_cls=SummarizationObservation,
)
