"""FastAPI application entry point for the Summarization environment."""
import sys
import os

import uvicorn

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


@app.get("/")
def root() -> dict:
    """Friendly landing page for Spaces and browser visits."""
    return {
        "name": "Long-Context Summarization",
        "status": "healthy",
        "docs": {
            "health": "/health",
            "schema": "/schema",
            "metadata": "/metadata",
            "reset": "POST /reset",
            "step": "POST /step",
            "state": "GET /state",
        },
    }


def main() -> None:
    """Run the environment server for local validation and script entrypoints."""
    port = int(os.environ.get("PORT", "7860"))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port, workers=1)


if __name__ == "__main__":
    main()
