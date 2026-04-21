"""
FastAPI application for the Spaces Pipeline Pro Environment.

Endpoints:
    - POST /reset: Reset environment
    - POST /step: Execute action
    - GET /state: Get current state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError(
        "openenv-core is required. Install with 'uv sync' or pip."
    ) from e

try:
    from spaces_pipeline_env.models import (
        SpacesPipelineAction,
        SpacesPipelineObservation,
    )
    from spaces_pipeline_env.server.spaces_pipeline_environment import (
        SpacesPipelineEnvironment,
    )
except ImportError:
    try:
        from ..models import SpacesPipelineAction, SpacesPipelineObservation
        from .spaces_pipeline_environment import SpacesPipelineEnvironment
    except ImportError:
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from models import SpacesPipelineAction, SpacesPipelineObservation
        from server.spaces_pipeline_environment import SpacesPipelineEnvironment


app = create_app(
    SpacesPipelineEnvironment,
    SpacesPipelineAction,
    SpacesPipelineObservation,
    env_name="spaces_pipeline_env",
    max_concurrent_envs=1,
)


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Run the server directly."""
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)
