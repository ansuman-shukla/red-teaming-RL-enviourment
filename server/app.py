"""FastAPI application wiring for RedTeamEnv."""

from __future__ import annotations

import os

try:
    from openenv.core.env_server.http_server import create_app
    from openenv.core.env_server import web_interface as openenv_web_interface
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "openenv-core is required for the RedTeamEnv server. "
        "Install dependencies with `uv sync` inside red_teaming_env/`."
    ) from exc

try:
    from ..env_config import load_env_file
    from ..models import RedTeamAction, RedTeamObservation
    from .environment import RedTeamingEnvironment
    from .ui import CUSTOM_CSS, build_redteam_gradio_app
except ImportError:  # pragma: no cover
    from env_config import load_env_file
    from models import RedTeamAction, RedTeamObservation
    from server.environment import RedTeamingEnvironment
    from server.ui import CUSTOM_CSS, build_redteam_gradio_app

load_env_file()
os.environ.setdefault("ENABLE_WEB_INTERFACE", "1")
openenv_web_interface.build_gradio_app = build_redteam_gradio_app
if CUSTOM_CSS not in openenv_web_interface.OPENENV_GRADIO_CSS:
    openenv_web_interface.OPENENV_GRADIO_CSS = (
        f"{openenv_web_interface.OPENENV_GRADIO_CSS}\n{CUSTOM_CSS}"
    )

app = create_app(
    RedTeamingEnvironment,
    RedTeamAction,
    RedTeamObservation,
    env_name="red_teaming_env",
    max_concurrent_envs=20,
)


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
