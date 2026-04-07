"""Async OpenEnv client for RedTeamEnv."""

from __future__ import annotations

from typing import Any

from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient

from .models import RedTeamAction, RedTeamObservation, RedTeamState


class RedTeamingEnv(EnvClient[RedTeamAction, RedTeamObservation, RedTeamState]):
    """WebSocket client for the RedTeamEnv environment."""

    def _step_payload(self, action: RedTeamAction) -> dict[str, Any]:
        return action.model_dump(mode="json")

    def _parse_result(self, payload: dict[str, Any]) -> StepResult[RedTeamObservation]:
        observation = RedTeamObservation(**payload.get("observation", {}))
        reward = payload.get("reward", observation.reward)
        done = bool(payload.get("done", observation.done))
        return StepResult(observation=observation, reward=reward, done=done)

    def _parse_state(self, payload: dict[str, Any]) -> RedTeamState:
        return RedTeamState(**payload)

