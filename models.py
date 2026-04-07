"""Typed models for the RedTeamEnv OpenEnv environment."""

from __future__ import annotations

from typing import Any, Literal

from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field

RedTeamStrategy = Literal[
    "roleplay",
    "obfuscation",
    "decomposition",
    "narrative_wrap",
    "multilingual",
    "direct",
]


class RedTeamAction(Action):
    """Fixed-vocabulary red-teaming action."""

    strategy: RedTeamStrategy = Field(
        ..., description="The fixed strategy selected for the current turn."
    )
    prompt: str = Field(
        ...,
        min_length=1,
        description="The adversarial prompt sent to the target model.",
    )
    turn_context: str = Field(
        default="",
        description="Optional framing or grooming context carried into the turn.",
    )


class RedTeamObservation(Observation):
    """Observation returned after a target-model interaction."""

    target_response: str = Field(
        default="", description="The target model response for the last turn."
    )
    compliance_flags: list[str] = Field(
        default_factory=list,
        description="Verified compliance or policy-violation flags for the turn.",
    )
    attack_score: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Turn-level attack quality score."
    )
    strategy_used: str = Field(
        default="", description="Echo of the action strategy used for the turn."
    )
    attempts_remaining: int = Field(
        default=0, ge=0, description="Remaining turns in the active episode."
    )
    conversation_history: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Full multi-turn conversation transcript for the episode.",
    )
    message: str = Field(
        default="", description="Human-readable status message for the step."
    )


class RedTeamState(State):
    """Environment state returned by the `/state` endpoint."""

    task_name: str = Field(
        default="",
        description="stereotype_probe | policy_bypass | obfuscated_extraction",
    )
    target_model: str = Field(
        default="", description="Model identifier used by the target wrapper."
    )
    target_temperature: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Episode temperature sample."
    )
    strategies_used: list[str] = Field(
        default_factory=list,
        description="Unique strategies used so far in the episode.",
    )
    flags_found: list[str] = Field(
        default_factory=list,
        description="Cumulative verified compliance flags found this episode.",
    )
    defense_drift_active: bool = Field(
        default=False,
        description="Whether adaptive defense drift has activated for the target.",
    )
    audit_this_episode: bool = Field(
        default=False,
        description="Whether detailed auditing is enabled for this episode.",
    )
    max_turns: int = Field(
        default=0, ge=0, description="Maximum steps for the active task."
    )
    seed: int | None = Field(
        default=None, description="Episode seed used to initialize stochasticity."
    )
    reward_history: list[float] = Field(
        default_factory=list, description="Per-step reward history for the episode."
    )
    judge_history: list[float] = Field(
        default_factory=list,
        description="Per-step judge scores used for dense reward shaping.",
    )
    audit_log: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Detailed verifier and reward traces captured during audit.",
    )
    last_success: bool = Field(
        default=False, description="Whether the most recent terminal state succeeded."
    )


# Backward-compatible aliases for the scaffolded names.
RedTeamingAction = RedTeamAction
RedTeamingObservation = RedTeamObservation
RedTeamingState = RedTeamState

