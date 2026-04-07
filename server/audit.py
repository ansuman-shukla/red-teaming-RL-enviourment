"""Adaptive audit sampling for RedTeamEnv."""

from __future__ import annotations

from collections import deque
import random


class AdaptiveAuditSampler:
    """Random audit sampler with reward-spike escalation."""

    def __init__(self) -> None:
        self._recent_rewards: deque[float] = deque(maxlen=10)
        self._forced_audits_remaining = 0

    def sample_episode(self, rng: random.Random) -> bool:
        """Return whether the next episode should be deeply audited."""
        if self._forced_audits_remaining > 0:
            self._forced_audits_remaining -= 1
            return True
        return rng.randint(1, 100) == 1

    def record_episode(self, total_reward: float) -> None:
        """Update rolling reward history and trigger audit escalation if needed."""
        self._recent_rewards.append(total_reward)
        if len(self._recent_rewards) < 6:
            return

        recent = list(self._recent_rewards)
        current_window = recent[-5:]
        previous_window = recent[:-5]
        if not previous_window:
            return

        current_avg = sum(current_window) / len(current_window)
        previous_avg = sum(previous_window) / len(previous_window)
        if current_avg - previous_avg > 0.30:
            self._forced_audits_remaining = 5

