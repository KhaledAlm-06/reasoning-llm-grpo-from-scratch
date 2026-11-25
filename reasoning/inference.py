"""Inference-time scaling via format-weighted self-consistency voting."""

from __future__ import annotations

from collections import Counter
from typing import List, Optional, Tuple

from .policy import ReasoningPolicy
from .verifier import MathProblem, verify_rollout


def self_consistency_solve(
    policy: ReasoningPolicy,
    problem: MathProblem,
    n_samples: int = 9,
    temperature: float = 0.9,
) -> Tuple[Optional[int], List[str]]:
    """Sample stochastic reasoning rollouts and pick the weighted majority answer."""
    votes: Counter[int] = Counter()
    traces: List[str] = []

    for _ in range(n_samples):
        trace, _, _ = policy.rollout(problem, temperature=temperature, greedy=False)
        traces.append(trace)
        reward, pred = verify_rollout(problem, trace)
        if pred is not None:
            # Weight well-formatted or verified traces higher than bare wrong answers.
            weight = 2 if reward >= 0.2 else 1
            votes[pred] += weight

    if not votes:
        return None, traces
    return votes.most_common(1)[0][0], traces
