"""Problem definitions and rule-based Chain-of-Thought rollout verifier."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Tuple


_THINK_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL)
_ANSWER_RE = re.compile(r"<answer>\s*(-?\d+)\s*</answer>")


@dataclass(frozen=True)
class MathProblem:
    """Simple arithmetic problem used to evaluate reasoning rollouts."""

    a: int
    b: int
    op: str

    @property
    def prompt(self) -> str:
        return f"{self.a} {self.op} {self.b}"

    @property
    def answer(self) -> int:
        if self.op == "+":
            return self.a + self.b
        if self.op == "-":
            return self.a - self.b
        if self.op == "*":
            return self.a * self.b
        raise ValueError(f"Unsupported operator: {self.op!r}")


def verify_rollout(problem: MathProblem, trace: str) -> Tuple[float, Optional[int]]:
    """Score format compliance (up to 0.2) and mathematical correctness (0.8)."""
    has_think = bool(_THINK_RE.search(trace))
    ans_match = _ANSWER_RE.search(trace)
    if not ans_match:
        return (0.1 if has_think else 0.0), None

    pred = int(ans_match.group(1))
    format_reward = 0.2 if has_think else 0.05
    correct_reward = 0.8 if pred == problem.answer else 0.0
    return format_reward + correct_reward, pred
