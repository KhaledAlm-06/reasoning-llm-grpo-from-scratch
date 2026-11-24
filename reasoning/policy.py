"""Factorized tabular softmax policy for structured reasoning rollouts."""

from __future__ import annotations

import math
import random
from typing import Dict, List, Tuple

from .verifier import MathProblem


class ReasoningPolicy:
    """Factorized tabular/linear softmax policy over reasoning action strategies.

    For each problem operator, models logits over:
    - format mode: 0 = direct answer (no <think>), 1 = structured <think> CoT
    - step arithmetic policy: candidate offsets around true answer [delta in -2..+2]
    """

    OFFSETS = (-2, -1, 0, 1, 2)

    def __init__(
        self,
        prior_cot_logit: float = 0.0,
        prior_acc_bias: float = 0.35,
        seed: int = 123,
    ) -> None:
        self.rng = random.Random(seed)
        self.format_logits: Dict[str, List[float]] = {
            op: [0.0, prior_cot_logit] for op in ("+", "-", "*")
        }
        # Index 2 corresponds to offset 0 (exact arithmetic answer).
        self.step_logits: Dict[str, List[float]] = {
            op: [0.1, 0.25, prior_acc_bias, 0.25, 0.1] for op in ("+", "-", "*")
        }

    def clone(self) -> "ReasoningPolicy":
        copy_p = ReasoningPolicy()
        copy_p.format_logits = {k: list(v) for k, v in self.format_logits.items()}
        copy_p.step_logits = {k: list(v) for k, v in self.step_logits.items()}
        return copy_p

    @staticmethod
    def softmax(logits: List[float], temperature: float = 1.0) -> List[float]:
        temp = max(1e-3, temperature)
        scaled = [logit / temp for logit in logits]
        max_logit = max(scaled)
        exps = [math.exp(s - max_logit) for s in scaled]
        total = sum(exps)
        return [e / total for e in exps]

    def _sample_idx(self, probs: List[float]) -> int:
        r = self.rng.random()
        cum = 0.0
        for idx, p in enumerate(probs):
            cum += p
            if r <= cum:
                return idx
        return len(probs) - 1

    def rollout(
        self,
        problem: MathProblem,
        temperature: float = 1.0,
        greedy: bool = False,
    ) -> Tuple[str, Tuple[int, int], float]:
        fmt_probs = self.softmax(self.format_logits[problem.op], temperature)
        step_probs = self.softmax(self.step_logits[problem.op], temperature)

        if greedy:
            fmt_idx = max(range(2), key=lambda i: fmt_probs[i])
            off_idx = max(range(len(self.OFFSETS)), key=lambda i: step_probs[i])
        else:
            fmt_idx = self._sample_idx(fmt_probs)
            off_idx = self._sample_idx(step_probs)

        delta = self.OFFSETS[off_idx]
        pred = problem.answer + delta
        if fmt_idx == 1:
            trace = (
                f"<think>decompose {problem.a} {problem.op} {problem.b} -> {pred}</think> "
                f"<answer>{pred}</answer>"
            )
        else:
            trace = f"<answer>{pred}</answer>"

        log_prob = math.log(max(1e-12, fmt_probs[fmt_idx])) + math.log(
            max(1e-12, step_probs[off_idx])
        )
        return trace, (fmt_idx, off_idx), log_prob

    def action_log_prob(self, op: str, action: Tuple[int, int]) -> float:
        fmt_idx, off_idx = action
        fp = self.softmax(self.format_logits[op])[fmt_idx]
        sp = self.softmax(self.step_logits[op])[off_idx]
        return math.log(max(1e-12, fp)) + math.log(max(1e-12, sp))

    def kl_divergence(self, ref: "ReasoningPolicy", op: str) -> float:
        kl = 0.0
        for p_vec, q_vec in (
            (self.softmax(self.format_logits[op]), ref.softmax(ref.format_logits[op])),
            (self.softmax(self.step_logits[op]), ref.softmax(ref.step_logits[op])),
        ):
            for p, q in zip(p_vec, q_vec):
                kl += p * math.log(max(1e-12, p) / max(1e-12, q))
        return kl
