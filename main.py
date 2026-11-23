"""Smoke runner for the reasoning verifier milestone."""

from __future__ import annotations

import sys

from reasoning.verifier import MathProblem, verify_rollout

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def main() -> None:
    sample = MathProblem(12, 7, "+")
    traces = [
        "<think>12 + 7 = 19</think> <answer>19</answer>",
        "<answer>19</answer>",
        "<think>guess 20</think> <answer>20</answer>",
        "<think>incomplete</think>",
    ]

    print(f"Problem: {sample.prompt} (expected {sample.answer})")
    for trace in traces:
        reward, pred = verify_rollout(sample, trace)
        print(f"  reward={reward:.2f}, pred={pred} <- {trace}")


if __name__ == "__main__":
    main()
