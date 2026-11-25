"""Demo runner for reasoning policy rollouts, verification, and self-consistency."""

from __future__ import annotations

import sys

from reasoning.inference import self_consistency_solve
from reasoning.policy import ReasoningPolicy
from reasoning.verifier import MathProblem, verify_rollout

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def main() -> None:
    problems = [
        MathProblem(12, 7, "+"),
        MathProblem(19, 8, "+"),
        MathProblem(25, 9, "-"),
        MathProblem(42, 17, "-"),
        MathProblem(6, 7, "*"),
        MathProblem(8, 9, "*"),
    ]

    policy = ReasoningPolicy(prior_cot_logit=-0.4, prior_acc_bias=0.35, seed=99)
    policy.step_logits["+"] = [0.4, 0.5, 0.35, 0.45, 0.2]
    policy.step_logits["-"] = [0.2, 0.55, 0.30, 0.40, 0.2]
    policy.step_logits["*"] = [0.3, 0.45, 0.25, 0.50, 0.25]

    print("Policy Rollout & Self-Consistency Demo:")
    greedy_correct = 0
    sc_correct = 0
    for prob in problems:
        greedy_trace, greedy_act, greedy_lp = policy.rollout(prob, greedy=True)
        g_reward, g_pred = verify_rollout(prob, greedy_trace)
        if g_pred == prob.answer:
            greedy_correct += 1

        sc_pred, sc_traces = self_consistency_solve(
            policy, prob, n_samples=15, temperature=0.8
        )
        if sc_pred == prob.answer:
            sc_correct += 1

        print(
            f"  [{prob.prompt} = {prob.answer}] greedy_pred={g_pred} (reward={g_reward:.2f}, "
            f"act={greedy_act}, lp={greedy_lp:.3f}) | self_consistency(N={len(sc_traces)})={sc_pred}"
        )

    print(
        f"Accuracy: greedy={greedy_correct / len(problems):.2f}, "
        f"self-consistency={sc_correct / len(problems):.2f}"
    )

    ref = policy.clone()
    print(f"KL(policy || clone) on '+': {policy.kl_divergence(ref, '+'):.6f}")


if __name__ == "__main__":
    main()
