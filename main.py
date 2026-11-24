"""Demo runner for reasoning policy rollouts and verification."""

from __future__ import annotations

import sys

from reasoning.policy import ReasoningPolicy
from reasoning.verifier import MathProblem, verify_rollout

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def main() -> None:
    problems = [
        MathProblem(12, 7, "+"),
        MathProblem(25, 9, "-"),
        MathProblem(6, 7, "*"),
    ]

    policy = ReasoningPolicy(prior_cot_logit=-0.4, prior_acc_bias=0.35, seed=99)
    policy.step_logits["+"] = [0.4, 0.5, 0.35, 0.45, 0.2]
    policy.step_logits["-"] = [0.2, 0.55, 0.30, 0.40, 0.2]
    policy.step_logits["*"] = [0.3, 0.45, 0.25, 0.50, 0.25]

    print("Policy Rollout Demo:")
    for prob in problems:
        greedy_trace, greedy_act, greedy_lp = policy.rollout(prob, greedy=True)
        g_reward, g_pred = verify_rollout(prob, greedy_trace)
        print(
            f"  [{prob.prompt} = {prob.answer}] greedy action={greedy_act} "
            f"log_prob={greedy_lp:.3f} reward={g_reward:.2f} pred={g_pred} -> {greedy_trace}"
        )

        sample_trace, sample_act, sample_lp = policy.rollout(prob, temperature=0.9, greedy=False)
        s_reward, s_pred = verify_rollout(prob, sample_trace)
        print(
            f"    sample action={sample_act} log_prob={sample_lp:.3f} "
            f"reward={s_reward:.2f} pred={s_pred} -> {sample_trace}"
        )

    ref = policy.clone()
    print(f"KL(policy || clone) on '+': {policy.kl_divergence(ref, '+'):.6f}")


if __name__ == "__main__":
    main()
