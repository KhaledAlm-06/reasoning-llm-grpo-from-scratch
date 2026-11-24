import unittest

from reasoning.policy import ReasoningPolicy
from reasoning.verifier import MathProblem, verify_rollout


class TestReasoningPolicy(unittest.TestCase):
    def test_softmax_properties_and_temperature(self) -> None:
        probs = ReasoningPolicy.softmax([1000.0, 1001.0, 999.0])
        self.assertAlmostEqual(sum(probs), 1.0, places=9)
        self.assertGreater(probs[1], probs[0])
        self.assertGreater(probs[0], probs[2])

        sharp = ReasoningPolicy.softmax([0.0, 1.0], temperature=0.2)
        flat = ReasoningPolicy.softmax([0.0, 1.0], temperature=3.0)
        self.assertGreater(sharp[1], flat[1])

    def test_greedy_rollout_and_action_log_prob(self) -> None:
        policy = ReasoningPolicy(prior_cot_logit=1.5, prior_acc_bias=2.0, seed=42)
        prob = MathProblem(12, 7, "+")

        trace, action, lp = policy.rollout(prob, greedy=True)
        self.assertEqual(action, (1, 2))
        self.assertIn("<think>", trace)
        self.assertIn("<answer>19</answer>", trace)
        self.assertAlmostEqual(lp, policy.action_log_prob("+", action), places=9)

        reward, pred = verify_rollout(prob, trace)
        self.assertAlmostEqual(reward, 1.0)
        self.assertEqual(pred, 19)

    def test_direct_format_when_cot_logit_negative(self) -> None:
        policy = ReasoningPolicy(prior_cot_logit=-1.2, prior_acc_bias=1.0, seed=42)
        prob = MathProblem(8, 3, "-")

        trace, action, _ = policy.rollout(prob, greedy=True)
        self.assertEqual(action, (0, 2))
        self.assertEqual(trace, "<answer>5</answer>")

    def test_stochastic_rollout_reproducibility(self) -> None:
        prob = MathProblem(6, 7, "*")
        p1 = ReasoningPolicy(prior_cot_logit=0.1, prior_acc_bias=0.35, seed=77)
        p2 = ReasoningPolicy(prior_cot_logit=0.1, prior_acc_bias=0.35, seed=77)

        r1 = [p1.rollout(prob, temperature=0.9, greedy=False) for _ in range(5)]
        r2 = [p2.rollout(prob, temperature=0.9, greedy=False) for _ in range(5)]
        self.assertEqual(r1, r2)

    def test_clone_isolation_and_kl_divergence(self) -> None:
        policy = ReasoningPolicy(prior_cot_logit=-0.4, prior_acc_bias=0.5)
        cloned = policy.clone()

        self.assertAlmostEqual(policy.kl_divergence(cloned, "+"), 0.0, places=9)

        cloned.format_logits["+"][1] += 1.5
        cloned.step_logits["+"][2] += 1.0
        self.assertNotEqual(policy.format_logits["+"], cloned.format_logits["+"])
        self.assertNotEqual(policy.step_logits["+"], cloned.step_logits["+"])
        self.assertGreater(cloned.kl_divergence(policy, "+"), 0.0)


if __name__ == "__main__":
    unittest.main()
