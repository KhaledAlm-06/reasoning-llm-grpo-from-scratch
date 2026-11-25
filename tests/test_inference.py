import unittest

from reasoning.inference import self_consistency_solve
from reasoning.policy import ReasoningPolicy
from reasoning.verifier import MathProblem


class _StubPolicy:
    """Deterministic stub returning pre-scripted traces for voting tests."""

    def __init__(self, traces: list[str]) -> None:
        self._traces = list(traces)
        self._idx = 0

    def rollout(
        self,
        problem: MathProblem,
        temperature: float = 1.0,
        greedy: bool = False,
    ) -> tuple[str, tuple[int, int], float]:
        trace = self._traces[self._idx % len(self._traces)]
        self._idx += 1
        return trace, (0, 2), 0.0


class TestSelfConsistencySolve(unittest.TestCase):
    def test_returns_majority_answer_and_trace_count(self) -> None:
        policy = ReasoningPolicy(prior_cot_logit=1.2, prior_acc_bias=1.8, seed=11)
        prob = MathProblem(12, 7, "+")

        pred, traces = self_consistency_solve(policy, prob, n_samples=9, temperature=0.8)
        self.assertEqual(pred, 19)
        self.assertEqual(len(traces), 9)

    def test_zero_samples_returns_none(self) -> None:
        policy = ReasoningPolicy(seed=5)
        prob = MathProblem(4, 5, "*")

        pred, traces = self_consistency_solve(policy, prob, n_samples=0)
        self.assertIsNone(pred)
        self.assertEqual(traces, [])

    def test_unparseable_traces_return_none(self) -> None:
        stub = _StubPolicy(["<think>no final tag</think>", "just thinking"])
        prob = MathProblem(10, 3, "-")

        pred, traces = self_consistency_solve(stub, prob, n_samples=2)
        self.assertIsNone(pred)
        self.assertEqual(len(traces), 2)

    def test_format_weighting_prefers_cot_on_raw_count_tie(self) -> None:
        # One <think>+<answer> vote (weight=2) outweighs one bare wrong <answer> (weight=1).
        stub = _StubPolicy(
            [
                "<think>12 + 7 = 18</think> <answer>18</answer>",
                "<answer>20</answer>",
            ]
        )
        prob = MathProblem(12, 7, "+")

        pred, traces = self_consistency_solve(stub, prob, n_samples=2)
        self.assertEqual(pred, 18)
        self.assertEqual(len(traces), 2)

    def test_reproducibility_with_matching_seed(self) -> None:
        prob = MathProblem(25, 9, "-")
        p1 = ReasoningPolicy(prior_cot_logit=-0.2, prior_acc_bias=0.6, seed=42)
        p2 = ReasoningPolicy(prior_cot_logit=-0.2, prior_acc_bias=0.6, seed=42)

        res1 = self_consistency_solve(p1, prob, n_samples=11, temperature=0.9)
        res2 = self_consistency_solve(p2, prob, n_samples=11, temperature=0.9)
        self.assertEqual(res1, res2)


if __name__ == "__main__":
    unittest.main()
