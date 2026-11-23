import unittest

from reasoning.verifier import MathProblem, verify_rollout


class TestMathProblem(unittest.TestCase):
    def test_arithmetic_operations(self) -> None:
        self.assertEqual(MathProblem(12, 7, "+").answer, 19)
        self.assertEqual(MathProblem(25, 9, "-").answer, 16)
        self.assertEqual(MathProblem(6, 7, "*").answer, 42)

    def test_prompt_format(self) -> None:
        self.assertEqual(MathProblem(8, 9, "*").prompt, "8 * 9")

    def test_invalid_operator_raises(self) -> None:
        with self.assertRaises(ValueError):
            _ = MathProblem(4, 2, "/").answer


class TestVerifyRollout(unittest.TestCase):
    def setUp(self) -> None:
        self.problem = MathProblem(12, 7, "+")

    def test_full_cot_and_correct_answer(self) -> None:
        trace = "<think>step by step\n12 + 7 = 19</think>\n<answer>19</answer>"
        reward, pred = verify_rollout(self.problem, trace)
        self.assertAlmostEqual(reward, 1.0)
        self.assertEqual(pred, 19)

    def test_direct_correct_answer_without_think(self) -> None:
        reward, pred = verify_rollout(self.problem, "<answer>19</answer>")
        self.assertAlmostEqual(reward, 0.85)
        self.assertEqual(pred, 19)

    def test_cot_with_wrong_answer(self) -> None:
        reward, pred = verify_rollout(self.problem, "<think>12+7</think> <answer>18</answer>")
        self.assertAlmostEqual(reward, 0.2)
        self.assertEqual(pred, 18)

    def test_negative_answer_parsing(self) -> None:
        prob = MathProblem(5, 12, "-")
        reward, pred = verify_rollout(prob, "<think>5 - 12 = -7</think> <answer> -7 </answer>")
        self.assertAlmostEqual(reward, 1.0)
        self.assertEqual(pred, -7)

    def test_missing_answer_tag(self) -> None:
        reward_think, pred_think = verify_rollout(self.problem, "<think>19</think>")
        self.assertAlmostEqual(reward_think, 0.1)
        self.assertIsNone(pred_think)

        reward_empty, pred_empty = verify_rollout(self.problem, "just 19")
        self.assertAlmostEqual(reward_empty, 0.0)
        self.assertIsNone(pred_empty)


if __name__ == "__main__":
    unittest.main()
