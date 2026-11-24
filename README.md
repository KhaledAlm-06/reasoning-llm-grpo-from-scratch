# A Reasoning Model from Scratch

Pure-Python implementation of core reasoning LLM concepts: structured `<think>...</think><answer>...</answer>` rollouts, rule-based verification, inference-time scaling, Group Relative Policy Optimization (GRPO), and knowledge distillation.

## Current Progress
- [x] Problem definition (`MathProblem`) and rule-based `<think>`/`<answer>` verifier (`verify_rollout`)
- [x] Factorized reasoning policy (`ReasoningPolicy`)
- [ ] Inference-time scaling (`self_consistency_solve`)
- [ ] GRPO RL trainer (`train_grpo`)
- [ ] Knowledge distillation (`distill_student`)

## Quick Start
```bash
python main.py
python -m unittest discover -s tests -p "test_*.py" -v
```
