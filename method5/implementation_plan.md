# Implement Multi-Agent RL + Diversity-Aware Reward

This plan implements a Multi-Agent architecture for the TAS-Net Reinforcement Learning codebase.

## User Review Required

- **Combine Method:** I will add `--combine_method` (`average` or `union`).
- **Diversity Penalty:** I will calculate similarity using cosine similarity of the action probabilities of the agents.
- **Shared Parameters:** I will add `--shared_params`. If provided, `Network1` (Feature Extraction) is shared, and only `Network2` (Actor) is duplicated across agents. If not provided, both are duplicated.
- **Logging:** Metrics will be logged per epoch to `logs/multi_agent_metrics.csv` using Python's `csv` module.

## Proposed Changes

### TAS-Net

#### [MODIFY] main.py
- Add `argparse` arguments: `--num_agents`, `--shared_params`, `--combine_method`, `--lambda_div`.
- In training and testing loops, initialize a list of `Network1` and `Network2` depending on `--shared_params`.
- Run forward passes for all agents to get a list of `sig_probs`.
- Compute pairwise cosine similarity between agents' `sig_probs` to compute `R_div`.
- Combine `sig_probs` via `mean` (average) or `max` (union).
- Sample actions from combined probabilities.
- Compute task reward, and calculate final R: `R = task_reward + lambda_div * R_div`.
- During evaluation, also use the combined outputs to compute `probs_importance`.
- Add dictionary/list structures to collect `reward`, `diversity_penalty`, `agent_similarity`, and `recall` per epoch.
- Save these metrics to `logs/multi_agent_metrics.csv`.

#### [MODIFY] utils.py (Optional)
- Add any helper functions if needed (e.g., metric saving, but can also be in `main.py`).

## Open Questions

- Should the similarity measure for the diversity penalty be Cosine Similarity of probabilities, or a different metric (e.g. L2 distance, or Jaccard similarity of sampled discrete actions)? Cosine similarity of continuous probabilities is differentiable and widely used.
- For `--shared_params`: Do you prefer true/false boolean or a specific string flag? I'll use a boolean `--shared_params` flag.

## Verification Plan

### Automated Tests
- Run `main.py` with `--training --epochs 2 --num_agents 3` and verify the script runs without errors, multi-agent losses are computed, and `logs/multi_agent_metrics.csv` is correctly created and populated.
