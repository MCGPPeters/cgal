**Depends on:** issue 1 (Setup)

## Description

Modify Monty's evidence-based learning module so that the strength of pattern updates is modulated by the learning module's agreement with the network-wide voting consensus. This is the central CGAL claim: that *voting consensus serves as a credit-assignment signal*, replacing the role backpropagation plays in deep learning.

## Background

In current Monty, a learning module commits observations to its pattern graph based on its own internal logic (typically: when the agent has explored enough or moved sufficiently). Voting between modules is used to reach a network-wide consensus about object identity, but the *learning rule itself* is not gated by voting outcomes.

CGAL claims that voting can play a credit-assignment role: when a module's hypothesis agrees with consensus, that's evidence its observation was correctly attributed and reinforcement should proceed; when it disagrees, that's evidence the observation may be miscredited and reinforcement should be reduced (or the observation should be routed to a new pattern).

## Acceptance criteria

- [ ] A new config flag `consensus_gated_plasticity: bool` added to the LM config, defaulting to `False`.
- [ ] When the flag is `True`, the LM's update step applies a *gating factor* `g_m ∈ [0, 1]` to the magnitude of pattern updates, where `g_m` is computed from agreement with the current consensus.
- [ ] `g_m = α * agreement + (1-α) * baseline_rate`, where `α ∈ [0, 1]` is a config parameter (default 0.7), and `agreement` is a number in `[0, 1]` representing how well the LM's most-confident hypothesis matches the network consensus.
- [ ] When `consensus_gated_plasticity=False`, behavior should be byte-identical to baseline Monty (no regression).
- [ ] At least one unit test in `tests/unit/` verifying:
  - `g_m = 1` when LM perfectly agrees with consensus.
  - `g_m = 1-α` (the baseline rate) when LM completely disagrees.
  - Behavior reduces to baseline when flag is off.

## Implementation notes

- The relevant file is likely `src/tbp/monty/frameworks/models/evidence_matching/learning_module.py` (or similar — verify in issue 1).
- "Agreement with consensus" is computed as similarity between the LM's top hypothesis (object-id, pose) and the consensus (winning object-id, average winning pose). Suggested metric: 1.0 if same object-id and pose within tolerance; 0.5 if same object-id but different pose; 0.0 if different object-id. This is a starting heuristic — refine if the experiment results suggest the metric matters.
- The "current consensus" should be derived from the most recent voting round. If the LM has not yet voted (e.g., very early in an episode), default `g_m` to 1.0 (no gating yet — let the network learn before gating learning).
- Where exactly to insert the gating depends on Monty's update flow. Roughly: find the place where the LM commits an observation to its pattern graph (likely in a method called `add_observation` or `update_object_model` or similar) and multiply the update magnitude by `g_m`.
- Keep the modification *local*: don't refactor surrounding code unless necessary. Goal is minimal, reversible diff.

## Notes for Copilot

This is the most important sub-issue and the easiest to get wrong. Before writing code, **read the relevant LM code carefully** and identify exactly where pattern updates happen. The goal is a 50–100 line change, not a refactor. If the change requires touching more than ~3 files, stop and reconsider. Also: respect the project's CONTRIBUTING.md style guidelines.
