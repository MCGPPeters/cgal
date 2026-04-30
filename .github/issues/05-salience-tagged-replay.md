**Depends on:** issue 1 (Setup); ideally also issues 2 and 3 (uses outputs from those)

## Description

Add a salience-tagging mechanism that boosts plasticity for high-confidence consensus events and high-novelty events, and use these tags during a designated "replay phase" to deepen consolidation of valued patterns. This implements CGAL's intrinsic-motivation coupling (Section 3.17) and replay-driven consolidation (Section 3.13).

## Background

CGAL claims that:
- Patterns that produce strong consensus or are flagged as novel should be tagged with high salience.
- During off-line phases (analog of sleep), the system should preferentially "replay" high-salience patterns, with replay driving Hebbian-like updates.
- This produces value-aligned consolidation: patterns associated with reliable network consensus or with novelty get deepened, while others fade through homeostatic downscaling.

## Acceptance criteria

- [ ] Config flag `salience_tagged_replay: bool` added, defaulting to `False`.
- [ ] Each pattern in the LM's graph stores a `salience: float` value, initialized to 0.0.
- [ ] On every observation, salience is incremented by `α_consensus * agreement_score + α_novelty * novelty_score` (config params, defaults 0.5 each), where the agreement and novelty scores are taken from issues 2 and 3 respectively.
- [ ] Salience decays over time: `salience *= decay_rate` per step (config param, default 0.99).
- [ ] At configurable intervals (every `replay_interval` episodes, default 10), the system runs an off-line *replay phase*:
  - Sample patterns weighted by salience (higher salience = more likely to be replayed).
  - For each replayed pattern, run the existing learning rule on it again (effectively boosting it).
  - This is independent of new observations — runs on stored pattern data only.
- [ ] Optionally apply *homeostatic downscaling* during the replay phase: multiply all pattern weights by `homeostatic_factor` (default 0.99) so that un-replayed patterns gradually weaken.
- [ ] When flag is `False`, no salience tracking or replay phase happens; behavior is byte-identical to baseline.
- [ ] Unit test: after several episodes with salience-tagged replay enabled, patterns with high salience should have stronger associations than low-salience patterns.

## Implementation notes

- This is the most architecturally novel change. Replay requires running the learning step on previously-seen data without new sensory input — this may not have a natural place in Monty's current loop, which is sensorimotor.
- Suggested implementation: add a `run_replay()` method on the LM that takes its own pattern graph as input and re-runs the update logic. Hook this into a new "between episodes" lifecycle event in the experiment runner.
- Salience decay can happen lazily (compute decayed value on read) or eagerly (update every step). Lazy is cleaner.
- If integrating salience-driven sampling into Monty's existing flow is awkward, an acceptable simpler version: add a separate "replay session" between training episodes that randomly re-trains on stored patterns weighted by salience.

## Notes for Copilot

This is the largest sub-issue. Don't try to be clever about integrating with Monty's lifecycle. A standalone replay phase that runs between episodes, on the LM's stored data, is acceptable. The cleaner architectural integration can come later if results justify pursuing it.
