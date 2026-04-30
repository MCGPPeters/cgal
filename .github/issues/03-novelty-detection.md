**Depends on:** issue 1 (Setup); can proceed in parallel with issue 2

## Description

Add a novelty signal to the learning module that's computed from the *shape* of the hypothesis distribution (entropy + peak confidence). When novelty is high, route observations to new pattern allocation rather than reinforcing existing patterns. This implements CGAL's claim (Section 3.8) that hypothesis-generation machinery already does novelty detection as a side effect.

## Background

CGAL claims:
- Sharp distribution + high peak confidence = familiar pattern.
- Broad distribution + low peak confidence = novel — allocate new pattern.
- Sharp distribution + low peak confidence = ambiguous (different from novel — wait for more evidence).

This collapses what would otherwise be a separate novelty-detection mechanism into the existing inference machinery.

## Acceptance criteria

- [ ] Config flag `hypothesis_novelty_detection: bool` added, defaulting to `False`.
- [ ] When flag is `True`, the LM computes a `novelty_score ∈ [0, 1]` per observation, defined as:
  - `novelty_score = entropy(hypothesis_distribution) * (1 - peak_confidence)`
  - normalized to `[0, 1]` (specify normalization in implementation).
- [ ] When `novelty_score > novelty_threshold` (config param, default 0.7), observations get routed to a new pattern allocation rather than added to existing patterns.
- [ ] When flag is `False`, behavior is byte-identical to baseline.
- [ ] Logs the novelty score per step so it can be inspected in experiment results.
- [ ] Unit test verifying novelty score is high for fresh inputs the LM has not seen and low for inputs that match an existing pattern strongly.

## Implementation notes

- The hypothesis distribution lives in the LM's evidence tracking. In Monty's `EvidenceGraphLM` or similar, there's typically a structure like `self.evidence` mapping (object_id, pose_hypothesis) to confidence values.
- Entropy: `-sum(p * log(p))` over the normalized distribution.
- Peak confidence: `max(p)` after normalization.
- Routing observations to new patterns: investigate how Monty currently handles unknown-object cases. There may already be a "new object" code path that can be triggered explicitly when novelty exceeds threshold.
- The novelty score should also be made *available* to downstream code (e.g., for use as a salience tag in the salience-tagged replay issue).

## Notes for Copilot

This change is more localized than issue 2. Most of the work is computing the right summary statistics over an existing distribution. The hardest part is figuring out where new-pattern allocation already happens in Monty and triggering it under the new condition. If new-pattern allocation isn't easily separable, document the obstacle in a code comment and proceed with a simpler version (e.g., just log the novelty score without yet using it for routing).
