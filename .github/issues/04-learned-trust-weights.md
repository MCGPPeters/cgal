**Depends on:** issue 1 (Setup); can proceed in parallel with issues 2 and 3

## Description

Add a per-pair trust weight `w_mn ∈ [0, 1]` for each pair of LMs that exchange votes. Update the trust weight over time based on how often LM `n`'s votes consistently track the winning consensus. Use trust weights to modulate vote influence in subsequent voting rounds. This implements CGAL's claim that the system should learn *who to listen to*.

## Background

In current Monty, voting between LMs treats all neighbors equally (or weights them by some fixed scheme). CGAL claims that the system should learn which LMs are reliable voters — modules whose votes consistently track winning consensus get more weight in future votes; modules whose votes consistently dissent (and lose) get less weight.

## Acceptance criteria

- [ ] Config flag `learned_trust_weights: bool` added, defaulting to `False`.
- [ ] A trust matrix `W: dict[(int, int), float]` (LM-id pairs to weights) initialized at 1.0 for all pairs.
- [ ] After each voting round, trust is updated: `W[m, n] += γ * (agreement(n, consensus) - W[m, n])`, where `γ` (config param, default 0.05) is the trust learning rate, and `agreement(n, consensus) ∈ [0, 1]` is how well LM `n`'s vote aligned with the winning consensus.
- [ ] Trust weights clipped to `[trust_min, 1.0]` where `trust_min` (config param, default 0.1) prevents complete silencing of any module.
- [ ] When voting, neighbor `n`'s vote is weighted by `W[m, n]` from m's perspective.
- [ ] When flag is `False`, behavior reduces to baseline (uniform unit weights).
- [ ] Logs trust weights periodically (every N episodes) for inspection.
- [ ] Unit test: a deliberately broken LM (returns random votes) should have its incoming trust weights from other LMs decay over time, while normal LMs maintain weights near 1.0.

## Implementation notes

- The voting mechanism in Monty is typically in a class handling inter-LM communication. Look for a class managing `MontyExperiment` or in `src/tbp/monty/frameworks/models/monty_base.py` or similar.
- Trust weights are *directional*: `W[m, n]` is m's trust in n, which can differ from n's trust in m.
- For the agreement function, a simple version: `agreement(n, consensus) = 1 if n's-top-hypothesis matches consensus else 0`. Refine if results suggest the function matters.
- Storage: a dict mapping (m_id, n_id) → float is fine for small numbers of LMs. If experiments scale to many LMs, consider a numpy 2D array indexed by LM-ids.

## Notes for Copilot

This change requires understanding Monty's voting flow more than the others. Before writing, trace through one voting cycle in the existing code to understand the data flow. The trust matrix is stored at the experiment level (or wherever the LM-network coordination happens), not per-LM, since it's about pairs.
