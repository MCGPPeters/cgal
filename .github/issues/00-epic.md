Fork [tbp.monty](https://github.com/thousandbrainsproject/tbp.monty) and add a set of modifications inspired by the CGAL framework (Consensus-Gated Associative Learning) to test whether a small set of additions to Monty's existing voting mechanism improves continual-learning, robustness to noisy modules, few-shot learning, and sample efficiency.

## Background

Monty already implements:
- Reference-frame-based learning modules (LMs).
- Hypothesis generation via graph-based pattern matching.
- Voting consensus between LMs via the Cortical Messaging Protocol (CMP).
- Hebbian-like associative learning.

CGAL proposes to extend this with:
1. **Plasticity gated by agreement with voting consensus** — instead of always committing observations, modulate the strength of pattern updates based on whether the LM's hypothesis agreed with the network-wide consensus.
2. **Novelty detection from hypothesis-distribution shape** — use the entropy/peak-confidence of the hypothesis distribution to gate whether observations are routed to existing patterns or trigger new pattern allocation.
3. **Learned trust weights between LMs** — adjust the weight given to each neighboring LM's votes based on how often that neighbor's votes track the winning consensus over time.
4. **Salience-tagged replay** — during off-line periods, prioritize replay of patterns tagged with high salience (consensus-confirmed, surprising, or rewarded) to deepen consolidation of valued sequences.

These claims correspond to specific architectural commitments in the CGAL framework. This experiment tests whether they produce measurable improvements when bolted onto Monty.

## Hypotheses (predictions before running)

- **H1 (Continual learning):** CGAL-Monty will exhibit less catastrophic interference than baseline Monty when objects are added incrementally over training time. Expected: lower drop in accuracy on early objects after late objects are added.
- **H2 (Noise robustness):** When a subset of LMs is fed noisy inputs, CGAL-Monty's trust-weight mechanism will down-weight those modules and preserve overall accuracy better than baseline. Expected: smaller accuracy drop under noise.
- **H3 (Sample efficiency):** CGAL-Monty will reach a target accuracy with fewer training observations per object due to consensus-gated reinforcement focusing learning on consistent patterns. Expected: 1.5–3x fewer observations to threshold.
- **H4 (No regression on baseline task):** CGAL-Monty should not perform worse than baseline Monty on the standard YCB classification task in the absence of noise or continual-learning pressure. Expected: within ±2% of baseline accuracy.

If H4 fails (CGAL hurts baseline performance), the framework needs revisiting before pursuing further modifications. If H1–H3 all fail, the central CGAL claim about consensus-as-credit-assignment is empirically weak and the framework needs revisiting. If some hypotheses hold and others don't, that's diagnostic — partial validation is the most informative outcome.

## Scope

**In scope:**
- Modifying Monty's evidence-based learning module to add the four mechanisms above.
- Adding configuration flags so each mechanism can be toggled independently.
- Running comparison experiments on YCB-based benchmarks.
- Producing a writeup of methodology and results.

**Out of scope (deferred to future experiments):**
- The carrier system / token-based short-term memory (CGAL Section 3.15, 3.18).
- Temporal correlation machinery (CGAL Section 3.16).
- Goal representation and the intent module (CGAL Section 3.19).
- Inhibition mechanisms beyond what's needed for hypothesis competition (CGAL Section 3.20).
- Multi-scale hierarchical carriers (CGAL Section 3.16d).

## Non-goals

- We are not trying to beat transformer-based ViT baselines on standard YCB classification. We are testing whether CGAL modifications improve over baseline *Monty* on continual-learning, robustness, and few-shot tasks specifically.
- We are not replicating CGAL's full architectural commitments. This is a minimal first test of the consensus-gated learning claim, not a full implementation of the framework.

## Deliverables

- A fork of `tbp.monty` with CGAL modifications behind config flags.
- Experiment configs for baseline and CGAL conditions.
- Result data and plots for the four hypotheses.
- A writeup (README in fork) describing what was changed, what was tested, and what was found.

## Sub-issues

<!-- Sub-issue links are posted as a comment by the create-cgal-issues workflow once all issues are created. -->
