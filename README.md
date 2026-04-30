# CGAL — Consensus-Gated Associative Learning (Experimental)

This repository tracks a research experiment: adding CGAL-inspired modifications to [tbp.monty](https://github.com/thousandbrainsproject/tbp.monty) and measuring whether they produce empirical advantages over baseline Monty on continual-learning, noise-robustness, and few-shot tasks.

## Overview

CGAL (Consensus-Gated Associative Learning) proposes that voting consensus between cortical columns can serve as a credit-assignment signal, replacing the role backpropagation plays in deep learning. This experiment tests a minimal version of that claim by adding four mechanisms to Monty:

1. **Consensus-gated plasticity** — learning-rate modulation based on agreement with network-wide consensus.
2. **Novelty detection from hypothesis distributions** — entropy/peak-confidence of the hypothesis distribution gates new pattern allocation.
3. **Learned trust weights** — per-pair LM trust updated based on voting-round alignment with consensus.
4. **Salience-tagged replay** — off-line replay phase prioritising high-salience (consensus-confirmed or novel) patterns.

All modifications live behind config flags so the fork can run in pure-baseline mode at any time.

## Issue tracking

GitHub Issues are used to track all work items. To create the full set of issues (epic + 7 sub-issues) in this repository, trigger the **Create CGAL Issues** workflow manually from the [Actions tab](../../actions/workflows/create-cgal-issues.yml).

The issue body templates live in [`.github/issues/`](.github/issues/).

## Repository structure

```
.github/
  issues/          # Markdown templates for each GitHub issue
  workflows/
    create-cgal-issues.yml   # One-shot workflow: creates labels + all issues
```

## Hypotheses

| ID | Claim | Expected |
|----|-------|----------|
| H1 | CGAL-Monty exhibits less catastrophic interference | Lower accuracy drop on early objects after new objects added |
| H2 | Trust weights preserve accuracy under module noise | Smaller accuracy drop when 30% of LMs get noisy input |
| H3 | Consensus gating improves sample efficiency | 1.5–3× fewer observations to reach accuracy threshold |
| H4 | No regression on standard YCB task | Within ±2% of baseline accuracy |

## Related work

- [tbp.monty](https://github.com/thousandbrainsproject/tbp.monty) — base repository (MIT licensed).
- Leadholm et al. 2025, arXiv:2507.04494 — Monty paper.
- CGAL framework documentation (internal).