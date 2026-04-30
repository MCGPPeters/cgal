# CGAL — Consensus-Gated Associative Learning (Experimental)

This repository tracks a research experiment: adding CGAL-inspired modifications to [tbp.monty](https://github.com/thousandbrainsproject/tbp.monty) and measuring whether they produce empirical advantages over baseline Monty on continual-learning, noise-robustness, and few-shot tasks.

## Overview

CGAL (Consensus-Gated Associative Learning) proposes that voting consensus between cortical columns can serve as a credit-assignment signal, replacing the role backpropagation plays in deep learning. This experiment tests a minimal version of that claim by adding four mechanisms to Monty:

1. **Consensus-gated plasticity** — learning-rate modulation based on agreement with network-wide consensus.
2. **Novelty detection from hypothesis distributions** — entropy/peak-confidence of the hypothesis distribution gates new pattern allocation.
3. **Learned trust weights** — per-pair LM trust updated based on voting-round alignment with consensus.
4. **Salience-tagged replay** — off-line replay phase prioritising high-salience (consensus-confirmed or novel) patterns.

All modifications live behind config flags so the fork can run in pure-baseline mode at any time.

## Quick Start

1. **Create Issues:** Trigger the [Create CGAL Issues workflow](../../actions/workflows/create-cgal-issues.yml) to generate all tracking issues
2. **Review Plan:** Read [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) for detailed implementation guidance
3. **Start Work:** Begin with Issue #1 (Setup) - fork tbp.monty and validate environment

## Issue tracking

GitHub Issues are used to track all work items. To create the full set of issues (epic + 7 sub-issues) in this repository, trigger the **Create CGAL Issues** workflow manually from the [Actions tab](../../actions/workflows/create-cgal-issues.yml).

The issue body templates live in [`.github/issues/`](.github/issues/).

## Repository structure

```
.
├── .github/
│   ├── issues/          # Markdown templates for each GitHub issue
│   └── workflows/
│       └── create-cgal-issues.yml   # One-shot workflow: creates labels + all issues
├── cgal_squad.py        # Multi-agent assistant for CGAL research
├── IMPLEMENTATION_PLAN.md  # Detailed implementation guide
└── README.md            # This file
```

**Note:** This is a **tracking repository**. The actual CGAL modifications to Monty will be implemented in a separate fork of `tbp.monty` (created in Issue #1).

## Hypotheses

| ID | Claim | Expected |
|----|-------|----------|
| H1 | CGAL-Monty exhibits less catastrophic interference | Lower accuracy drop on early objects after new objects added |
| H2 | Trust weights preserve accuracy under module noise | Smaller accuracy drop when 30% of LMs get noisy input |
| H3 | Consensus gating improves sample efficiency | 1.5–3× fewer observations to reach accuracy threshold |
| H4 | No regression on standard YCB task | Within ±2% of baseline accuracy |

## Related work

- [tbp.monty](https://github.com/thousandbrainsproject/tbp.monty) — base repository (MIT licensed)
- Leadholm et al. 2025, arXiv:2507.04494 — Monty paper
- CGAL framework documentation (internal)

## Implementation Resources

- **[IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md)** - Comprehensive guide covering all phases, mechanisms, and deliverables
- **[.github/issues/](.github/issues/)** - Templates for epic and 7 sub-issues
- **[cgal_squad.py](cgal_squad.py)** - Multi-agent assistant for specialized help (theorist, neuroscientist, Monty engineer, experiment designer, writer)

## Getting Help

The `cgal_squad.py` script provides access to specialized agents:

```bash
pip install "agent-squad[anthropic]"
export ANTHROPIC_API_KEY=your-key
python cgal_squad.py
```

Available specialists:
- **Theorist** - CGAL framework theory and architecture
- **Neuroscientist** - Empirical neuroscience grounding
- **Monty-engineer** - tbp.monty codebase implementation
- **Experiment-designer** - Experiment design and statistical analysis
- **Writer** - Documentation and writeup assistance