# CGAL — Consensus-Gated Associative Learning (Experimental)

This repository tracks a research experiment: adding CGAL-inspired modifications to [tbp.monty](https://github.com/thousandbrainsproject/tbp.monty) and measuring whether they produce empirical advantages over baseline Monty on continual-learning, noise-robustness, and few-shot tasks.

## 🎯 Current Status: Ready for Monty Integration

**✅ Completed:**
- All 4 CGAL mechanisms implemented and tested (Issues #2-5)
- 126 unit tests, all passing
- Comprehensive documentation (75+ pages)
- Experiment infrastructure designed (Issue #6)
- Results writeup with honest assessment (Issue #7)

**📍 Next Step:** Issue #1 - Fork tbp.monty and integrate CGAL mechanisms

**👉 Start Here:** [MONTY_INTEGRATION_QUICKSTART.md](MONTY_INTEGRATION_QUICKSTART.md) (5-minute overview)

## Overview

CGAL (Consensus-Gated Associative Learning) proposes that voting consensus between cortical columns can serve as a credit-assignment signal, replacing the role backpropagation plays in deep learning. This experiment tests a minimal version of that claim by adding four mechanisms to Monty:

1. **Consensus-gated plasticity** — learning-rate modulation based on agreement with network-wide consensus.
2. **Novelty detection from hypothesis distributions** — entropy/peak-confidence of the hypothesis distribution gates new pattern allocation.
3. **Learned trust weights** — per-pair LM trust updated based on voting-round alignment with consensus.
4. **Salience-tagged replay** — off-line replay phase prioritising high-salience (consensus-confirmed or novel) patterns.

All modifications live behind config flags so the fork can run in pure-baseline mode at any time.

## 🚀 Quick Start for Monty Integration

### Option A: Quick Start (30 minutes)
Read [MONTY_INTEGRATION_QUICKSTART.md](MONTY_INTEGRATION_QUICKSTART.md) for a condensed guide to get started immediately.

### Option B: Comprehensive Guide (2-3 hours)
Read [MONTY_INTEGRATION_GUIDE.md](MONTY_INTEGRATION_GUIDE.md) for complete step-by-step instructions with code examples.

### Integration Steps Summary:
1. **Fork** `tbp.monty` to your organization
2. **Copy** CGAL modules from this repo to your fork
3. **Add** configuration parameters (5 lines)
4. **Integrate** mechanisms into learning loop (4 phases)
5. **Test** at each step (unit → integration → experiment)
6. **Run** validation experiments
7. **Report** findings honestly

**Estimated Time:** 2-3 weeks for a developer familiar with Monty

## 📚 Repository Structure

```
.
├── src/cgal/              # Standalone CGAL implementations (Issues #2-5)
│   ├── config/            # Configuration classes
│   └── learning_modules/  # Core mechanism implementations
├── tests/                 # 126 unit tests (all passing)
├── examples/              # Working demonstrations of each mechanism
├── experiments/           # Experiment harness infrastructure
├── docs/
│   ├── MONTY_INTEGRATION_GUIDE.md      # Full integration guide (30+ pages)
│   ├── MONTY_INTEGRATION_QUICKSTART.md # Quick start (5 minutes)
│   ├── ISSUE_2_IMPLEMENTATION.md       # Consensus gating details
│   ├── ISSUE_3_IMPLEMENTATION.md       # Novelty detection details
│   ├── ISSUE_4_IMPLEMENTATION.md       # Trust weights details
│   ├── ISSUE_5_IMPLEMENTATION.md       # Salience replay details
│   └── CGAL_RESULTS.md                 # Current status and findings
└── README.md             # This file
```

**Note:** This is a **tracking repository** with standalone CGAL prototypes. The actual integration with Monty happens in a separate fork of `tbp.monty` (to be created).

## 📊 Implementation Status

| Issue | Status | Tests | Documentation |
|-------|--------|-------|---------------|
| #1 Setup & Integration | 📋 **Ready to Start** | - | ✅ [Integration Guide](MONTY_INTEGRATION_GUIDE.md) |
| #2 Consensus-Gated Plasticity | ✅ Complete | 31/31 ✅ | ✅ [Details](ISSUE_2_IMPLEMENTATION.md) |
| #3 Novelty Detection | ✅ Complete | 31/31 ✅ | ✅ [Details](ISSUE_3_IMPLEMENTATION.md) |
| #4 Learned Trust Weights | ✅ Complete | 32/32 ✅ | ✅ [Details](ISSUE_4_IMPLEMENTATION.md) |
| #5 Salience-Tagged Replay | ✅ Complete | 32/32 ✅ | ✅ [Details](ISSUE_5_IMPLEMENTATION.md) |
| #6 Experiment Harness | ✅ Complete | - | ✅ Included in guide |
| #7 Results Writeup | ✅ Complete | - | ✅ [CGAL_RESULTS.md](CGAL_RESULTS.md) |

**Total:** 126 tests, all passing ✅

## 🎯 Hypotheses to Test

| ID | Claim | Expected | Status |
|----|-------|----------|--------|
| H1 | CGAL-Monty exhibits less catastrophic interference | Lower accuracy drop on early objects after new objects added | 🔄 Awaiting integration |
| H2 | Trust weights preserve accuracy under module noise | Smaller accuracy drop when 30% of LMs get noisy input | 🔄 Awaiting integration |
| H3 | Consensus gating improves sample efficiency | 1.5–3× fewer observations to reach accuracy threshold | 🔄 Awaiting integration |
| H4 | No regression on standard YCB task | Within ±2% of baseline accuracy | 🔄 Awaiting integration |

## 🛠️ For Implementers

### What's Ready
✅ **All CGAL mechanisms implemented** - Production-ready code with comprehensive tests
✅ **Integration guide complete** - Step-by-step instructions with code examples
✅ **Configuration approach defined** - Feature flags for baseline preservation
✅ **Test strategy documented** - Unit, integration, and experiment tests
✅ **Example scripts working** - Demonstrations of each mechanism

### What's Needed
📋 **Fork tbp.monty** - Create experimental fork for CGAL modifications
📋 **Integrate mechanisms** - Follow integration guide to add CGAL to Monty
📋 **Run experiments** - Test hypotheses H1-H4 empirically
📋 **Report findings** - Document results honestly (positive or negative)

### Getting Started
1. Read [MONTY_INTEGRATION_QUICKSTART.md](MONTY_INTEGRATION_QUICKSTART.md)
2. Review standalone implementations in `src/cgal/`
3. Run example scripts to understand each mechanism
4. Follow integration guide to add to Monty
5. Run experiments and report findings

## 📖 Documentation

### Integration Resources
- **[Quick Start](MONTY_INTEGRATION_QUICKSTART.md)** - Get started in 30 minutes
- **[Full Guide](MONTY_INTEGRATION_GUIDE.md)** - Complete integration instructions
- **[Implementation Plan](IMPLEMENTATION_PLAN.md)** - Original project plan

### Mechanism Documentation
- **[Consensus Gating](ISSUE_2_IMPLEMENTATION.md)** - Learning rate modulation (15 pages)
- **[Novelty Detection](ISSUE_3_IMPLEMENTATION.md)** - Pattern novelty scoring (12 pages)
- **[Trust Weights](ISSUE_4_IMPLEMENTATION.md)** - Inter-module trust learning (18 pages)
- **[Salience Replay](ISSUE_5_IMPLEMENTATION.md)** - Pattern consolidation (21 pages)

### Project Status
- **[Results Writeup](CGAL_RESULTS.md)** - Current status, findings, and next steps (23 pages)

## 🔬 Related Work

- [tbp.monty](https://github.com/thousandbrainsproject/tbp.monty) — Base repository (MIT licensed)
- Leadholm et al. 2025, arXiv:2507.04494 — Monty paper
- CGAL framework documentation (internal)

## 🆘 Getting Help

### Integration Support
- **Common issues:** See [Integration Guide - Common Issues](MONTY_INTEGRATION_GUIDE.md#common-issues-and-solutions)
- **Code examples:** Check `src/cgal/` for reference implementations
- **Test examples:** See `tests/test_*.py` for usage patterns

### Multi-Agent Assistant
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

## 📄 License

MIT License - See [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

This is an experimental research project testing CGAL mechanisms in the Monty framework. The standalone implementations demonstrate the concepts work correctly; integration with Monty will test their effectiveness in practice.

---

**Ready to integrate?** Start with [MONTY_INTEGRATION_QUICKSTART.md](MONTY_INTEGRATION_QUICKSTART.md) 🚀
