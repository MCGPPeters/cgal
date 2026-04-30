# CGAL Implementation Plan

This document provides a detailed implementation plan for adding CGAL (Consensus-Gated Associative Learning) mechanisms to Monty.

## Overview

This experiment tests whether adding four consensus-based mechanisms to Monty improves its performance on continual learning, noise robustness, and sample efficiency tasks.

## Repository Structure

This repository (`MCGPPeters/cgal`) serves as the **tracking and planning repository** for the CGAL experiment. The actual implementation work will be done in a fork of `tbp.monty`.

```
cgal/ (this repo - tracking)
├── .github/
│   ├── issues/          # Issue templates for all sub-tasks
│   └── workflows/       # Workflow to create GitHub issues
├── README.md            # Project overview
├── cgal_squad.py        # Multi-agent assistant for CGAL work
└── IMPLEMENTATION_PLAN.md  # This file

tbp.monty.cgal/ (separate fork - implementation)
├── src/tbp/monty/
│   └── frameworks/models/  # Where CGAL modifications go
├── conf/experiment/cgal/    # CGAL experiment configs
└── experiments/cgal_results/  # Results from experiments
```

## Implementation Phases

### Phase 1: Setup (Issue #1)

**Goal:** Fork tbp.monty, set up environment, validate baseline reproduction.

**Deliverables:**
- Fork of tbp.monty at `<your-org>/tbp.monty.cgal`
- Branch `cgal/main` created
- Python environment working with conda
- Baseline experiment runs successfully
- `CGAL_NOTES.md` added to fork root

**Key Files:**
- None modified in tracking repo
- All work happens in the Monty fork

**Validation:**
- At least one baseline experiment completes
- Results match published paper within ±2% accuracy

### Phase 2: Core Mechanisms (Issues #2-5)

These can proceed in parallel after Phase 1 completes.

#### Issue #2: Consensus-Gated Plasticity

**Goal:** Modulate learning rate based on agreement with voting consensus.

**Config Additions:**
```python
consensus_gated_plasticity: bool = False  # Feature flag
consensus_alpha: float = 0.7              # Weight of consensus in gating
```

**Key Formula:**
```
g_m = α * agreement + (1-α) * baseline_rate
```

**Implementation Location:**
- `src/tbp/monty/frameworks/models/evidence_matching/learning_module.py`
- Modify pattern update method to apply gating factor `g_m`

**Tests:**
- Unit test verifying g_m=1 for perfect agreement
- Unit test verifying g_m=(1-α) for complete disagreement
- Integration test confirming baseline behavior when flag is off

#### Issue #3: Novelty Detection

**Goal:** Detect novelty from hypothesis distribution shape (entropy + peak confidence).

**Config Additions:**
```python
hypothesis_novelty_detection: bool = False
novelty_threshold: float = 0.7
```

**Key Formula:**
```
novelty_score = entropy(hypothesis_dist) * (1 - peak_confidence)
```

**Implementation Location:**
- Same learning module as Issue #2
- Add method to compute entropy and peak confidence from evidence distribution
- Route to new pattern allocation when novelty exceeds threshold

**Tests:**
- Unit test verifying high novelty for unseen inputs
- Unit test verifying low novelty for familiar patterns
- Verify logging of novelty scores

#### Issue #4: Learned Trust Weights

**Goal:** Learn which LMs are reliable voters over time.

**Config Additions:**
```python
learned_trust_weights: bool = False
trust_learning_rate: float = 0.05
trust_min: float = 0.1
```

**Key Formula:**
```
W[m,n] += γ * (agreement(n, consensus) - W[m,n])
W[m,n] = clip(W[m,n], trust_min, 1.0)
```

**Implementation Location:**
- `src/tbp/monty/frameworks/models/monty_base.py` or voting coordinator
- Add trust matrix storage at experiment level
- Update trust after each voting round
- Weight votes by trust in subsequent rounds

**Tests:**
- Unit test with deliberately broken LM showing trust decay
- Verify trust weights persist across episodes
- Verify baseline behavior with flag off

#### Issue #5: Salience-Tagged Replay

**Goal:** Prioritize consolidation of high-salience patterns during off-line replay.

**Config Additions:**
```python
salience_tagged_replay: bool = False
alpha_consensus: float = 0.5
alpha_novelty: float = 0.5
salience_decay: float = 0.99
replay_interval: int = 10  # episodes between replay phases
homeostatic_factor: float = 0.99
```

**Key Formula:**
```
salience += α_consensus * agreement + α_novelty * novelty
salience *= decay_rate  (per step)
```

**Implementation Location:**
- Add salience field to pattern storage in learning module
- Add `run_replay()` method to LM
- Hook replay phase into experiment runner between episodes

**Tests:**
- Unit test verifying salience updates correctly
- Test that high-salience patterns have stronger associations after replay
- Verify no replay happens when flag is off

### Phase 3: Experiment Harness (Issue #6)

**Goal:** Create experiment configurations for testing CGAL hypotheses.

**Deliverables:**
- Four experiment configs in `conf/experiment/cgal/`:
  1. `cgal_baseline_regression.py` - Standard YCB, all flags off
  2. `cgal_continual_learning.py` - Incremental object introduction
  3. `cgal_noise_robustness.py` - Corrupted input to subset of LMs
  4. `cgal_few_shot.py` - Limited observations per object
- Runner script `run_cgal_experiments.py`
- Plot generation script

**Experiment Details:**

1. **Baseline Regression (H4):**
   - Standard YCB classification
   - All CGAL flags off
   - Should match published baseline ±2%

2. **Continual Learning (H1):**
   - Introduce objects in groups (5 at a time)
   - Test accuracy on all previous objects after each group
   - Measure catastrophic forgetting
   - Run with CGAL off vs. on

3. **Noise Robustness (H2):**
   - Inject noise into 30% of LMs
   - Remaining 70% see clean data
   - Measure classification accuracy
   - Run with CGAL off vs. on

4. **Few-Shot Learning (H3):**
   - Limit observations per object (5, 10, 20)
   - Measure accuracy curves
   - Run with CGAL off vs. on

**Statistical Rigor:**
- At least 5 random seeds per condition
- Record mean and standard error
- Save all raw results for later analysis

### Phase 4: Run and Analyze (Issue #7)

**Goal:** Execute experiments and produce honest writeup.

**Deliverables:**
- Results saved in `experiments/cgal_results/`
- Plots comparing baseline vs CGAL
- `CGAL_RESULTS.md` writeup

**Writeup Structure:**
1. Summary of changes made
2. Description of each experiment
3. Quantitative results table
4. Per-hypothesis assessment (H1-H4)
5. Honest interpretation:
   - Which claims are supported
   - Which are not supported
   - Which need more testing
   - Unexpected findings
6. Limitations
7. Suggested next experiments

**Key Principle:** Honest reporting. Null results are as valuable as positive ones.

## Hypotheses to Test

| ID | Hypothesis | Metric | Expected Result |
|----|------------|--------|-----------------|
| H1 | CGAL reduces catastrophic forgetting | Accuracy on early objects after late objects added | Smaller drop in CGAL vs baseline |
| H2 | Trust weights preserve accuracy under noise | Classification accuracy with 30% noisy LMs | Smaller accuracy drop in CGAL vs baseline |
| H3 | Consensus gating improves sample efficiency | Observations needed to reach target accuracy | 1.5-3× fewer in CGAL vs baseline |
| H4 | No regression on baseline task | Accuracy on standard YCB | Within ±2% of baseline |

## Implementation Guidelines

### Code Style
- Follow tbp.monty's CONTRIBUTING.md guidelines
- Use ruff for formatting
- Keep changes minimal and reversible
- Avoid refactoring existing code

### Config Flag Pattern
All CGAL features must be behind boolean flags:
```python
# In config
cgal_config = {
    "consensus_gated_plasticity": False,
    "hypothesis_novelty_detection": False,
    "learned_trust_weights": False,
    "salience_tagged_replay": False,
}
```

When all flags are `False`, behavior must be byte-identical to baseline Monty.

### Testing Strategy
1. Unit tests for each mechanism in isolation
2. Integration tests verifying baseline behavior preserved
3. Experiment runs to validate hypotheses
4. Manual inspection of intermediate outputs

### Documentation
- Add inline comments explaining CGAL-specific logic
- Update `CGAL_NOTES.md` as implementation progresses
- Log important metrics (novelty scores, trust weights, salience) for debugging

## Out of Scope

The following CGAL features are deferred to future experiments:
- Carrier system / token-based short-term memory
- Temporal correlation machinery
- Goal representation and intent module
- Inhibition mechanisms beyond hypothesis competition
- Multi-scale hierarchical carriers

## Success Criteria

**Minimum Success:**
- All four mechanisms implemented behind config flags
- Baseline behavior preserved when flags are off
- All experiments run to completion
- Honest writeup produced (regardless of results)

**Strong Success:**
- H4 passes (no regression on baseline)
- At least 2 of H1-H3 show positive results
- Results are statistically significant (p < 0.05)
- Findings suggest productive next experiments

**Important:** Even if H1-H3 all fail, this is valuable! It tells us the consensus-gating claim needs revision. Null results deserve the same rigorous reporting as positive ones.

## Timeline Estimate

This is research work with unknown unknowns, so estimates are approximate:

- Issue #1 (Setup): 1-2 days
- Issues #2-4 (Core mechanisms): 3-5 days each (can parallelize)
- Issue #5 (Replay): 5-7 days (most architecturally novel)
- Issue #6 (Harness): 2-3 days
- Issue #7 (Run & writeup): 5-7 days (includes compute time)

**Total:** 3-5 weeks of focused work

## Dependencies

### External
- tbp.monty repository (MIT licensed)
- Python 3.10+
- conda for environment management
- habitat-sim (may have platform-specific issues)

### Internal
- Issue #1 must complete before any others
- Issues #2-4 can proceed in parallel after #1
- Issue #5 ideally waits for #2 and #3 (uses their outputs)
- Issue #6 can start once config interfaces are stable
- Issue #7 requires all of #2-6 complete

## Risks and Mitigations

### Risk: Baseline doesn't reproduce
**Mitigation:** Document discrepancy, proceed with fork's baseline as reference

### Risk: CGAL hurts baseline performance (H4 fails)
**Mitigation:** This is diagnostic - report honestly, investigate which mechanism causes regression

### Risk: Implementation requires major refactor
**Mitigation:** Start with simpler version, document obstacles, iterate if results justify

### Risk: Results are noisy/inconclusive
**Mitigation:** Report uncertainty clearly, suggest what larger-scale experiment would need

## Contact and Collaboration

- Repository: https://github.com/MCGPPeters/cgal
- Monty fork: (to be created in Issue #1)
- Issues: Track all work via GitHub issues created by workflow
- Agent assistance: Use `cgal_squad.py` for specialized help

## Appendix: File Locations

### In Tracking Repo (cgal)
- `.github/issues/*.md` - Issue templates
- `.github/workflows/create-cgal-issues.yml` - Issue creation workflow
- `README.md` - Project overview
- `cgal_squad.py` - Multi-agent assistant
- `IMPLEMENTATION_PLAN.md` - This document

### In Implementation Repo (tbp.monty.cgal fork)
- `src/tbp/monty/frameworks/models/evidence_matching/learning_module.py` - Issues #2, #3, #5
- `src/tbp/monty/frameworks/models/monty_base.py` - Issue #4
- `src/tbp/monty/conf/experiment/cgal/*.py` - Issue #6
- `experiments/cgal_results/` - Issue #7 outputs
- `CGAL_NOTES.md` - Implementation notes
- `CGAL_RESULTS.md` - Final writeup

## References

- [tbp.monty](https://github.com/thousandbrainsproject/tbp.monty) - Base repository
- Leadholm et al. 2025, arXiv:2507.04494 - Monty paper
- CGAL framework documentation (internal)
