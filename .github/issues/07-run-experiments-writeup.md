**Depends on:** issues 2, 3, 4, 5, and 6 — all must be complete.

## Description

Run the experiment harness, collect results, and produce a writeup documenting what was changed, what was tested, and what was found. The writeup should include both successes and failures, with honest interpretation.

## Acceptance criteria

- [ ] All four experiment types run with at least 5 seeds per condition (baseline + CGAL).
- [ ] Results saved in fork at `experiments/cgal_results/`.
- [ ] Plots generated for each experiment showing baseline vs CGAL accuracy with error bars.
- [ ] Writeup at `CGAL_RESULTS.md` in fork root containing:
  - Summary of changes made.
  - Description of each experiment.
  - Quantitative results table.
  - Per-hypothesis assessment (H1–H4 from the epic).
  - Honest interpretation including:
    - Which CGAL claims are empirically supported by these results.
    - Which are not supported.
    - Which require further experiments to assess.
    - Any unexpected findings.
  - Discussion of limitations (small scale, narrow benchmarks, single codebase).
  - Suggested next experiments.

## Implementation notes

- Honest interpretation is the most important deliverable. If CGAL doesn't help, say so plainly. If it helps in some conditions but not others, characterize where and speculate why. If results are noisy or inconclusive, say that — don't overclaim.
- The writeup should be readable by someone who hasn't seen the CGAL framework or the Monty codebase, with appropriate links to context.
- Include cost summary: total wall-clock compute, peak memory, any tooling that was needed.

## Notes for Copilot

Resist the temptation to oversell. The most useful writeup is one that clearly documents what was tested, what was found, and what would need to be different in a more rigorous follow-up. If the experiments fail to show an effect, that is *also* a useful result — it tells the framework where it needs to be revised. Treat null results with the same care as positive ones.
