**Depends on:** issue 1 (Setup); can be done in parallel with issues 2–5 once the configuration interfaces are stable.

## Description

Build experiment configurations and harness scripts to compare baseline Monty against CGAL-modified Monty on three task types:
1. Continual learning (objects added incrementally over training).
2. Noise robustness (subset of LMs given corrupted input).
3. Few-shot learning (very few observations per object).

Plus a baseline regression test (standard YCB classification) to verify CGAL modifications don't hurt non-stress-test performance.

## Acceptance criteria

- [ ] Four experiment configs in `src/tbp/monty/conf/experiment/cgal/`:
  - `cgal_baseline_regression.py` — standard YCB, all CGAL flags off (should match published baseline).
  - `cgal_continual_learning.py` — incremental object introduction, CGAL flags on/off via toggle.
  - `cgal_noise_robustness.py` — N% of LMs given corrupted input, CGAL flags on/off.
  - `cgal_few_shot.py` — limited observations per object, CGAL flags on/off.
- [ ] A runner script `run_cgal_experiments.py` that:
  - Runs each experiment in baseline and CGAL conditions.
  - Saves results with clear naming.
  - Records elapsed wall-clock time, accuracy, learned-trust-matrix evolution, and salience distributions.
- [ ] Each experiment specifies its random seeds (for reproducibility); each condition runs with at least 5 seeds.
- [ ] Output: for each experiment, a CSV or JSON file recording per-seed results.
- [ ] A simple plot script that produces comparison bar charts (baseline vs CGAL) with error bars.

## Implementation notes

- For continual learning: introduce objects in groups (e.g., 5 objects at a time, then 5 more, then 5 more). Test accuracy on *all* previously-seen objects after each group is added. Catastrophic forgetting shows up as accuracy drop on early objects after late objects are added.
- For noise robustness: pick a fraction (e.g., 30%) of LMs and inject noise into their feature observations. The remaining LMs see clean data. Measure object-classification accuracy.
- For few-shot: limit the number of observations per object during training (e.g., 5, 10, 20 observations). Compare accuracy curves between baseline and CGAL.
- Run sizes: keep each experiment small enough to complete in reasonable wall-clock time (under a few hours per run on a single machine). The point is to get directional results, not paper-final benchmarks.
- Use existing Monty experiment infrastructure (Hydra configs, the existing `run.py`/`run_parallel.py` scripts) rather than writing parallel infrastructure.

## Notes for Copilot

The main work here is config files plus a thin runner script. Don't reinvent Monty's experiment loop. Many of the YCB-related dataset utilities already exist in Monty — use them rather than rolling your own.
