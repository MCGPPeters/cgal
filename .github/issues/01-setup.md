**Depends on:** none (do this first)

## Description

Fork [tbp.monty](https://github.com/thousandbrainsproject/tbp.monty) and create a development branch for CGAL modifications. Set up the Python environment per the project's [getting-started guide](https://thousandbrainsproject.readme.io/docs/getting-started). Validate that the baseline Monty experiments reproduce expected results before any modifications.

## Acceptance criteria

- [ ] Fork created at `<your-org>/tbp.monty.cgal` or similar.
- [ ] Branch `cgal/main` created off `main`.
- [ ] Python environment `tbp.monty` set up successfully on a Linux/macOS dev machine with conda.
- [ ] Pretrained YCB models downloaded to `~/tbp/results/monty/pretrained_models/pretrained_ycb_v12/`.
- [ ] At least one baseline experiment runs to completion (suggested: `randrot_noise_10distinctobj_surf_agent`) and produces results matching documented expectations within ±2% accuracy.
- [ ] CONTRIBUTING.md guidelines reviewed; code style (ruff, formatting) configured locally.
- [ ] A `CGAL_NOTES.md` file added to the fork's root explaining this is an experimental fork, with link back to the epic issue.

## Implementation notes

- The repo is at `https://github.com/thousandbrainsproject/tbp.monty`. It is MIT-licensed, so forking and modification is unrestricted.
- Use conda, not uv (uv is currently experimental per the README).
- The codebase is Python 3.10+. Key directories:
  - `src/tbp/monty/frameworks/models/` — core model classes including learning modules.
  - `src/tbp/monty/conf/experiment/` — experiment configs.
  - `tests/` — unit tests.
- Baseline reproduction is mainly a sanity check that the environment works. If results diverge significantly from the published paper (`Leadholm et al. 2025, arXiv:2507.04494`), stop and investigate before proceeding.

## Notes for Copilot

This issue is mostly setup and doesn't require deep code changes. The main deliverable is a working environment plus a brief reproduction check. If you encounter platform-specific issues (e.g., habitat-sim install failures), document them in `CGAL_NOTES.md` rather than working around them silently.
