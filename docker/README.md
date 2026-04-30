# Monty + CGAL benchmark stack (Docker + OTel)

Runs Monty YCB benchmarks twice — stock LM vs. all four CGAL mixins
enabled — inside Docker with an OpenTelemetry collector sidecar, then
auto-generates a markdown comparison report.

## Why these experiments?

CGAL adds four mechanisms (consensus gating, novelty detection, salience
replay, trust weights). The matrix below is chosen so each mechanism is
exercised in the regime where it should matter most:

| Experiment | Tests | CGAL mechanism most stressed |
| --- | --- | --- |
| `randrot_10distinctobj_surf_agent` | clean random rotations | sanity / parity |
| `randrot_noise_10distinctobj_surf_agent` | sensor noise | consensus gating |
| `randrot_noise_10simobj_surf_agent` | similar objects | novelty + consensus |
| `randrot_noise_10distinctobj_5lms_dist_agent` | multi-LM voting | trust weights |
| `surf_agent_unsupervised_10distinctobj_noise` | continual / unsupervised | salience replay |

The runner exposes three presets:

| `SUITE=` | Experiments | Use case |
| --- | --- | --- |
| `smoke`  | `randrot_noise_10distinctobj_surf_agent` only | quick validation (~minutes) |
| `noise`  | first three rows above | recommended default comparison |
| `full`   | all five rows | publication-quality matrix (long) |

Override entirely with `EXPERIMENTS="exp_a exp_b ..."`. Shrink any run
with `N_EVAL_EPOCHS=1`. Skip an arm with `RUN_ONLY=baseline|cgal`.

## Layout

| File | Purpose |
| --- | --- |
| [Dockerfile](Dockerfile) + [env.yml](env.yml) | micromamba image with `habitat-sim` + OTel SDK |
| [entrypoint.sh](entrypoint.sh) | editable-install Monty + CGAL on first run |
| [docker-compose.yml](docker-compose.yml) | Docker variant of the `monty` + `otel-collector` stack |
| [podman-compose.yml](podman-compose.yml) | Podman variant (rootless-friendly, SELinux relabels, no healthcheck) |
| [compose.sh](compose.sh) | Wrapper that auto-picks docker / podman-compose / podman compose |
| [otel-collector-config.yaml](otel-collector-config.yaml) | OTLP → stdout + JSONL files |
| [otel_shim.py](otel_shim.py) | monkey-patches `MontyExperiment` to emit spans/metrics |
| [run_with_otel.py](run_with_otel.py) | tiny launcher (loads shim, runs `run.py`) |
| [run_benchmark.sh](run_benchmark.sh) | drives the experiment matrix |
| [generate_report.py](generate_report.py) | builds `COMPARISON_REPORT.md` |
| [download_ycb.sh](download_ycb.sh) | host helper to fetch YCB + pretrained models |

## Prerequisites

1. The Monty fork checked out **next to** this repo:
   ```
   ~/RiderProjects/cgal/        ← this repo (mounted at /workspace/cgal)
   ~/RiderProjects/tbp.monty/   ← MCGPPeters/tbp.monty @ cgal/main (mounted at /workspace/monty)
   ```
2. YCB scenes + pretrained models on the host:
   ```bash
   ./download_ycb.sh
   ```
   Populates `~/tbp/data/habitat/...` and
   `~/tbp/results/monty/pretrained_models/...`.

## Run

Use the wrapper to auto-detect docker vs. podman:

```bash
cd docker
./compose.sh up --build                    # default: SUITE=smoke
SUITE=noise ./compose.sh up                # recommended comparison
SUITE=full N_EVAL_EPOCHS=2 ./compose.sh up
EXPERIMENTS="randrot_noise_10distinctobj_5lms_dist_agent" ./compose.sh up
```

Or call the tool directly:

```bash
docker compose -f docker-compose.yml up --build       # Docker
podman-compose -f podman-compose.yml up --build       # Podman (classic)
podman compose -f podman-compose.yml up --build       # Podman (native v5+)
```

Force a specific engine through the wrapper with `ENGINE=docker` or
`ENGINE=podman`. The two compose files are kept in sync; pick the one
that matches your installed runtime.

## Outputs

- Per-experiment per-arm Monty logs: `./logs/<experiment>/<baseline|cgal>/`
  (each contains `eval_stats.csv`, `train_stats.csv`, wandb dir, `run.log`).
- OTel JSONL: `./otel-out/traces.jsonl`, `./otel-out/metrics.jsonl`
  (also live on `docker compose logs otel-collector`).
- **`./logs/COMPARISON_REPORT.md`** — auto-generated after the matrix
  finishes. Sections:
  - **Summary** — one row per (experiment, arm) with episodes / accuracy /
    mean steps / mean time.
  - **Deltas** — `cgal − baseline` for the same metrics, per experiment.
  - Per-experiment expansions including `terminal_condition` counts and,
    when OTel is up, counter totals (`monty.episodes`,
    `monty.matching_steps`, `monty.terminal_conditions`).

## Notes

- The OTel shim is a no-op if `MontyExperiment` hooks aren't found —
  instrumentation never blocks the benchmark.
- `habitat-sim` is the headless build from the `aihabitat` channel; CPU
  works (slower) — no GPU required.
- The report degrades gracefully: missing arms show `n/a` rather than
  failing the run.
