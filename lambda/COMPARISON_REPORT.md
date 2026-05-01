# Monty baseline vs. CGAL comparison report

Generated automatically by `generate_report.py`. Each experiment
was run twice — once with the stock learning module and once with
all four CGAL mixins enabled (`consensus_gating`, `novelty_detection`,
`salience_replay`, `trust_weights`).

## Summary

| Experiment | Arm | Episodes | Accuracy | Mean steps | Mean time (s) |
| --- | --- | ---: | ---: | ---: | ---: |
| randrot_10distinctobj_surf_agent | baseline | 100 | 1.000 | 29.420 | 1.869 |
| randrot_10distinctobj_surf_agent | cgal | 100 | 1.000 | 29.420 | 1.870 |
| randrot_noise_10distinctobj_surf_agent | baseline | 100 | 1.000 | 72.940 | 6.543 |
| randrot_noise_10distinctobj_surf_agent | cgal | 100 | 1.000 | 72.940 | 6.581 |
| randrot_noise_10simobj_surf_agent | baseline | 100 | 0.970 | 155.740 | 14.219 |
| randrot_noise_10simobj_surf_agent | cgal | 100 | 0.970 | 155.740 | 14.296 |
| randrot_noise_10distinctobj_5lms_dist_agent | baseline | 500 | 0.998 | 20.654 | 6.533 |
| randrot_noise_10distinctobj_5lms_dist_agent | cgal | 500 | 0.998 | 20.654 | 6.483 |
| surf_agent_unsupervised_10distinctobj_noise | baseline | n/a | n/a | n/a | n/a |
| surf_agent_unsupervised_10distinctobj_noise | cgal | n/a | n/a | n/a | n/a |

## Deltas (cgal − baseline)

| Experiment | Δ accuracy | Δ mean steps | Δ mean time (s) |
| --- | ---: | ---: | ---: |
| randrot_10distinctobj_surf_agent | +0.000 | +0.000 | +0.002 |
| randrot_noise_10distinctobj_surf_agent | +0.000 | +0.000 | +0.038 |
| randrot_noise_10simobj_surf_agent | +0.000 | +0.000 | +0.077 |
| randrot_noise_10distinctobj_5lms_dist_agent | +0.000 | +0.000 | -0.049 |
| surf_agent_unsupervised_10distinctobj_noise | n/a | n/a | n/a |

## randrot_10distinctobj_surf_agent

### baseline

| Metric | Value |
| --- | --- |
| episodes | 100 |
| stats_path | /home/ubuntu/cgal/docker/logs/randrot_10distinctobj_surf_agent/baseline/eval_stats.csv |
| accuracy | 1.000 |
| n_correct | 100 |
| mean_num_steps | 29.420 |
| mean_monty_matching_steps | 29.420 |
| mean_time | 1.869 |

### cgal

| Metric | Value |
| --- | --- |
| episodes | 100 |
| stats_path | /home/ubuntu/cgal/docker/logs/randrot_10distinctobj_surf_agent/cgal/eval_stats.csv |
| accuracy | 1.000 |
| n_correct | 100 |
| mean_num_steps | 29.420 |
| mean_monty_matching_steps | 29.420 |
| mean_time | 1.870 |

## randrot_noise_10distinctobj_surf_agent

### baseline

| Metric | Value |
| --- | --- |
| episodes | 100 |
| stats_path | /home/ubuntu/cgal/docker/logs/randrot_noise_10distinctobj_surf_agent/baseline/eval_stats.csv |
| accuracy | 1.000 |
| n_correct | 100 |
| mean_num_steps | 72.940 |
| mean_monty_matching_steps | 72.940 |
| mean_time | 6.543 |

### cgal

| Metric | Value |
| --- | --- |
| episodes | 100 |
| stats_path | /home/ubuntu/cgal/docker/logs/randrot_noise_10distinctobj_surf_agent/cgal/eval_stats.csv |
| accuracy | 1.000 |
| n_correct | 100 |
| mean_num_steps | 72.940 |
| mean_monty_matching_steps | 72.940 |
| mean_time | 6.581 |

## randrot_noise_10simobj_surf_agent

### baseline

| Metric | Value |
| --- | --- |
| episodes | 100 |
| stats_path | /home/ubuntu/cgal/docker/logs/randrot_noise_10simobj_surf_agent/baseline/eval_stats.csv |
| accuracy | 0.970 |
| n_correct | 97 |
| mean_num_steps | 155.740 |
| mean_monty_matching_steps | 155.740 |
| mean_time | 14.219 |

### cgal

| Metric | Value |
| --- | --- |
| episodes | 100 |
| stats_path | /home/ubuntu/cgal/docker/logs/randrot_noise_10simobj_surf_agent/cgal/eval_stats.csv |
| accuracy | 0.970 |
| n_correct | 97 |
| mean_num_steps | 155.740 |
| mean_monty_matching_steps | 155.740 |
| mean_time | 14.296 |

## randrot_noise_10distinctobj_5lms_dist_agent

### baseline

| Metric | Value |
| --- | --- |
| episodes | 500 |
| stats_path | /home/ubuntu/cgal/docker/logs/randrot_noise_10distinctobj_5lms_dist_agent/baseline/eval_stats.csv |
| accuracy | 0.998 |
| n_correct | 499 |
| mean_num_steps | 20.654 |
| mean_monty_matching_steps | 54.790 |
| mean_time | 6.533 |

### cgal

| Metric | Value |
| --- | --- |
| episodes | 500 |
| stats_path | /home/ubuntu/cgal/docker/logs/randrot_noise_10distinctobj_5lms_dist_agent/cgal/eval_stats.csv |
| accuracy | 0.998 |
| n_correct | 499 |
| mean_num_steps | 20.654 |
| mean_monty_matching_steps | 54.790 |
| mean_time | 6.483 |

## surf_agent_unsupervised_10distinctobj_noise

### baseline

_No `eval_stats.csv` produced — see run.log._

### cgal

_No `eval_stats.csv` produced — see run.log._
