#!/usr/bin/env bash
# Drive a comparison matrix of Monty experiments, running each twice
# (baseline vs. CGAL-enabled), with OTel instrumentation, then call the
# report generator to emit a markdown comparison.
#
# Env:
#   SUITE        Which preset matrix to run (default: smoke).
#                  smoke  - one fast experiment, both arms        (~minutes)
#                  noise  - clean + 2 noise variants              (recommended)
#                  full   - 5-experiment matrix below             (long)
#   EXPERIMENTS  Override SUITE with an explicit space-separated list.
#   N_EVAL_EPOCHS  Optional override for monty.experiment_args.n_eval_epochs.
#   RUN_ONLY     baseline | cgal | both (default: both)
set -euo pipefail

# Curated comparison matrix. Each row is a Hydra experiment name from
# src/tbp/monty/conf/experiment/. The choice exercises the four CGAL
# mechanisms across the regimes where they should matter most:
#
#   randrot_10distinctobj_surf_agent          - clean / sanity (parity expected)
#   randrot_noise_10distinctobj_surf_agent    - sensor noise   (consensus gating)
#   randrot_noise_10simobj_surf_agent         - similar objs   (novelty + consensus)
#   randrot_noise_10distinctobj_5lms_dist_agent - multi-LM     (trust weights)
#   surf_agent_unsupervised_10distinctobj_noise - continual    (salience replay)
#
declare -A SUITES=(
    [smoke]="randrot_noise_10distinctobj_surf_agent"
    [noise]="randrot_10distinctobj_surf_agent randrot_noise_10distinctobj_surf_agent randrot_noise_10simobj_surf_agent"
    [full]="randrot_10distinctobj_surf_agent randrot_noise_10distinctobj_surf_agent randrot_noise_10simobj_surf_agent randrot_noise_10distinctobj_5lms_dist_agent surf_agent_unsupervised_10distinctobj_noise"
)

SUITE="${SUITE:-smoke}"
RUN_ONLY="${RUN_ONLY:-both}"
EXPERIMENTS="${EXPERIMENTS:-${SUITES[$SUITE]:-}}"
if [[ -z "${EXPERIMENTS}" ]]; then
    echo "Unknown SUITE='${SUITE}'. Choices: ${!SUITES[*]}" >&2
    exit 2
fi

EXTRA=()
[[ -n "${N_EVAL_EPOCHS:-}" ]] && EXTRA+=("++monty.experiment_args.n_eval_epochs=${N_EVAL_EPOCHS}")

cd /workspace/monty
export PYTHONPATH="/workspace:${PYTHONPATH:-}"

run_one() {
    local exp="$1"; shift
    local arm="$1"; shift
    local cgal="$1"; shift
    local logdir="/logs/${exp}/${arm}"
    mkdir -p "${logdir}"

    echo "================================================================"
    echo "[matrix] experiment=${exp} arm=${arm} (CGAL_ENABLED=${cgal})"
    echo "================================================================"

    local overrides=("experiment=${exp}" "${EXTRA[@]}" "++monty_logs_dir=${logdir}")
    if [[ "${cgal}" == "true" ]]; then
        overrides+=(
            "++monty.learning_module_configs.learning_module_0.learning_module_args.cgal_consensus_gating=true"
            "++monty.learning_module_configs.learning_module_0.learning_module_args.cgal_novelty_detection=true"
            "++monty.learning_module_configs.learning_module_0.learning_module_args.cgal_salience_replay=true"
            "++monty.learning_module_configs.learning_module_0.learning_module_args.cgal_trust_weights=true"
        )
    fi

    CGAL_ENABLED="${cgal}" EXPERIMENT="${exp}" ARM="${arm}" \
        python /workspace/run_with_otel.py "${overrides[@]}" \
        2>&1 | tee "${logdir}/run.log" || {
            echo "[matrix] !! ${exp}/${arm} FAILED -- continuing matrix" >&2
        }
}

for exp in ${EXPERIMENTS}; do
    [[ "${RUN_ONLY}" == "both" || "${RUN_ONLY}" == "baseline" ]] && run_one "${exp}" "baseline" "false"
    [[ "${RUN_ONLY}" == "both" || "${RUN_ONLY}" == "cgal"     ]] && run_one "${exp}" "cgal"     "true"
done

echo "[matrix] All runs complete. Generating report…"
python /workspace/generate_report.py \
    --logs-root /logs \
    --otel-dir /otel-out \
    --experiments ${EXPERIMENTS} \
    --output /logs/COMPARISON_REPORT.md

echo "[matrix] Report at /logs/COMPARISON_REPORT.md"
