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
    # Stress suite: same base experiments, but Hydra overrides amplify the
    # failure modes each CGAL module is supposed to catch. We append an alias
    # suffix (-stress) to the experiment id so logs land in their own dir.
    [stress]="randrot_noise_10distinctobj_surf_agent-stress_highnoise randrot_noise_10simobj_surf_agent-stress_highnoise randrot_noise_10distinctobj_5lms_dist_agent-stress_asymmetric"
)

# Per-stress-id Hydra overrides. The key matches the alias used in SUITES,
# the value is a space-separated list of `++path=value` overrides applied to
# both arms (so baseline and CGAL see the *same* stressed environment).
#
# stress_highnoise  : 3x sensor noise on the patch SM (consensus + novelty pressure)
# stress_asymmetric : crank noise on patch_0 only in the 5-LM rig (trust weights pressure)
#
declare -A STRESS_OVERRIDES=(
    [stress_highnoise]="++experiment.config.monty_config.sensor_module_configs.sensor_module_0.sensor_module_args.noise_params.features.hsv=0.3 ++experiment.config.monty_config.sensor_module_configs.sensor_module_0.sensor_module_args.noise_params.features.principal_curvatures_log=0.3 ++experiment.config.monty_config.sensor_module_configs.sensor_module_0.sensor_module_args.noise_params.features.pose_vectors=5 ++experiment.config.monty_config.sensor_module_configs.sensor_module_0.sensor_module_args.noise_params.location=0.005"
    [stress_asymmetric]="++experiment.config.monty_config.sensor_module_configs.sensor_module_0.sensor_module_args.noise_params.features.hsv=0.4 ++experiment.config.monty_config.sensor_module_configs.sensor_module_0.sensor_module_args.noise_params.features.principal_curvatures_log=0.4 ++experiment.config.monty_config.sensor_module_configs.sensor_module_0.sensor_module_args.noise_params.features.pose_vectors=8 ++experiment.config.monty_config.sensor_module_configs.sensor_module_0.sensor_module_args.noise_params.location=0.008"
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
    local exp_id="$1"; shift
    local arm="$1"; shift
    local cgal="$1"; shift
    # exp_id may be either a plain experiment name or "BASE-stress_X". Split.
    local exp="${exp_id%%-stress_*}"
    local stress_id=""
    if [[ "${exp_id}" == *-stress_* ]]; then
        stress_id="stress_${exp_id##*-stress_}"
    fi
    local logdir="/logs/${exp_id}/${arm}"
    mkdir -p "${logdir}"

    echo "================================================================"
    echo "[matrix] experiment=${exp_id} arm=${arm} (CGAL_ENABLED=${cgal})${stress_id:+ stress=${stress_id}}"
    echo "================================================================"

    local overrides=("experiment=${exp}" "${EXTRA[@]}" "++monty_logs_dir=${logdir}")
    if [[ -n "${stress_id}" ]]; then
        # shellcheck disable=SC2206
        local stress_args=(${STRESS_OVERRIDES[$stress_id]:-})
        overrides+=("${stress_args[@]}")
    fi
    if [[ "${cgal}" == "true" ]]; then
        # The LM expects each cgal_* kwarg as a dict that becomes a *Config
        # dataclass. Each config also has a master enable bool that defaults
        # to False, so we must set it explicitly. The shorthand `++monty.…`
        # would land at a stray top-level `monty:` key that the LM never
        # reads (silently no-op'd two prior runs), so the *full* package path
        # is required.
        #
        # Conservative defaults (post-stress regression). The original defaults
        # (alpha=0.7, trust_min=0.1, gamma=0.05) over-corrected on the 5LM
        # asymmetric-noise stress (-2.2% accuracy). New defaults: smaller alpha
        # (mostly pass plasticity through), wider agreement tolerance, higher
        # trust floor (cap down-weighting at 2x), slower trust learning.
        local LM_ARGS="++experiment.config.monty_config.learning_module_configs.learning_module_0.learning_module_args"
        overrides+=(
            "${LM_ARGS}.cgal_consensus_gating={consensus_gated_plasticity:true,alpha:0.3,agreement_tolerance:0.2}"
            "${LM_ARGS}.cgal_novelty_detection={hypothesis_novelty_detection:true}"
            "${LM_ARGS}.cgal_salience_replay={salience_tagged_replay:true}"
            "${LM_ARGS}.cgal_trust_weights={learned_trust_weights:true,trust_learning_rate:0.02,trust_min:0.5}"
        )
    fi

    CGAL_ENABLED="${cgal}" EXPERIMENT="${exp_id}" ARM="${arm}" \
        python /workspace/run_with_otel.py "${overrides[@]}" \
        2>&1 | tee "${logdir}/run.log" || {
            echo "[matrix] !! ${exp_id}/${arm} FAILED -- continuing matrix" >&2
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
