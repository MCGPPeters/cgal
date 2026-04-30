"""Head-to-head comparison: baseline EvidenceGraphLM vs CGAL-enabled.

This script drives two ``EvidenceGraphLM`` instances through the same synthetic
observation stream and reports a small set of metrics that highlight where
CGAL is expected to help:

* **Object allocation discipline**: under noisy / repeated exposures, the
  baseline tends to allocate a fresh ``new_object{n}`` whenever its terminal
  condition fires "no_match". CGAL's novelty gate suppresses those
  allocations when the hypothesis distribution is not actually novel
  (peaked).
* **Sample efficiency**: number of matching steps required to lock onto the
  correct object the second time it is shown.
* **Telemetry**: average gating factor, average novelty score, and number
  of dream-replay events triggered.

Why synthetic? The real Monty benchmarks (YCB / Habitat) require
``habitat-sim``, which fails to build on Apple Silicon + Python 3.13 in this
environment. The synthetic stream uses the same ``Message``/``RuntimeContext``
plumbing as ``tests/unit/frameworks/models/evidence_matching/evidence_lm_test.py``
so the two LMs see byte-identical inputs.

Usage::

    cd ~/RiderProjects/tbp.monty
    source .venv-cgal/bin/activate
    PYTHONPATH=src:tests python ~/RiderProjects/cgal/experiments/monty_cgal_comparison.py

Outputs ``cgal_vs_baseline_results.json`` next to this script.
"""

from __future__ import annotations

import copy
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# Ensure tbp.monty src and its tests/ helper module are importable.
_MONTY_ROOT = Path.home() / "RiderProjects" / "tbp.monty"
sys.path.insert(0, str(_MONTY_ROOT / "src"))
sys.path.insert(0, str(_MONTY_ROOT))

from tbp.monty.cmp import Message  # noqa: E402
from tbp.monty.frameworks.experiments.mode import ExperimentMode  # noqa: E402
from tbp.monty.frameworks.models.evidence_matching.learning_module import (  # noqa: E402
    EvidenceGraphLM,
)
from tests.unit.resources.unit_test_utils import BaseGraphTest  # noqa: E402


# ---------------------------------------------------------------------------
# LM construction
# ---------------------------------------------------------------------------

_TOLERANCES = {
    "patch": {
        "hsv": [0.1, 1, 1],
        "principal_curvatures_log": [1, 1],
    }
}
_FEATURE_WEIGHTS = {"patch": {"hsv": np.array([1, 0, 0])}}


def _make_baseline_lm() -> EvidenceGraphLM:
    lm = EvidenceGraphLM(
        max_match_distance=0.005,
        tolerances=_TOLERANCES,
        feature_weights=_FEATURE_WEIGHTS,
        max_graph_size=10,
        hypotheses_updater_args=dict(initial_possible_poses="informed"),
    )
    lm.mode = ExperimentMode.TRAIN
    return lm


def _make_cgal_lm() -> EvidenceGraphLM:
    lm = EvidenceGraphLM(
        max_match_distance=0.005,
        tolerances=_TOLERANCES,
        feature_weights=_FEATURE_WEIGHTS,
        max_graph_size=10,
        hypotheses_updater_args=dict(initial_possible_poses="informed"),
        cgal_consensus_gating={
            "consensus_gated_plasticity": True,
            "alpha": 0.7,
            "baseline_rate": 1.0,
        },
        cgal_novelty_detection={
            "hypothesis_novelty_detection": True,
            "novelty_threshold": 0.6,
        },
        cgal_trust_weights={"learned_trust_weights": True},
        cgal_salience_replay={
            "salience_tagged_replay": True,
            "replay_interval": 1,
            "num_replay_samples": 4,
        },
    )
    lm.mode = ExperimentMode.TRAIN
    return lm


# ---------------------------------------------------------------------------
# Synthetic episodes
# ---------------------------------------------------------------------------


_PLACEHOLDER_TARGET = {"object": "placeholder", "quat_rotation": [1, 0, 0, 0]}


def _learn_object(lm: EvidenceGraphLM, ctx, fake_obs, label: str) -> str:
    """Walk an LM through a TRAIN episode for one object and call post_episode."""
    lm.mode = ExperimentMode.TRAIN
    target = dict(_PLACEHOLDER_TARGET)
    target["object"] = label
    lm.pre_episode(primary_target=target)
    for obs in fake_obs:
        lm.exploratory_step(ctx, [obs])
    # The default test scaffolding manually pins detected_object so post_episode
    # actually writes a graph. We do the same.
    lm.detected_object = label
    lm.detected_rotation_r = None
    lm.buffer.stats["detected_location_rel_body"] = (
        lm.buffer.get_current_location(input_channel="first")
    )
    lm.post_episode()
    return label


def _eval_episode(
    lm: EvidenceGraphLM, ctx, fake_obs, max_steps: int = 20
) -> tuple[str | None, int]:
    """Run an EVAL episode and return (detected_object, steps_taken)."""
    lm.mode = ExperimentMode.EVAL
    lm.pre_episode(primary_target=dict(_PLACEHOLDER_TARGET))
    steps = 0
    for i in range(max_steps):
        obs = fake_obs[i % len(fake_obs)]
        lm.add_lm_processing_to_buffer_stats(lm_processed=True)
        lm.matching_step(ctx, [obs])
        steps += 1
        ts = lm.update_terminal_condition()
        if ts in {"match", "no_match", "time_out", "pose_time_out"}:
            lm.set_detected_object(ts)
            return lm.detected_object, steps
    lm.set_detected_object("time_out")
    return lm.detected_object, steps


def _add_noise(fake_obs, sigma: float, rng: np.random.Generator):
    noisy = []
    for obs in fake_obs:
        new = copy.deepcopy(obs)
        new.location = np.asarray(new.location, dtype=float) + rng.normal(
            0.0, sigma, size=3
        )
        noisy.append(new)
    return noisy


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


@dataclass
class RunResult:
    label: str
    learned_objects: list[str] = field(default_factory=list)
    eval_records: list[dict[str, Any]] = field(default_factory=list)
    cgal_telemetry: dict[str, Any] = field(default_factory=dict)

    @property
    def num_graphs_in_memory(self) -> int:
        return len(self.learned_objects)

    @property
    def correct_matches(self) -> int:
        return sum(1 for r in self.eval_records if r["expected"] == r["detected"])

    @property
    def spurious_new_objects(self) -> int:
        return sum(
            1
            for r in self.eval_records
            if r["detected"] is not None
            and str(r["detected"]).startswith("new_object")
            and not str(r["expected"]).startswith("new_object")
        )

    @property
    def avg_steps(self) -> float:
        if not self.eval_records:
            return 0.0
        return float(np.mean([r["steps"] for r in self.eval_records]))


def _votes_from_lm(lm: EvidenceGraphLM, fallback_graph: str | None = None) -> dict:
    """Synthesize a ``vote_data`` dict from a single LM's current evidence.

    Each known graph contributes one ``Message`` whose location is the
    most-likely hypothesis location and whose confidence is the LM's max
    evidence for that graph. This is enough to drive
    :py:meth:`EvidenceGraphLM._cgal_preprocess_votes`.
    """
    vote_data: dict = {}
    for graph_id in lm.get_all_known_object_ids():
        ev = lm.evidence.get(graph_id)
        if ev is None or ev.size == 0:
            continue
        idx = int(np.argmax(ev))
        loc = lm.possible_locations[graph_id][idx]
        # Squash to [0, 1] - the Message contract requires bounded confidence.
        confidence = float(np.clip(ev[idx], 0.0, None))
        confidence = float(confidence / (1.0 + confidence))
        vote_data[graph_id] = [
            Message(
                location=np.asarray(loc, dtype=float),
                morphological_features={
                    "pose_vectors": np.eye(3),
                    "pose_fully_defined": True,
                },
                non_morphological_features=None,
                confidence=confidence,
                use_state=True,
                sender_id=lm.learning_module_id,
                sender_type="LM",
            )
        ]
    if not vote_data and fallback_graph is not None:
        vote_data[fallback_graph] = [
            Message(
                location=np.zeros(3),
                morphological_features={
                    "pose_vectors": np.eye(3),
                    "pose_fully_defined": True,
                },
                non_morphological_features=None,
                confidence=1.0,
                use_state=True,
                sender_id=lm.learning_module_id,
                sender_type="LM",
            )
        ]
    return vote_data


def _multi_lm_voting_round(
    lm_a: EvidenceGraphLM,
    lm_b: EvidenceGraphLM,
    ctx,
    fake_obs,
    expected_label: str,
    max_steps: int = 10,
) -> dict[str, Any]:
    """Run a 2-LM EVAL episode where the two LMs exchange votes each step.

    Both LMs independently process every observation and then exchange
    synthesised vote_data. This is the path that exercises CGAL's consensus
    gating and trust weights.
    """
    for lm in (lm_a, lm_b):
        lm.mode = ExperimentMode.EVAL
        lm.pre_episode(primary_target=dict(_PLACEHOLDER_TARGET))

    detected_a: str | None = None
    detected_b: str | None = None
    steps = 0
    for i in range(max_steps):
        obs = fake_obs[i % len(fake_obs)]
        for lm in (lm_a, lm_b):
            lm.add_lm_processing_to_buffer_stats(lm_processed=True)
            lm.matching_step(ctx, [obs])
        steps += 1

        # Cross-vote.
        votes_from_b = _votes_from_lm(lm_b)
        votes_from_a = _votes_from_lm(lm_a)
        if votes_from_b:
            lm_a.receive_votes(votes_from_b)
        if votes_from_a:
            lm_b.receive_votes(votes_from_a)

        ts_a = lm_a.update_terminal_condition()
        ts_b = lm_b.update_terminal_condition()
        if ts_a in {"match", "no_match", "time_out", "pose_time_out"}:
            lm_a.set_detected_object(ts_a)
            detected_a = lm_a.detected_object
        if ts_b in {"match", "no_match", "time_out", "pose_time_out"}:
            lm_b.set_detected_object(ts_b)
            detected_b = lm_b.detected_object
        if detected_a is not None and detected_b is not None:
            break
    if detected_a is None:
        lm_a.set_detected_object("time_out")
        detected_a = lm_a.detected_object
    if detected_b is None:
        lm_b.set_detected_object("time_out")
        detected_b = lm_b.detected_object

    return {
        "expected": expected_label,
        "detected_a": detected_a,
        "detected_b": detected_b,
        "steps": steps,
    }


# ---------------------------------------------------------------------------
# Main scenario
# ---------------------------------------------------------------------------


def run(seed: int = 0) -> dict[str, Any]:
    rng = np.random.default_rng(seed)

    base = BaseGraphTest()
    base.setUp()

    # Four learnable objects from the existing test fixture. With more
    # known objects the novelty-score distribution has more dynamic range
    # (a peaked match maps to low novelty, a flat one to high).
    train_objects = {
        "learn": base.fake_obs_learn,
        "square": base.fake_obs_square,
        "house": base.fake_obs_house,
        "house3d": base.fake_obs_house_3d,
    }

    # The eval stream interleaves clean and noisy presentations of every
    # learned object. Mild noise tends to keep one known object strongly
    # supported (peaked evidence -> low novelty); heavy noise flattens the
    # distribution (high novelty). The CGAL novelty gate should suppress
    # spurious ``new_object`` allocation in the peaked case while still
    # allowing it in the flat case.
    eval_stream = []
    for label, obs in train_objects.items():
        eval_stream.append((label, copy.deepcopy(obs)))  # clean
        eval_stream.append((label, _add_noise(obs, sigma=0.02, rng=rng)))
        eval_stream.append((label, _add_noise(obs, sigma=0.08, rng=rng)))

    results: dict[str, RunResult] = {}
    for arm_label, factory in (
        ("baseline", _make_baseline_lm),
        ("cgal", _make_cgal_lm),
    ):
        lm = factory()
        run_result = RunResult(label=arm_label)

        # --- Training phase: learn each object once. ---
        for obj_label, obs in train_objects.items():
            _learn_object(lm, base.ctx, obs, label=f"obj_{obj_label}")
            run_result.learned_objects = list(lm.get_all_known_object_ids())

        # --- Eval phase. ---
        for expected_label, obs in eval_stream:
            detected, steps = _eval_episode(lm, base.ctx, obs)
            run_result.eval_records.append(
                {
                    "expected": f"obj_{expected_label}",
                    "detected": detected,
                    "steps": steps,
                }
            )
            run_result.learned_objects = list(lm.get_all_known_object_ids())

        # --- CGAL telemetry, if enabled. ---
        if hasattr(lm, "get_cgal_telemetry") and lm.cgal_enabled:
            run_result.cgal_telemetry = lm.get_cgal_telemetry()

        results[arm_label] = run_result

    # --- Multi-LM voting scenario (engages consensus_gating + trust). ---
    multi_lm_results: dict[str, list[dict[str, Any]]] = {}
    for arm_label, factory in (
        ("baseline", _make_baseline_lm),
        ("cgal", _make_cgal_lm),
    ):
        lm_a = factory()
        lm_a.learning_module_id = f"{arm_label}_LM_A"
        lm_b = factory()
        lm_b.learning_module_id = f"{arm_label}_LM_B"
        for obj_label, obs in train_objects.items():
            for lm in (lm_a, lm_b):
                _learn_object(lm, base.ctx, obs, label=f"obj_{obj_label}")

        records = []
        for expected_label, obs in eval_stream:
            rec = _multi_lm_voting_round(
                lm_a, lm_b, base.ctx, obs, expected_label=f"obj_{expected_label}"
            )
            records.append(rec)
        multi_lm_results[arm_label] = records
        if hasattr(lm_a, "get_cgal_telemetry") and lm_a.cgal_enabled:
            results[arm_label].cgal_telemetry["multi_lm_a"] = lm_a.get_cgal_telemetry()
        if hasattr(lm_b, "get_cgal_telemetry") and lm_b.cgal_enabled:
            results[arm_label].cgal_telemetry["multi_lm_b"] = lm_b.get_cgal_telemetry()

    summary = {
        arm: {
            "num_graphs_in_memory": r.num_graphs_in_memory,
            "learned_objects": r.learned_objects,
            "correct_matches": r.correct_matches,
            "spurious_new_objects": r.spurious_new_objects,
            "avg_steps": r.avg_steps,
            "eval_records": r.eval_records,
            "cgal_telemetry": r.cgal_telemetry,
            "multi_lm_records": multi_lm_results[arm],
        }
        for arm, r in results.items()
    }
    return summary


def main() -> None:
    summary = run(seed=0)
    out_path = Path(__file__).with_name("cgal_vs_baseline_results.json")
    out_path.write_text(json.dumps(summary, indent=2, default=str))

    print("\n=== Monty baseline vs CGAL synthetic comparison ===\n")
    header = f"{'arm':<10}{'graphs':>8}{'correct':>9}{'spurious':>10}{'avg_steps':>12}"
    print(header)
    print("-" * len(header))
    for arm in ("baseline", "cgal"):
        s = summary[arm]
        print(
            f"{arm:<10}"
            f"{s['num_graphs_in_memory']:>8}"
            f"{s['correct_matches']:>9}"
            f"{s['spurious_new_objects']:>10}"
            f"{s['avg_steps']:>12.2f}"
        )
    if summary["cgal"]["cgal_telemetry"]:
        print("\nCGAL telemetry (single-LM arm):")
        for k, v in summary["cgal"]["cgal_telemetry"].items():
            print(f"  {k}: {v}")

    print("\n=== Multi-LM voting scenario ===\n")
    header = (
        f"{'arm':<10}{'correct_a':>11}{'correct_b':>11}"
        f"{'spurious_a':>12}{'spurious_b':>12}{'avg_steps':>12}"
    )
    print(header)
    print("-" * len(header))
    for arm in ("baseline", "cgal"):
        recs = summary[arm]["multi_lm_records"]
        correct_a = sum(1 for r in recs if r["expected"] == r["detected_a"])
        correct_b = sum(1 for r in recs if r["expected"] == r["detected_b"])
        spurious_a = sum(
            1
            for r in recs
            if r["detected_a"] is not None
            and str(r["detected_a"]).startswith("new_object")
            and not str(r["expected"]).startswith("new_object")
        )
        spurious_b = sum(
            1
            for r in recs
            if r["detected_b"] is not None
            and str(r["detected_b"]).startswith("new_object")
            and not str(r["expected"]).startswith("new_object")
        )
        avg_steps = float(np.mean([r["steps"] for r in recs])) if recs else 0.0
        print(
            f"{arm:<10}{correct_a:>11}{correct_b:>11}"
            f"{spurious_a:>12}{spurious_b:>12}{avg_steps:>12.2f}"
        )

    print(f"\nFull results written to {out_path}")


if __name__ == "__main__":
    main()
