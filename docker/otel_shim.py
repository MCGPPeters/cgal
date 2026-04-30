"""OpenTelemetry instrumentation shim for Monty experiment runs.

This module is imported by ``run_benchmark.sh`` *before* ``run.py`` so it
can monkey-patch ``MontyExperiment`` lifecycle hooks. We emit:

    - one span per experiment run (root)
    - one span per episode
    - counters for matching steps, terminal conditions and CGAL events

The shim degrades gracefully: if Monty's API changes or a hook is
missing the patch is silently skipped — instrumentation must never
break the benchmark.
"""
from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Any, Callable

from opentelemetry import metrics, trace
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
    OTLPMetricExporter,
)
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
    OTLPSpanExporter,
)
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

log = logging.getLogger("cgal.otel_shim")

_resource = Resource.create(
    {
        "service.name": os.environ.get("OTEL_SERVICE_NAME", "monty"),
        "service.namespace": "cgal",
        "experiment": os.environ.get("EXPERIMENT", "unknown"),
        "cgal.enabled": os.environ.get("CGAL_ENABLED", "false"),
    }
)

_tracer_provider = TracerProvider(resource=_resource)
_tracer_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
trace.set_tracer_provider(_tracer_provider)

_meter_provider = MeterProvider(
    resource=_resource,
    metric_readers=[PeriodicExportingMetricReader(OTLPMetricExporter(), export_interval_millis=2000)],
)
metrics.set_meter_provider(_meter_provider)

tracer = trace.get_tracer("cgal.monty")
meter = metrics.get_meter("cgal.monty")

episode_counter = meter.create_counter("monty.episodes", unit="1")
step_counter = meter.create_counter("monty.matching_steps", unit="1")
terminal_counter = meter.create_counter("monty.terminal_conditions", unit="1")


def _wrap(obj: Any, name: str, wrapper: Callable[[Callable], Callable]) -> bool:
    fn = getattr(obj, name, None)
    if fn is None or getattr(fn, "_cgal_wrapped", False):
        return False
    new_fn = wrapper(fn)
    new_fn._cgal_wrapped = True  # type: ignore[attr-defined]
    setattr(obj, name, new_fn)
    return True


@contextmanager
def _span(name: str, **attrs: Any):
    with tracer.start_as_current_span(name) as sp:
        for k, v in attrs.items():
            try:
                sp.set_attribute(k, v)
            except Exception:  # noqa: BLE001
                pass
        yield sp


def install() -> None:
    """Patch Monty's experiment runner to emit OTel signals."""
    try:
        from tbp.monty.frameworks.experiments.monty_experiment import (
            MontyExperiment,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not import MontyExperiment; skipping OTel patch: %s", exc)
        return

    cgal_enabled = os.environ.get("CGAL_ENABLED", "false")

    orig_run = getattr(MontyExperiment, "run_episode", None)
    if orig_run is not None:

        def run_episode(self, *args, **kwargs):  # type: ignore[no-redef]
            episode_counter.add(1, {"cgal_enabled": cgal_enabled})
            with _span("monty.episode", cgal_enabled=cgal_enabled):
                result = orig_run(self, *args, **kwargs)
                # Attempt to read terminal state from common attribute names.
                term = getattr(self, "terminal_state", None) or getattr(
                    self, "last_terminal_condition", None
                )
                if term is not None:
                    terminal_counter.add(1, {"condition": str(term)})
                return result

        MontyExperiment.run_episode = run_episode  # type: ignore[assignment]

    orig_step = getattr(MontyExperiment, "pre_step", None)
    if orig_step is not None:

        def pre_step(self, *args, **kwargs):  # type: ignore[no-redef]
            step_counter.add(1)
            return orig_step(self, *args, **kwargs)

        MontyExperiment.pre_step = pre_step  # type: ignore[assignment]

    log.info("CGAL OTel shim installed (cgal_enabled=%s)", cgal_enabled)


# Install on import so callers only need ``import otel_shim``.
install()
