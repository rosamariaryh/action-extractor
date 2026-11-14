"""Optional OpenTelemetry tracing for Gemini calls, collected by Phoenix."""

from __future__ import annotations

import atexit
import os
from threading import Lock

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from openinference.instrumentation.google_genai import GoogleGenAIInstrumentor


_lock = Lock()
_provider: TracerProvider | None = None


def setup_phoenix_tracing() -> bool:
    """Instrument Google GenAI once when a Phoenix OTLP endpoint is configured."""
    global _provider

    endpoint = os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "").strip()
    if not endpoint:
        return False

    with _lock:
        if _provider is not None:
            return True

        headers = None
        api_key = os.getenv("PHOENIX_API_KEY", "").strip()
        if api_key:
            headers = {"authorization": f"Bearer {api_key}"}

        provider = TracerProvider(
            resource=Resource.create({"service.name": "action-extractor"})
        )
        provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint, headers=headers))
        )
        trace.set_tracer_provider(provider)
        GoogleGenAIInstrumentor().instrument(tracer_provider=provider)
        _provider = provider
        atexit.register(provider.shutdown)

    return True
