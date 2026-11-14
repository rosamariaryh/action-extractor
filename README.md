# Action Extractor

Action Extractor processes diarized meeting transcripts and returns decisions, action items, and follow-ups. Each extracted item includes its description, speaker, source quote, and transcript timestamps.

Gemini is the currently supported LLM provider. The extractor uses speaker-labeled transcript segments, structured Gemini output, and Pydantic schemas.

## Configuration

Copy `.env.example` to `.env` and set `GEMINI_API_KEY`. `GEMINI_MODEL_NAME` is optional; the application uses `gemini-2.5-flash` when it is omitted. `LLM_PROVIDER` must be `gemini`.

Request processing can be tuned with `LLM_MAX_CONCURRENCY`, `LLM_MAX_RETRIES`, and `LLM_REQ_PER_MINUTE` (the settings loader also accepts the legacy `LLM_REQUESTS_PER_MINUTE`). Transcript chunks target 12,000 serialized characters; an individual segment is kept intact even if it exceeds that target.

## Run locally

Extract a transcript to `output/`:

```bash
uv run python -m app.main input_data/input1.json
```

Start the API:

```bash
uv run uvicorn app.api:app --host 0.0.0.0 --port 8000
```

Run the unit tests:

```bash
uv run pytest
```

## API

- `GET /health` returns `{"status": "ok"}`.
- `POST /extract` accepts a transcript JSON object and returns its original fields plus `decisions`, `action_items`, and `follow_ups`.
- OpenAPI documentation is available at `http://127.0.0.1:8000/docs`.

The API returns `422` for invalid transcript segment structure, `503` when the LLM is not configured, and `502` when extraction fails.

## Docker Compose

Build and start the API, Redis, and Phoenix services:

```bash
docker compose up --build
```

The `.env` file is passed to the application container and is excluded from the image. The API is available at `http://127.0.0.1:8000/docs`; Phoenix is available at `http://127.0.0.1:6006`.

Run evaluations in a one-off container. The default input is the existing extraction at `output/input1.json`:

```bash
docker compose --profile eval run --build --rm evals
```

To run extraction and evaluation on a raw transcript instead:

```bash
docker compose --profile eval run --build --rm evals input_data/input2.json
```

The eval container mounts `input_data/`, `output/`, `eval_output/`, and `evals/output/` from the host. Its JSON report is therefore available under the host's `evals/output/` directory.

## Evaluations

Run all evaluations locally against the default `output/input1.json` file:

```bash
uv run python -m evals.main
```

Pass another transcript or extracted transcript as the positional path. Raw transcripts are extracted into `eval_output/`; extracted files are evaluated as-is.

```bash
uv run python -m evals.main input_data/input2.json
uv run python -m evals.main output/input1.json
```

The runner produces three deterministic metrics and one Gemini LLM judge:

- `valid_json_schema` checks the three extraction arrays and validates their items against the output schema.
- `valid_timestamps` checks timestamp format and that each extracted range falls within a transcript segment.
- `quote_grounding` checks that each extracted quote appears in a transcript segment.
- `actionability` uses Gemini to assess whether extracted action items are grounded, actionable commitments.

Each run creates one timestamped JSON report in `evals/output/`. It stores the input file path and transcript once, then lists metric type, name, score, label, and explanation in `metrics`. Set `EVAL_OUTPUT_DIR` to change the report directory. Expected action count and notes can be provided as guidance:

```bash
uv run python -m evals.main input_data/input1.json --expected-action-count 2 --notes "One webhook fix and one documentation follow-up."
```

## Phoenix observability

The Gemini extractor and actionability judge are instrumented with OpenTelemetry and OpenInference. In the Phoenix UI, the judge call appears as a Gemini generation span nested under `eval.actionability`. Gemini token counts are shown when the response includes usage metadata. Deterministic evaluation spans and the corresponding evaluation annotations are also sent to Phoenix.

Docker Compose configures the OTLP gRPC collector at `http://phoenix:4317` and the Phoenix API/UI at `http://phoenix:6006`. Compose currently runs Phoenix with authentication disabled. If authentication is enabled in another deployment, configure its API key as `PHOENIX_API_KEY`. For local application runs outside Compose, set `PHOENIX_COLLECTOR_ENDPOINT` (for example, `http://127.0.0.1:4317`); set `PHOENIX_UI_ENDPOINT` if the Phoenix API endpoint cannot be inferred from the collector address.

## Dependencies

The project uses `uv`:

```bash
uv sync
uv add <package>
```
## Future work

- Test massive amounts of data and edge cases.
- Study latency, cost, and quality trade-offs by testing more models and prompts.
- Adjust LLM concurrency and requests per minute accordingly.
- Optimize token usage by caching methods.
- Flag bad-quality evals to handle regressions.
