# Action Extractor

Action Extractor reads diarized meeting transcripts and returns grounded decisions, action items, and follow-ups. Each item includes its source quote, speaker, and transcript timestamps. Gemini is the currently supported LLM provider.

## Quick start

1. Copy `.env.example` to `.env` and add your `GEMINI_API_KEY`.
2. Start the services:

   ```bash
   docker compose up --build
   ```

3. Open the API docs at <http://localhost:8000/docs>. Phoenix is available at <http://localhost:6006>.

The Compose stack starts the API, Redis, and Phoenix. Phoenix data is stored in the `phoenix_data` volume.

## API

The API accepts a transcript JSON object. `/extract` returns the original transcript fields with `decisions`, `action_items`, and `follow_ups` added. Each extracted item has a description, optional owner and due date, source quote, speaker, and start/end timestamps.

| Method | Path | Purpose | Success | Errors |
| --- | --- | --- | --- | --- |
| `GET` | `/health` | Check that the API is running. | `200` with `{"status":"ok"}` | — |
| `POST` | `/extract` | Extract decisions, action items, and follow-ups from a transcript JSON body. | `200` with the extraction result | `422` invalid transcript structure; `503` Gemini is not configured; `502` extraction failed. |
| `GET` | `/docs` | Open the interactive Swagger UI. | `200` HTML page | — |
| `GET` | `/redoc` | Open the ReDoc API reference. | `200` HTML page | — |
| `GET` | `/openapi.json` | Retrieve the OpenAPI schema. | `200` JSON document | — |

Example request:

```bash
curl -X POST http://localhost:8000/extract \
  -H "Content-Type: application/json" \
  --data @input_data/input1.json
```

## Configuration

Set `LLM_PROVIDER=gemini` and `GEMINI_API_KEY` in `.env`. `GEMINI_MODEL_NAME` is optional and defaults to `gemini-2.5-flash`.

Request handling can be tuned with:

| Variable | Purpose | Default |
| --- | --- | --- |
| `LLM_MAX_CONCURRENCY` | Maximum concurrent transcript chunk requests. | `4` |
| `LLM_MAX_RETRIES` | Retries for retryable LLM request failures. | `3` |
| `LLM_REQ_PER_MINUTE` | Request rate limit. The legacy `LLM_REQUESTS_PER_MINUTE` is also accepted. | `60` |

Transcript chunks target 12,000 serialized characters. An individual segment stays intact even if it exceeds that target.

## Run from the command line

Install dependencies with [uv](https://docs.astral.sh/uv/), then extract a transcript to `output/`:

```bash
uv sync
uv run python -m app.main input_data/input1.json
```

Run the unit tests with:

```bash
uv run pytest
```

## Evaluations

Run the evaluations in Docker against the existing extraction at `output/input1.json`:

```bash
docker compose --profile eval run --build --rm evals
```

Pass a raw transcript to extract it first, or pass an existing extraction to evaluate it as-is:

```bash
docker compose --profile eval run --build --rm evals input_data/input2.json
docker compose --profile eval run --build --rm evals output/input1.json
```

The runner produces three deterministic metrics and one Gemini judge:

- `valid_json_schema` validates the extraction arrays and each item against the output schema.
- `valid_timestamps` checks timestamp format and that each extracted range falls within a source segment.
- `quote_grounding` checks that each extracted quote appears in a transcript segment.
- `actionability` uses Gemini to assess whether action items are grounded, actionable commitments.

Each run writes a timestamped JSON report to `evals/output/`. The report stores the transcript once and includes expected guidance and each metric's type, name, score, label, and explanation. In Docker, the report directory is mounted from the host. Set `EVAL_OUTPUT_DIR` to use another directory.

Expected action count and notes can guide the judge:

```bash
docker compose --profile eval run --build --rm evals input_data/input1.json \
  --expected-action-count 2 \
  --notes "One webhook fix and one documentation follow-up."
```

Run evaluations locally with `uv run python -m evals.main`. For example:

```bash
uv run python -m evals.main input_data/input1.json --expected-action-count 2
```

## Phoenix observability

The extractor and actionability judge send traces to Phoenix. Evaluation metrics are attached as annotations. In Phoenix, the Gemini judge call appears as a generation span nested under `eval.actionability`; token counts are available when Gemini returns usage metadata.

Compose configures the OTLP gRPC collector at `http://phoenix:4317` and the Phoenix UI/API at `http://phoenix:6006`. Authentication is disabled in the local Compose setup. For an authenticated Phoenix deployment, set `PHOENIX_API_KEY`.

When running the app outside Compose, set `PHOENIX_COLLECTOR_ENDPOINT` (for example, `http://127.0.0.1:4317`). Set `PHOENIX_UI_ENDPOINT` if the Phoenix API endpoint cannot be inferred from the collector endpoint.
