# llm-verse-avneesh

Async LLM router across AWS Bedrock (Claude, Amazon Nova) and Google Gemini.

A single `Router.get_response(...)` call validates its input, dispatches to
the requested model by name, and always returns a plain `dict` shaped like
[`LLMResponse`](src/llm_verse_avneesh/types.py) — regardless of which
underlying provider handled the call.

## Install

```bash
pip install "llm-verse-avneesh[aws]"      # Bedrock (Claude, Nova) only
pip install "llm-verse-avneesh[google]"   # Gemini only
pip install "llm-verse-avneesh[all]"      # both
```

## Usage

```python
import asyncio
from llm_verse_avneesh import Router

async def main():
    router = Router()
    result = await router.get_response(
        llm_name="nova-lite",
        system_prompt="You are a helpful assistant.",
        user_prompt="Say hello in one sentence.",
        context=None,
        temperature=0.5,
        pydantic_model=None,
        max_tokens=200,
        repo_name="my-repo",
        llm_identifier="greeting-1",
        region_name="us-east-1",
        aws_access_key_id="...",
        aws_secret_access_key="...",
    )
    print(result["response"])

asyncio.run(main())
```

For Gemini, pass `google_api_key` instead of the AWS credential fields.

## Registered model names

Registry keys match the name each provider documents publicly for the
model — not an internal AWS/Anthropic model ID:

| `llm_name`                 | Provider | Notes                                  |
|-----------------------------|----------|-----------------------------------------|
| `claude-haiku-4-5`          | Bedrock  | plain text / structured output          |
| `claude-haiku-4-5-agentic`  | Bedrock  | tool-calling loop, pass `tools=[...]`   |
| `nova-lite`                 | Bedrock  |                                          |
| `nova-2-lite`                | Bedrock  |                                          |
| `nova-2-lite-grounding`      | Bedrock  | web grounding; US regions only          |
| `nova-pro`                  | Bedrock  |                                          |
| `gemini-3.1-flash-lite`      | Google   |                                          |

## Validation

- `LLMRequest` (in `models.py`) does basic sanity checks: non-empty strings,
  non-negative temperature, positive `max_tokens`, and that credentials for
  the selected provider are present.
- `Router.get_response` then enforces the **actual per-model** limits from
  [`limits.py`](src/llm_verse_avneesh/limits.py) — e.g. Gemini allows
  temperature up to 2.0 while the Bedrock models here cap at 1.0. Exceeding a
  model's limit raises `RouterValidationError` naming the offending model and
  its limit. Update `limits.py` if a provider changes its published limits.

## Credentials

`aws_secret_access_key` and `google_api_key` are stored as `pydantic.SecretStr`
so they don't leak into `repr()`, `str()`, logs, or `model_dump()` output by
accident. Call `.get_secret_value()` only at the point where a client needs
the raw value.

## Exceptions

All exceptions inherit from `LLMRouterError`:

- `RouterValidationError` — bad input or a model-specific limit was exceeded.
- `ProviderNotFoundError` — `llm_name` isn't registered.
- `LLMCallError` — the underlying provider call failed or raised.

## Development

```bash
pip install -e ".[dev]"
pytest
```
