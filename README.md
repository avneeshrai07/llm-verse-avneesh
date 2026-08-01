# llm-verse-avneesh

Async LLM router across AWS Bedrock (Claude, Amazon Nova), Google Gemini, and Groq (GPT OSS, Qwen).

A single `Router.get_response(...)` call validates its input, dispatches to
the requested model by name, and always returns a plain `dict` shaped like
[`LLMResponse`](src/llm_verse_avneesh/types.py) — regardless of which
underlying provider handled the call.

## Install

```bash
pip install "llm-verse-avneesh[aws]"      # Bedrock (Claude, Nova) only
pip install "llm-verse-avneesh[google]"   # Gemini only
pip install "llm-verse-avneesh[groq]"     # Groq (GPT OSS, Qwen) only
pip install "llm-verse-avneesh[all]"      # all three
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

For Gemini, pass `google_api_key` instead of the AWS credential fields. For
Groq, pass `groq_api_key` instead. For `qwen-3.6-27b` (the only vision-capable
model here), pass `images=[...]` (URLs or `data:image/...;base64,...` URIs,
max 5) alongside `user_prompt` to include images in the request.

## Discovering models

Three callables let you explore what's registered without reading source:

```python
import llm_verse_avneesh as lv

lv.help()             # documents list_models(), model_info(), and Router.get_response()
lv.list_models()       # [{"llm_name": "nova-lite", "display_name": "Amazon Nova Lite", "provider": "bedrock"}, ...]
lv.model_info("nova-lite")   # required vs. optional arguments for that specific model
```

`model_info(llm_name)` returns exactly what that model needs and accepts —
e.g. Gemini models list `google_api_key` under `required` and Groq models
list `groq_api_key`, never AWS credentials; only models that actually
implement an agentic tool-calling loop (all chat models except
`nova-2-lite-grounding`) list `tools`/`max_iterations` under `optional`; only
`qwen-3.6-27b` lists `images` (vision input). It raises
`ProviderNotFoundError` for an unknown `llm_name`.

## Registered model names

Registry keys match the name each provider documents publicly for the
model — not an internal AWS/Anthropic model ID. (See `list_models()` /
`model_info()` above for this same information at runtime.)

| `llm_name`                 | Provider | Notes                                  |
|-----------------------------|----------|-----------------------------------------|
| `claude-haiku-4-5`          | Bedrock  | plain text / structured output; pass `tools=[...]` for an agentic tool-calling loop |
| `nova-lite`                 | Bedrock  | plain text / structured output; pass `tools=[...]` for an agentic tool-calling loop |
| `nova-2-lite`                | Bedrock  | plain text / structured output; pass `tools=[...]` for an agentic tool-calling loop |
| `nova-2-lite-grounding`      | Bedrock  | web grounding; US regions only          |
| `nova-pro`                  | Bedrock  | plain text / structured output; pass `tools=[...]` for an agentic tool-calling loop |
| `gemini-3.1-flash-lite`      | Google   | plain text / structured output; pass `tools=[...]` for an agentic tool-calling loop |
| `gpt-oss-120b`               | Groq     | plain text / structured output; pass `tools=[...]` for an agentic tool-calling loop |
| `gpt-oss-20b`                | Groq     | plain text / structured output; pass `tools=[...]` for an agentic tool-calling loop |
| `qwen-3.6-27b`               | Groq     | plain text / structured output / vision (`images=[...]`, max 5); pass `tools=[...]` for an agentic tool-calling loop |

## Validation

- `LLMRequest` (in `models.py`) does basic sanity checks: non-empty strings,
  non-negative temperature, positive `max_tokens`, and that credentials for
  the selected provider are present.
- `Router.get_response` then enforces the **actual per-model** limits from
  [`limits.py`](src/llm_verse_avneesh/limits.py) — e.g. Gemini and Groq allow
  temperature up to 2.0 while the Bedrock models here cap at 1.0. Exceeding a
  model's limit raises `RouterValidationError` naming the offending model and
  its limit. Update `limits.py` if a provider changes its published limits.

## Credentials

`aws_secret_access_key`, `google_api_key`, and `groq_api_key` are stored as
`pydantic.SecretStr` so they don't leak into `repr()`, `str()`, logs, or
`model_dump()` output by accident. Call `.get_secret_value()` only at the
point where a client needs the raw value.

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
