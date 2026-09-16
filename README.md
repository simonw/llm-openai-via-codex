# llm-openai-via-codex

[![PyPI](https://img.shields.io/pypi/v/llm-openai-via-codex.svg)](https://pypi.org/project/llm-openai-via-codex/)
[![Changelog](https://img.shields.io/github/v/release/simonw/llm-openai-via-codex?include_prereleases&label=changelog)](https://github.com/simonw/llm-openai-via-codex/releases)
[![Tests](https://github.com/simonw/llm-openai-via-codex/actions/workflows/test.yml/badge.svg)](https://github.com/simonw/llm-openai-via-codex/actions/workflows/test.yml)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/simonw/llm-openai-via-codex/blob/main/LICENSE)

Access OpenAI models via an existing Codex subscription

Apparently [this is OK](https://twitter.com/romainhuet/status/2038699202834841962)! There's more background [on my blog](https://simonwillison.net/2026/Apr/23/gpt-5-5/).

## Installation

Install this plugin in the same environment as [LLM](https://llm.datasette.io/).
```bash
llm install llm-openai-via-codex
```
## Usage

First, make sure you have installed and authenticated [OpenAI Codex CLI](https://github.com/openai/codex).

To see the models available via your Codex subscription run:

```bash
llm models -q openai-codex
```
To run a prompt against one of those models:
```bash
llm -m openai-codex/gpt-5.5 'Generate an SVG of a pelican riding a bicycle'
```

### Luna Reserve

Some Codex subscriptions include a Luna Reserve: a second allowance on its own
meter that stays usable after the regular one is spent. It is reached through
the `gpt-reserve` model, not `gpt-5.6-luna` - that is the model the Reserve
presents as, but requesting it bills the regular allowance.

```bash
llm -m openai-codex/gpt-reserve 'Talk to me in Swedish'
```

To keep using your usual model and only drop to the Reserve once the regular
allowance runs out, pass `fallback_to_reserve`:

```bash
llm -m openai-codex/gpt-5.5 -o fallback_to_reserve 1 'Talk to me in Swedish'
```

The retry happens only for a `usage_limit_reached` error, only once, and never
after any output has already been streamed. Accounts without a Luna Reserve
will see the original error unchanged.

## Development

To set up this plugin locally, first checkout the code. Run the tests using `uv`:
```bash
cd llm-openai-via-codex
uv run pytest
```
To run LLM with the dev plugin installed:
```bash
uv run llm -m models
uv run llm -m openai-codex/gpt-5.5 'Talk to me in Swedish'
```
