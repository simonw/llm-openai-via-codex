import json
from unittest.mock import patch

import llm
import pytest

from llm_openai_via_codex import (
    BorrowKeyError,
    CodexResponsesModel,
    LUNA_RESERVE_MODEL,
    _fetch_codex_models,
    _is_usage_limit_error,
    DEFAULT_MODELS,
)


def test_plugin_is_installed():
    import llm_openai_via_codex


def test_models_are_registered():
    model_ids = [model.model_id for model in llm.get_models()]
    # At least the default models should be registered (or fetched ones)
    # Check the prefix is correct
    codex_models = [m for m in model_ids if m.startswith("openai-codex/")]
    assert len(codex_models) > 0


def test_model_id_prefix():
    model = CodexResponsesModel("gpt-5.4")
    assert model.model_id == "openai-codex/gpt-5.4"
    assert model.model_name == "gpt-5.4"
    assert str(model) == "OpenAI Codex: openai-codex/gpt-5.4"


def test_model_needs_no_key():
    model = CodexResponsesModel("gpt-5.4")
    assert model.needs_key is None


def test_model_can_stream():
    model = CodexResponsesModel("gpt-5.4")
    assert model.can_stream is True


def test_build_kwargs_basic():
    model = CodexResponsesModel("gpt-5.4")
    prompt = llm.Prompt(model=model, prompt="Hello")
    kwargs = model._build_kwargs(prompt, None)
    assert kwargs["model"] == "gpt-5.4"
    assert kwargs["store"] is False
    assert kwargs["stream"] is True
    assert kwargs["instructions"] == "You are a helpful assistant."
    assert kwargs["input"] == [{"role": "user", "content": "Hello"}]


def test_build_kwargs_with_system():
    model = CodexResponsesModel("gpt-5.4")
    prompt = llm.Prompt(model=model, prompt="Hello", system="Be brief.")
    kwargs = model._build_kwargs(prompt, None)
    assert kwargs["instructions"] == "Be brief."


def test_build_kwargs_with_options():
    model = CodexResponsesModel("gpt-5.4")
    prompt = llm.Prompt(model=model, prompt="Hello")
    prompt.options = model.Options(temperature=0.5, max_output_tokens=100, top_p=0.9)
    kwargs = model._build_kwargs(prompt, None)
    assert kwargs["temperature"] == 0.5
    assert kwargs["max_output_tokens"] == 100
    assert kwargs["top_p"] == 0.9


def test_build_kwargs_reasoning_effort():
    model = CodexResponsesModel("gpt-5.4")
    prompt = llm.Prompt(model=model, prompt="Hello")
    prompt.options = model.Options(reasoning_effort="high")
    kwargs = model._build_kwargs(prompt, None)
    assert kwargs["reasoning"] == {"effort": "high"}


def test_fetch_codex_models_fallback():
    with patch(
        "llm_openai_via_codex.borrow_codex_key",
        side_effect=BorrowKeyError("no auth"),
    ):
        models = _fetch_codex_models()
    assert models == DEFAULT_MODELS


class _UsageLimitError(Exception):
    """Shaped like the openai SDK's APIStatusError for a spent allowance."""

    status_code = 429
    body = {
        "type": "usage_limit_reached",
        "message": "The usage limit has been reached",
        "plan_type": "pro",
    }


def _catalog(*models):
    payload = json.dumps({"models": list(models)}).encode("utf-8")

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return payload

    return _Response()


def test_api_supported_models_are_registered_regardless_of_visibility():
    # `visibility` is an app presentation hint; `supported_in_api` decides
    # whether a model can actually be requested. gpt-reserve is marked hidden
    # but is a normal, usable model.
    catalog = _catalog(
        {"slug": "gpt-5.4", "supported_in_api": True, "visibility": "list"},
        {"slug": LUNA_RESERVE_MODEL, "supported_in_api": True, "visibility": "hide"},
        {"slug": "not-in-api", "supported_in_api": False, "visibility": "list"},
    )
    with patch("llm_openai_via_codex.borrow_codex_key", return_value=("t", None)):
        with patch("urllib.request.urlopen", return_value=catalog):
            models = _fetch_codex_models()
    assert models == ["gpt-5.4", LUNA_RESERVE_MODEL]


@pytest.mark.parametrize(
    "exception,expected",
    (
        (_UsageLimitError(), True),
        (
            Exception("Error code: 429 - {'error': {'type': 'usage_limit_reached'}}"),
            True,
        ),
        (
            Exception("Error code: 429 - {'error': {'type': 'rate_limit_exceeded'}}"),
            False,
        ),
        (Exception("connection reset"), False),
    ),
)
def test_usage_limit_detection(exception, expected):
    assert _is_usage_limit_error(exception) is expected


def _prompt(**options):
    return type("Prompt", (), {"options": type("Options", (), options)()})()


def test_reserve_is_only_attempted_when_asked_for():
    model = CodexResponsesModel("gpt-5.4")
    assert model._models_to_try(_prompt(fallback_to_reserve=None)) == ["gpt-5.4"]
    assert model._models_to_try(_prompt(fallback_to_reserve=True)) == [
        "gpt-5.4",
        LUNA_RESERVE_MODEL,
    ]


def test_reserve_is_never_retried_against_itself():
    model = CodexResponsesModel(LUNA_RESERVE_MODEL)
    assert model._models_to_try(_prompt(fallback_to_reserve=True)) == [
        LUNA_RESERVE_MODEL
    ]


def test_retry_rules():
    model = CodexResponsesModel("gpt-5.4")
    limit = _UsageLimitError()
    assert model._should_retry(limit, produced=False, last=False) is True
    # Output already yielded: retrying would duplicate part of the answer.
    assert model._should_retry(limit, produced=True, last=False) is False
    # Nothing left to fall back to.
    assert model._should_retry(limit, produced=False, last=True) is False
    # Not a usage limit.
    assert model._should_retry(Exception("boom"), produced=False, last=False) is False


def test_fallback_option_is_exposed():
    model = CodexResponsesModel("gpt-5.4")
    assert "fallback_to_reserve" in model.Options.model_fields
