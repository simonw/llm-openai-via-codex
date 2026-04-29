import base64
import json
import os
import time
import urllib.request
import urllib.error
from enum import Enum
from typing import AsyncGenerator, Iterator, Optional

import llm
from llm import AsyncModel, Model, Options, hookimpl
from llm.utils import simplify_usage_dict
import openai
from pydantic import Field, create_model


# --- Vendored borrow_codex_key ---

REFRESH_URL = "https://auth.openai.com/oauth/token"
CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
REFRESH_SKEW_SECONDS = 30
CODEX_BASE_URL = "https://chatgpt.com/backend-api/codex"


class BorrowKeyError(Exception):
    pass


def borrow_codex_key():
    """
    Return (access_token, account_id) borrowed from the local Codex CLI
    ChatGPT OAuth credentials. Automatically refreshes if the access_token
    is expired or near-expiry.
    """
    auth_path = _auth_path()
    data = _read_auth(auth_path)

    tokens = data.get("tokens")
    if not tokens or not tokens.get("access_token"):
        raise BorrowKeyError(
            "No ChatGPT tokens found in auth.json. Run `codex login` first."
        )

    access_token = tokens["access_token"]
    account_id = tokens.get("account_id")
    exp = _jwt_exp(access_token)

    if exp is not None and time.time() < (exp - REFRESH_SKEW_SECONDS):
        return access_token, account_id

    refresh_token = tokens.get("refresh_token")
    if not refresh_token:
        raise BorrowKeyError(
            "No refresh token available. Run `codex login` to re-authenticate."
        )

    new_tokens = _refresh(refresh_token)

    if new_tokens.get("access_token"):
        tokens["access_token"] = new_tokens["access_token"]
    if new_tokens.get("id_token"):
        tokens["id_token"] = new_tokens["id_token"]
    if new_tokens.get("refresh_token"):
        tokens["refresh_token"] = new_tokens["refresh_token"]

    data["tokens"] = tokens
    data["last_refresh"] = time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime())

    _write_auth(auth_path, data)

    return tokens["access_token"], account_id


def _auth_path():
    codex_home = os.environ.get("CODEX_HOME", os.path.expanduser("~/.codex"))
    path = os.path.join(codex_home, "auth.json")
    if not os.path.exists(path):
        raise BorrowKeyError(
            f"Codex auth file not found at {path}. Run `codex login` first."
        )
    return path


def _read_auth(path):
    with open(path) as f:
        data = json.load(f)

    if "auth_mode" not in data:
        raise BorrowKeyError(
            "No auth_mode key found in auth.json. Run `codex login` to re-authenticate."
        )

    if data.get("auth_mode") != "chatgpt":
        raise BorrowKeyError(
            f"Expected auth_mode 'chatgpt', got '{data.get('auth_mode')}'. "
            "This library only supports ChatGPT OAuth tokens."
        )
    return data


def _write_auth(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)
    os.chmod(path, 0o600)


def _jwt_exp(token):
    try:
        payload_b64 = token.split(".")[1]
        payload_b64 += "=" * (4 - len(payload_b64) % 4)
        payload = json.loads(base64.urlsafe_b64decode(payload_b64))
        return payload.get("exp")
    except Exception:
        return None


def _refresh(refresh_token):
    body = json.dumps(
        {
            "client_id": CLIENT_ID,
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
        }
    ).encode()

    req = urllib.request.Request(
        REFRESH_URL,
        data=body,
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        error_body = e.read().decode(errors="replace")
        try:
            error_data = json.loads(error_body)
            error_code = error_data.get("error")
        except Exception:
            error_code = None

        if error_code in (
            "refresh_token_expired",
            "refresh_token_reused",
            "refresh_token_invalidated",
        ):
            raise BorrowKeyError(
                f"Refresh token is no longer valid ({error_code}). "
                "Run `codex login` to re-authenticate."
            ) from None

        raise BorrowKeyError(
            f"Token refresh failed (HTTP {e.code}): {error_body}"
        ) from None
    except urllib.error.URLError as e:
        raise BorrowKeyError(f"Token refresh failed (network error): {e}") from None


# --- Fetch available models from the Codex endpoint ---

DEFAULT_MODELS = [
    "gpt-5.4",
    "gpt-5.4-mini",
    "gpt-5.4-nano",
]


def _fetch_codex_models():
    """
    Fetch the list of available models from the Codex endpoint.
    Returns a list of model slug strings. Falls back to DEFAULT_MODELS on error.
    """
    try:
        token, account_id = borrow_codex_key()
    except BorrowKeyError:
        return DEFAULT_MODELS

    headers = {"Authorization": f"Bearer {token}"}
    if account_id:
        headers["ChatGPT-Account-ID"] = account_id

    req = urllib.request.Request(
        f"{CODEX_BASE_URL}/models?client_version=1.0.0",
        headers=headers,
    )

    try:
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read())
        return [
            m["slug"]
            for m in data.get("models", [])
            if m.get("supported_in_api") and m.get("visibility") == "list"
        ]
    except Exception:
        return DEFAULT_MODELS


# --- LLM Plugin ---


class ReasoningEffortEnum(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"
    xhigh = "xhigh"


class CodexOptions(Options):
    max_output_tokens: Optional[int] = Field(
        description="Upper bound for tokens in the response.",
        ge=0,
        default=None,
    )
    temperature: Optional[float] = Field(
        description=(
            "Sampling temperature, between 0 and 2. Higher values make output "
            "more random, lower values more focused."
        ),
        ge=0,
        le=2,
        default=None,
    )
    top_p: Optional[float] = Field(
        description="Nucleus sampling: only consider tokens in the top_p probability mass.",
        ge=0,
        le=1,
        default=None,
    )
    reasoning_effort: Optional[ReasoningEffortEnum] = Field(
        description="Reasoning effort level: low, medium, high, or xhigh.",
        default=None,
    )


class _SharedCodexResponses:
    needs_key = None  # We get the key from borrow_codex_key

    def __init__(self, model_name):
        self.model_id = "openai-codex/" + model_name
        self.model_name = model_name
        self.can_stream = True
        self.supports_schema = True
        self.supports_tools = True
        self.attachment_types = {
            "image/png",
            "image/jpeg",
            "image/webp",
            "image/gif",
        }
        self.Options = CodexOptions

    def __str__(self):
        return f"OpenAI Codex: {self.model_id}"

    def _get_client_kwargs(self):
        token, account_id = borrow_codex_key()
        headers = {}
        if account_id:
            headers["ChatGPT-Account-ID"] = account_id
        return {
            "api_key": token,
            "base_url": CODEX_BASE_URL,
            "default_headers": headers,
        }

    def set_usage(self, response, usage):
        if not usage:
            return
        if not isinstance(usage, dict):
            usage = usage.model_dump()
        input_tokens = usage.pop("input_tokens")
        output_tokens = usage.pop("output_tokens")
        usage.pop("total_tokens", None)
        response.set_usage(
            input=input_tokens, output=output_tokens, details=simplify_usage_dict(usage)
        )

    def _build_messages(self, prompt, conversation):
        messages = []
        if conversation is not None:
            for prev_response in conversation.responses:
                if prev_response.attachments:
                    attachment_message = []
                    if prev_response.prompt.prompt:
                        attachment_message.append(
                            {"type": "input_text", "text": prev_response.prompt.prompt}
                        )
                    for attachment in prev_response.attachments:
                        attachment_message.append(_attachment(attachment))
                    messages.append({"role": "user", "content": attachment_message})
                else:
                    messages.append(
                        {"role": "user", "content": prev_response.prompt.prompt}
                    )
                for tool_result in getattr(prev_response.prompt, "tool_results", []):
                    if not tool_result.tool_call_id:
                        continue
                    messages.append(
                        {
                            "type": "function_call_output",
                            "call_id": tool_result.tool_call_id,
                            "output": tool_result.output,
                        }
                    )
                prev_text = prev_response.text_or_raise()
                if prev_text:
                    messages.append({"role": "assistant", "content": prev_text})
                tool_calls = prev_response.tool_calls_or_raise()
                if tool_calls:
                    for tool_call in tool_calls:
                        messages.append(
                            {
                                "type": "function_call",
                                "call_id": tool_call.tool_call_id,
                                "name": tool_call.name,
                                "arguments": json.dumps(tool_call.arguments),
                            }
                        )
        if not prompt.attachments:
            messages.append({"role": "user", "content": prompt.prompt or ""})
        else:
            attachment_message = []
            if prompt.prompt:
                attachment_message.append({"type": "input_text", "text": prompt.prompt})
            for attachment in prompt.attachments:
                attachment_message.append(_attachment(attachment))
            messages.append({"role": "user", "content": attachment_message})
        for tool_result in getattr(prompt, "tool_results", []):
            if not tool_result.tool_call_id:
                continue
            messages.append(
                {
                    "type": "function_call_output",
                    "call_id": tool_result.tool_call_id,
                    "output": tool_result.output,
                }
            )
        return messages

    def _build_kwargs(self, prompt, conversation):
        messages = self._build_messages(prompt, conversation)
        kwargs = {
            "model": self.model_name,
            "input": messages,
            "store": False,
            "stream": True,
            "instructions": prompt.system or "You are a helpful assistant.",
        }
        for option in ("max_output_tokens", "temperature", "top_p"):
            value = getattr(prompt.options, option, None)
            if value is not None:
                kwargs[option] = value

        reasoning_effort = getattr(prompt.options, "reasoning_effort", None)
        if reasoning_effort is not None:
            kwargs["reasoning"] = {"effort": reasoning_effort}

        if prompt.tools:
            tool_defs = []
            for tool in prompt.tools:
                if not getattr(tool, "name", None):
                    continue
                parameters = tool.input_schema or {
                    "type": "object",
                    "properties": {},
                }
                tool_defs.append(
                    {
                        "type": "function",
                        "name": tool.name,
                        "description": tool.description or None,
                        "parameters": parameters,
                        "strict": False,
                    }
                )
            if tool_defs:
                kwargs["tools"] = tool_defs
        if self.supports_schema and prompt.schema:
            kwargs["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": "output",
                    "schema": prompt.schema,
                }
            }
        return kwargs

    def _handle_event(self, event, response):
        et = getattr(event, "type", None)
        if et == "response.output_text.delta":
            return event.delta

        if et == "response.output_item.done":
            item = event.item
            if hasattr(item, "model_dump"):
                data = item.model_dump()
            elif isinstance(item, dict):
                data = item
            else:
                data = getattr(item, "__dict__", {}) or {}
            if data.get("type") == "function_call":
                tool_id = data.get("call_id") or data.get("id") or "unknown"
                name = data.get("name") or "unknown_tool"
                arguments = data.get("arguments") or "{}"
                try:
                    parsed = json.loads(arguments)
                except Exception:
                    parsed = arguments
                response.add_tool_call(
                    llm.ToolCall(
                        tool_call_id=tool_id,
                        name=name,
                        arguments=parsed,
                    )
                )

        if et == "response.completed":
            response.response_json = event.response.model_dump()
            self.set_usage(response, event.response.usage)
            return None


class CodexResponsesModel(_SharedCodexResponses, Model):
    def execute(self, prompt, stream, response, conversation):
        client = openai.OpenAI(**self._get_client_kwargs())
        kwargs = self._build_kwargs(prompt, conversation)
        for event in client.responses.create(**kwargs):
            delta = self._handle_event(event, response)
            if delta is not None:
                yield delta


class AsyncCodexResponsesModel(_SharedCodexResponses, AsyncModel):
    async def execute(self, prompt, stream, response, conversation):
        client = openai.AsyncOpenAI(**self._get_client_kwargs())
        kwargs = self._build_kwargs(prompt, conversation)
        async for event in await client.responses.create(**kwargs):
            delta = self._handle_event(event, response)
            if delta is not None:
                yield delta


def _attachment(attachment):
    url = attachment.url
    if not url:
        base64_content = attachment.base64_content()
        url = f"data:{attachment.resolve_type()};base64,{base64_content}"
    return {"type": "input_image", "image_url": url, "detail": "low"}


@hookimpl
def register_models(register):
    model_names = _fetch_codex_models()
    for model_name in model_names:
        register(
            CodexResponsesModel(model_name),
            AsyncCodexResponsesModel(model_name),
        )
