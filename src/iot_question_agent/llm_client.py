from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any

import requests

from .settings import ProviderConfig


JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


@dataclass
class LLMResult:
    provider: str
    model: str
    ok: bool
    content: str
    data: dict[str, Any] | None
    error: str | None = None


def extract_json(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    if not text:
        raise ValueError("Empty LLM response")

    # Если модель всё-таки вернула markdown-блок, достанем JSON из него.
    m = JSON_BLOCK_RE.search(text)
    if m:
        text = m.group(1).strip()

    # Если вокруг JSON есть текст, пытаемся выделить объект по первой/последней фигурной скобке.
    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last != -1 and last > first:
        text = text[first : last + 1]

    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("Top-level JSON must be an object")
    return parsed


def _is_openai_provider(provider: ProviderConfig) -> bool:
    url = (provider.base_url or "").lower()
    name = (provider.name or "").lower()
    return "openai.com" in url or name in {"openai", "chatgpt"}


def _initial_token_param(provider: ProviderConfig) -> str:
    """
    OpenAI o-series/GPT-5 family models reject max_tokens and require max_completion_tokens.
    DeepSeek and most OpenAI-compatible providers still expect max_tokens.
    """
    model = (provider.model or "").lower()
    if _is_openai_provider(provider) and (
        model.startswith("gpt-5")
        or model.startswith("o1")
        or model.startswith("o3")
        or model.startswith("o4")
    ):
        return "max_completion_tokens"
    return "max_tokens"


def _build_payload(
    provider: ProviderConfig,
    system_prompt: str,
    user_prompt: str,
    *,
    temperature: float | None,
    max_tokens: int,
    use_json_mode: bool,
    token_param: str,
    reasoning_effort: str | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": provider.model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        token_param: max_tokens,
    }

    if temperature is not None:
        payload["temperature"] = temperature

    if use_json_mode:
        payload["response_format"] = {"type": "json_object"}

    # Для OpenAI GPT-5/o-series это резко снижает риск, что весь budget уйдёт
    # в скрытые reasoning-токены и content вернётся пустым с finish_reason=length.
    if reasoning_effort and _is_openai_provider(provider):
        payload["reasoning_effort"] = reasoning_effort

    return payload


def _error_text(response: requests.Response | None) -> str:
    if response is None:
        return ""
    try:
        return response.text[:8000]
    except Exception:
        return ""


def _should_retry_without_json_mode(status_code: int, body: str) -> bool:
    b = body.lower()
    return status_code >= 400 and "response_format" in b


def _should_retry_token_param(body: str, current_token_param: str) -> str | None:
    b = body.lower()
    if current_token_param == "max_tokens" and (
        "max_tokens" in b and "max_completion_tokens" in b
    ):
        return "max_completion_tokens"
    if current_token_param == "max_completion_tokens" and (
        "max_completion_tokens" in b and "max_tokens" in b
    ):
        return "max_tokens"
    return None


def _should_retry_without_temperature(body: str) -> bool:
    b = body.lower()
    return "temperature" in b and (
        "unsupported" in b or "does not support" in b or "only the default" in b
    )


def _should_retry_without_reasoning_effort(body: str) -> bool:
    b = body.lower()
    return "reasoning_effort" in b and (
        "unsupported" in b or "does not support" in b or "invalid" in b
    )


def _choice_meta(raw: dict[str, Any]) -> str:
    try:
        choice = raw.get("choices", [{}])[0]
        finish_reason = choice.get("finish_reason", "")
        usage = raw.get("usage", {})
        return json.dumps(
            {"finish_reason": finish_reason, "usage": usage},
            ensure_ascii=False,
            indent=2,
        )[:4000]
    except Exception:
        return ""


def call_chat_completions(
    provider: ProviderConfig,
    api_key: str,
    system_prompt: str,
    user_prompt: str,
    *,
    timeout_sec: int,
    temperature: float,
    max_tokens: int,
    use_json_mode: bool,
) -> LLMResult:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    token_param = _initial_token_param(provider)
    current_use_json_mode = use_json_mode
    current_temperature: float | None = temperature
    current_reasoning_effort = provider.reasoning_effort
    response: requests.Response | None = None
    retry_notes: list[str] = []

    try:
        # До 5 попыток: базовая, смена max_tokens/max_completion_tokens,
        # отключение response_format, temperature, reasoning_effort при несовместимости.
        for _attempt in range(5):
            payload = _build_payload(
                provider,
                system_prompt,
                user_prompt,
                temperature=current_temperature,
                max_tokens=max_tokens,
                use_json_mode=current_use_json_mode,
                token_param=token_param,
                reasoning_effort=current_reasoning_effort,
            )
            response = requests.post(provider.url, headers=headers, json=payload, timeout=timeout_sec)

            if response.status_code < 400:
                break

            body = _error_text(response)

            new_token_param = _should_retry_token_param(body, token_param)
            if new_token_param is not None:
                retry_notes.append(f"Retry with {new_token_param} instead of {token_param}")
                token_param = new_token_param
                continue

            if current_use_json_mode and _should_retry_without_json_mode(response.status_code, body):
                retry_notes.append("Retry without response_format=json_object")
                current_use_json_mode = False
                continue

            if current_temperature is not None and _should_retry_without_temperature(body):
                retry_notes.append("Retry without temperature parameter")
                current_temperature = None
                continue

            if current_reasoning_effort is not None and _should_retry_without_reasoning_effort(body):
                retry_notes.append("Retry without reasoning_effort parameter")
                current_reasoning_effort = None
                continue

            # Ошибка не похожа на известную несовместимость параметров.
            response.raise_for_status()

        response.raise_for_status()
        raw = response.json()
        choice = raw.get("choices", [{}])[0]
        content = choice.get("message", {}).get("content") or ""
        finish_reason = choice.get("finish_reason", "")

        try:
            data = extract_json(content)
        except Exception as exc:
            meta = _choice_meta(raw)
            err = f"{type(exc).__name__}: {exc}"
            if finish_reason == "length":
                err += "\nThe model stopped because token limit was reached. JSON is probably incomplete."
            if retry_notes:
                err += "\nRetries:\n- " + "\n- ".join(retry_notes)
            if meta:
                err += f"\nChoice/usage metadata:\n{meta}"
            if content:
                err += f"\nRaw assistant content head:\n{content[:4000]}"
            else:
                err += "\nRaw assistant content is empty."
            return LLMResult(provider.name, provider.model, False, content, None, err)

        return LLMResult(provider.name, provider.model, True, content, data)
    except Exception as exc:
        body = _error_text(response)
        err = f"{type(exc).__name__}: {exc}"
        if retry_notes:
            err += "\nRetries:\n- " + "\n- ".join(retry_notes)
        if body:
            err += f"\nAPI response body:\n{body}"
        return LLMResult(provider.name, provider.model, False, "", None, err)
