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


def _is_deepseek_provider(provider: ProviderConfig) -> bool:
    url = (provider.base_url or "").lower()
    name = (provider.name or "").lower()
    return "deepseek" in url or name == "deepseek"


def _uses_responses_api(provider: ProviderConfig) -> bool:
    endpoint = (provider.endpoint or "").lower().strip()
    return _is_openai_provider(provider) and endpoint.rstrip("/").endswith("/responses")


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


def _build_chat_payload(
    provider: ProviderConfig,
    system_prompt: str,
    user_prompt: str,
    *,
    temperature: float | None,
    max_tokens: int,
    use_json_mode: bool,
    token_param: str,
    reasoning_effort: str | None,
    thinking: str | None,
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

    if reasoning_effort and (_is_openai_provider(provider) or _is_deepseek_provider(provider)):
        payload["reasoning_effort"] = reasoning_effort

    # DeepSeek V4 по умолчанию может включать thinking mode. Для нашей задачи нужен
    # только финальный JSON, поэтому в config.yaml можно задать thinking: disabled.
    if thinking and _is_deepseek_provider(provider):
        payload["thinking"] = {"type": thinking}

    return payload


def _build_responses_payload(
    provider: ProviderConfig,
    system_prompt: str,
    user_prompt: str,
    *,
    temperature: float | None,
    max_tokens: int,
    use_json_mode: bool,
    reasoning_effort: str | None,
    include_text_format: bool,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": provider.model,
        "instructions": system_prompt,
        "input": user_prompt,
        "max_output_tokens": max_tokens,
        "store": False,
    }

    if temperature is not None:
        payload["temperature"] = temperature

    if reasoning_effort:
        payload["reasoning"] = {"effort": reasoning_effort}

    if use_json_mode and include_text_format:
        # Responses API использует контейнер text.format, а не response_format.
        # Если конкретная модель/аккаунт это не поддержит, ниже будет retry без него.
        payload["text"] = {"format": {"type": "json_object"}}

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
    return status_code >= 400 and (
        "response_format" in b or "json_object" in b or "text.format" in b or '"text"' in b
    )


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


def _should_retry_without_thinking(body: str) -> bool:
    b = body.lower()
    return "thinking" in b and (
        "unsupported" in b or "does not support" in b or "invalid" in b
    )


def _choice_meta(raw: dict[str, Any]) -> str:
    try:
        choice = raw.get("choices", [{}])[0]
        finish_reason = choice.get("finish_reason", "")
        usage = raw.get("usage", {})
        message = choice.get("message", {}) if isinstance(choice, dict) else {}
        return json.dumps(
            {
                "finish_reason": finish_reason,
                "usage": usage,
                "has_reasoning_content": bool(message.get("reasoning_content")),
            },
            ensure_ascii=False,
            indent=2,
        )[:4000]
    except Exception:
        return ""


def _responses_meta(raw: dict[str, Any]) -> str:
    try:
        return json.dumps(
            {
                "status": raw.get("status"),
                "incomplete_details": raw.get("incomplete_details"),
                "usage": raw.get("usage"),
            },
            ensure_ascii=False,
            indent=2,
        )[:4000]
    except Exception:
        return ""


def _extract_responses_text(raw: dict[str, Any]) -> str:
    # REST response may include output_text directly.
    output_text = raw.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text

    parts: list[str] = []
    output = raw.get("output", [])
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict):
                continue
            content = item.get("content", [])
            if isinstance(content, list):
                for c in content:
                    if isinstance(c, dict):
                        text = c.get("text")
                        if isinstance(text, str) and c.get("type") in {"output_text", "text"}:
                            parts.append(text)
    return "".join(parts).strip()


def _format_parse_error(exc: Exception, content: str, raw: dict[str, Any], retry_notes: list[str], *, is_responses: bool) -> str:
    meta = _responses_meta(raw) if is_responses else _choice_meta(raw)
    err = f"{type(exc).__name__}: {exc}"
    if retry_notes:
        err += "\nRetries:\n- " + "\n- ".join(retry_notes)
    if meta:
        err += f"\nChoice/usage metadata:\n{meta}"
    if content:
        err += f"\nRaw assistant content head:\n{content[:4000]}"
    else:
        err += "\nRaw assistant content is empty."
    return err


def _call_openai_responses(
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

    current_temperature: float | None = temperature
    current_reasoning_effort = provider.reasoning_effort
    current_json_mode = use_json_mode
    include_text_format = True
    response: requests.Response | None = None
    retry_notes: list[str] = []

    try:
        for _attempt in range(5):
            payload = _build_responses_payload(
                provider,
                system_prompt,
                user_prompt,
                temperature=current_temperature,
                max_tokens=max_tokens,
                use_json_mode=current_json_mode,
                reasoning_effort=current_reasoning_effort,
                include_text_format=include_text_format,
            )
            response = requests.post(provider.url, headers=headers, json=payload, timeout=timeout_sec)

            if response.status_code < 400:
                raw = response.json()
                content = _extract_responses_text(raw)

                # Если модель снова потратила бюджет на reasoning без видимого текста,
                # повторяем с увеличенным бюджетом и/или без JSON text.format.
                if not content.strip() and _attempt < 4:
                    if current_json_mode and include_text_format:
                        retry_notes.append("Retry Responses API without text.format=json_object after empty output")
                        include_text_format = False
                        continue
                    if current_reasoning_effort not in {None, "minimal"}:
                        retry_notes.append("Retry Responses API with reasoning effort minimal after empty output")
                        current_reasoning_effort = "minimal"
                        continue
                break

            body = _error_text(response)
            if current_json_mode and include_text_format and _should_retry_without_json_mode(response.status_code, body):
                retry_notes.append("Retry Responses API without text.format=json_object")
                include_text_format = False
                continue

            if current_temperature is not None and _should_retry_without_temperature(body):
                retry_notes.append("Retry Responses API without temperature parameter")
                current_temperature = None
                continue

            if current_reasoning_effort is not None and _should_retry_without_reasoning_effort(body):
                retry_notes.append("Retry Responses API without reasoning effort")
                current_reasoning_effort = None
                continue

            response.raise_for_status()

        response.raise_for_status()
        raw = response.json()
        content = _extract_responses_text(raw)

        try:
            data = extract_json(content)
        except Exception as exc:
            err = _format_parse_error(exc, content, raw, retry_notes, is_responses=True)
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
    """
    Unified LLM call.

    * OpenAI GPT-5.x should use Responses API when provider.endpoint=/responses.
      This avoids the common Chat Completions failure where all completion budget is
      spent on hidden reasoning tokens and message.content is empty.
    * DeepSeek V4 should use Chat Completions, but for JSON production we pass
      thinking: {type: disabled} when configured, because thinking mode may return
      reasoning_content without final content.
    """
    if _uses_responses_api(provider):
        return _call_openai_responses(
            provider,
            api_key,
            system_prompt,
            user_prompt,
            timeout_sec=timeout_sec,
            temperature=temperature,
            max_tokens=max_tokens,
            use_json_mode=use_json_mode,
        )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    token_param = _initial_token_param(provider)
    current_use_json_mode = use_json_mode
    current_temperature: float | None = temperature
    current_reasoning_effort = provider.reasoning_effort
    current_thinking = provider.thinking
    response: requests.Response | None = None
    retry_notes: list[str] = []

    try:
        # До 6 попыток: базовая, смена max_tokens/max_completion_tokens,
        # отключение response_format, temperature, reasoning_effort, thinking при несовместимости.
        for _attempt in range(6):
            payload = _build_chat_payload(
                provider,
                system_prompt,
                user_prompt,
                temperature=current_temperature,
                max_tokens=max_tokens,
                use_json_mode=current_use_json_mode,
                token_param=token_param,
                reasoning_effort=current_reasoning_effort,
                thinking=current_thinking,
            )
            response = requests.post(provider.url, headers=headers, json=payload, timeout=timeout_sec)

            if response.status_code < 400:
                raw = response.json()
                choice = raw.get("choices", [{}])[0]
                message = choice.get("message", {}) if isinstance(choice, dict) else {}
                content = message.get("content") or ""
                reasoning_content = message.get("reasoning_content") or ""

                # DeepSeek V4 thinking mode может вернуть только reasoning_content.
                # Для нашей задачи это непригодно, повторяем с thinking disabled.
                if (
                    not content.strip()
                    and reasoning_content
                    and _is_deepseek_provider(provider)
                    and current_thinking != "disabled"
                    and _attempt < 5
                ):
                    retry_notes.append("Retry DeepSeek with thinking disabled after empty content with reasoning_content")
                    current_thinking = "disabled"
                    current_reasoning_effort = None
                    continue
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

            if current_thinking is not None and _should_retry_without_thinking(body):
                retry_notes.append("Retry without thinking parameter")
                current_thinking = None
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
