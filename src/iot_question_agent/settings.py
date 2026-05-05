from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ProviderConfig:
    name: str
    enabled: bool
    prefix: str
    base_url: str
    endpoint: str
    model: str
    max_tokens: int | None = None
    reasoning_effort: str | None = None
    thinking: str | None = None

    @property
    def url(self) -> str:
        return self.base_url.rstrip("/") + "/" + self.endpoint.lstrip("/")


@dataclass(frozen=True)
class LLMRolesConfig:
    generator: str
    reviewer: str


@dataclass(frozen=True)
class ReviewConfig:
    enabled: bool
    max_rewrite_attempts: int
    save_review_files: bool


@dataclass(frozen=True)
class ReviewTestConfig:
    enabled: bool
    forced_bad_questions_count: int


@dataclass(frozen=True)
class AppConfig:
    root_dir: Path
    input_dir: Path
    output_dir: Path
    prompts_dir: Path
    secrets_file: Path
    question_count_per_document: int
    max_chars_per_llm_call: int
    timeout_sec: int
    temperature: float
    max_tokens: int
    use_json_mode: bool
    max_questions_per_llm_call: int
    llm_roles: LLMRolesConfig
    review: ReviewConfig
    review_test: ReviewTestConfig
    providers: list[ProviderConfig]
    excel_columns: list[str]


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def load_config(config_path: str | Path) -> AppConfig:
    config_path = Path(config_path).resolve()
    root_dir = config_path.parent
    raw = load_yaml(config_path)

    providers_raw = raw.get("providers", {})
    providers: list[ProviderConfig] = []
    if not isinstance(providers_raw, dict):
        raise ValueError("config.yaml field 'providers' must be a mapping")

    for name, item in providers_raw.items():
        if not isinstance(item, dict):
            continue
        providers.append(
            ProviderConfig(
                name=str(name).strip(),
                enabled=bool(item.get("enabled", True)),
                prefix=str(item.get("prefix", name)).strip(),
                base_url=str(item.get("base_url", "")).strip(),
                endpoint=str(item.get("endpoint", "/chat/completions")).strip(),
                model=str(item.get("model", "")).strip(),
                max_tokens=int(item["max_tokens"]) if item.get("max_tokens") is not None else None,
                reasoning_effort=str(item.get("reasoning_effort", "")).strip() or None,
                thinking=str(item.get("thinking", "")).strip() or None,
            )
        )

    roles_raw = raw.get("llm_roles", {})
    if not isinstance(roles_raw, dict):
        roles_raw = {}
    llm_roles = LLMRolesConfig(
        generator=str(roles_raw.get("generator", "chatgpt")).strip(),
        reviewer=str(roles_raw.get("reviewer", "deepseek")).strip(),
    )

    review_raw = raw.get("review", {})
    if not isinstance(review_raw, dict):
        review_raw = {}
    review = ReviewConfig(
        enabled=bool(review_raw.get("enabled", True)),
        max_rewrite_attempts=int(review_raw.get("max_rewrite_attempts", 1)),
        save_review_files=bool(review_raw.get("save_review_files", True)),
    )

    review_test_raw = raw.get("review_test", {})
    if not isinstance(review_test_raw, dict):
        review_test_raw = {}
    review_test = ReviewTestConfig(
        enabled=bool(review_test_raw.get("enabled", False)),
        forced_bad_questions_count=max(0, int(review_test_raw.get("forced_bad_questions_count", 0))),
    )

    return AppConfig(
        root_dir=root_dir,
        input_dir=root_dir / str(raw.get("input_dir", "input")),
        output_dir=root_dir / str(raw.get("output_dir", "output")),
        prompts_dir=root_dir / str(raw.get("prompts_dir", "prompts")),
        secrets_file=root_dir / str(raw.get("secrets_file", "secrets.yaml")),
        question_count_per_document=int(raw.get("question_count_per_document", 10)),
        max_chars_per_llm_call=int(raw.get("max_chars_per_llm_call", 6000)),
        timeout_sec=int(raw.get("timeout_sec", 180)),
        temperature=float(raw.get("temperature", 0.2)),
        max_tokens=int(raw.get("max_tokens", 8000)),
        use_json_mode=bool(raw.get("use_json_mode", True)),
        max_questions_per_llm_call=int(raw.get("max_questions_per_llm_call", 6)),
        llm_roles=llm_roles,
        review=review,
        review_test=review_test,
        providers=providers,
        excel_columns=list(raw.get("excel_columns", [])),
    )


def load_secrets(secrets_path: Path) -> dict[str, str]:
    raw = load_yaml(secrets_path)
    result: dict[str, str] = {}
    for provider_name, item in raw.items():
        if isinstance(item, dict):
            api_key = str(item.get("api_key", "")).strip()
            if api_key:
                result[str(provider_name)] = api_key
    return result
