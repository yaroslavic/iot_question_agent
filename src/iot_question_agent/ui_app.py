from __future__ import annotations

import json
import queue
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import requests
import yaml
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse

from .docx_reader import blocks_to_text, read_docx_blocks
from .excel_exporter import export_questions_xlsx
from .llm_client import call_chat_completions
from .main import (
    REVIEW_SYSTEM_PROMPT,
    REWRITE_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    build_generation_chunks,
    build_provider_index,
    build_review_prompt,
    build_rewrite_prompt,
    build_user_prompt,
    collect_review_items,
    distribute_questions,
    effective_forced_bad_questions_count,
    get_provider_or_raise,
    make_forced_bad_questions,
    normal_questions_target,
    provider_api_key,
    read_text,
    safe_name,
    save_json,
    save_text,
)
from .questions import collect_questions, deduplicate_questions
from .settings import AppConfig, ProviderConfig, load_config, load_secrets

APP_DIR = Path(__file__).resolve().parents[2]
CONFIG_PATH = APP_DIR / "config.yaml"
UPLOAD_DIR = APP_DIR / "input" / "uploaded"

QUESTION_COLUMNS = [
    "Вопрос",
    "Правильный ответ",
    "Неправильный ответ 1",
    "Неправильный ответ 2",
    "Раздел",
    "Пункт",
    "Пояснение правильного ответа",
    "Тема",
]

DEFAULT_MODEL_CATALOG = {
    "chatgpt": [
        {"id": "gpt-5.5", "description": "макс. качество"},
        {"id": "gpt-5.4", "description": "сильный баланс"},
        {"id": "gpt-5.4-mini", "description": "быстрее дешевле"},
        {"id": "gpt-5.4-nano", "description": "минимальная цена"},
        {"id": "gpt-4.1", "description": "стабильный текст"},
        {"id": "gpt-4.1-mini", "description": "быстрый текст"},
    ],
    "deepseek": [
        {"id": "deepseek-v4-pro", "description": "лучшее качество"},
        {"id": "deepseek-v4-flash", "description": "быстро дёшево"},
        {"id": "deepseek-chat", "description": "старое имя"},
        {"id": "deepseek-reasoner", "description": "старое reasoning"},
    ],
}

STATUS_LABELS = {
    "generated": "Сгенерирован",
    "reviewing": "На проверке",
    "accepted": "Хороший",
    "accepted_after_rewrite": "Исправлен",
    "needs_rewrite": "Плохой",
    "rejected": "Отклонён",
    "rewriting": "На переделке",
    "rewritten": "Переделан",
    "final_rejected": "Не принят",
    "generation_error": "Ошибка генерации",
    "review_error": "Ошибка проверки",
    "rewrite_error": "Ошибка переделки",
}

app = FastAPI(title="ИИ-агент вопросов по охране труда")


@dataclass
class Job:
    id: str
    file_path: Path
    events: queue.Queue[dict[str, Any]] = field(default_factory=queue.Queue)
    done: bool = False
    result: dict[str, Any] | None = None
    error: str | None = None


jobs: dict[str, Job] = {}


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        return {}
    return data


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False, width=120), encoding="utf-8")


def _provider_map(raw: dict[str, Any]) -> dict[str, Any]:
    providers = raw.get("providers")
    return providers if isinstance(providers, dict) else {}


def _model_catalog(raw: dict[str, Any]) -> dict[str, list[dict[str, str]]]:
    catalog = raw.get("model_catalog")
    result = dict(DEFAULT_MODEL_CATALOG)
    if isinstance(catalog, dict):
        for provider, items in catalog.items():
            normalized: list[dict[str, str]] = []
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, dict) and item.get("id"):
                        normalized.append(
                            {
                                "id": str(item.get("id", "")).strip(),
                                "description": str(item.get("description", "доступна через API")).strip(),
                            }
                        )
                    elif isinstance(item, str):
                        normalized.append({"id": item, "description": "доступна через API"})
            if normalized:
                result[str(provider)] = normalized
    return result


def _emit(job: Job, event_type: str, payload: dict[str, Any]) -> None:
    job.events.put({"type": event_type, "payload": payload})


def _log(job: Job, message: str, level: str = "info") -> None:
    _emit(job, "log", {"time": datetime.now().strftime("%H:%M:%S"), "message": message, "level": level})


def _sse(event: dict[str, Any]) -> str:
    return f"event: {event['type']}\ndata: {json.dumps(event['payload'], ensure_ascii=False)}\n\n"


def _safe_upload_name(name: str) -> str:
    base = Path(name).stem
    return safe_name(base) + ".docx"


def _preview_text(blocks: list[Any], max_chars: int = 12000) -> str:
    text = blocks_to_text(blocks)
    if len(text) > max_chars:
        return text[:max_chars] + "\n\n... [предпросмотр обрезан]"
    return text


def _known_description(provider_key: str, model_id: str, raw_config: dict[str, Any]) -> str:
    for item in _model_catalog(raw_config).get(provider_key, []):
        if item.get("id") == model_id:
            return item.get("description", "доступна через API")
    return "доступна через API"


def _fetch_provider_models(provider_key: str) -> list[dict[str, str]]:
    config = load_config(CONFIG_PATH)
    secrets = load_secrets(config.secrets_file) if config.secrets_file.exists() else {}
    provider_index = build_provider_index(config.providers)
    provider = get_provider_or_raise(provider_index, provider_key, "models")
    api_key = provider_api_key(secrets, provider)
    if not api_key:
        raise ValueError(f"API key for provider '{provider_key}' not found in secrets.yaml")

    url = provider.base_url.rstrip("/") + "/models"
    response = requests.get(
        url,
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()
    ids: list[str] = []
    for item in data.get("data", []):
        if isinstance(item, dict) and item.get("id"):
            ids.append(str(item["id"]))
    ids = sorted(set(ids))
    raw = _read_yaml(CONFIG_PATH)
    return [{"id": mid, "description": _known_description(provider_key, mid, raw)} for mid in ids]


def _ensure_model_in_catalog(raw: dict[str, Any], provider: str, model: str) -> None:
    if not model:
        return
    catalog = raw.setdefault("model_catalog", {})
    if not isinstance(catalog, dict):
        raw["model_catalog"] = {}
        catalog = raw["model_catalog"]
    items = catalog.setdefault(provider, [])
    if not isinstance(items, list):
        items = []
        catalog[provider] = items
    exists = False
    for item in items:
        if isinstance(item, dict) and item.get("id") == model:
            exists = True
            break
        if isinstance(item, str) and item == model:
            exists = True
            break
    if not exists:
        items.append({"id": model, "description": "пользовательская"})


def _update_config_from_ui(payload: dict[str, Any]) -> dict[str, Any]:
    raw = _read_yaml(CONFIG_PATH)

    question_count = int(payload.get("question_count_per_document", raw.get("question_count_per_document", 10)))
    raw["question_count_per_document"] = max(1, min(question_count, 200))

    raw.setdefault("input_dir", "input")
    raw.setdefault("output_dir", "output")
    raw.setdefault("prompts_dir", "prompts")
    raw.setdefault("secrets_file", "secrets.yaml")
    raw.setdefault("max_chars_per_llm_call", 6000)
    raw.setdefault("max_questions_per_llm_call", 6)
    raw.setdefault("timeout_sec", 180)
    raw.setdefault("temperature", 0.2)
    raw.setdefault("max_tokens", 8000)
    raw.setdefault("use_json_mode", True)

    generator_provider = str(payload.get("generator_provider", "chatgpt")).strip()
    reviewer_provider = str(payload.get("reviewer_provider", "deepseek")).strip()
    raw["llm_roles"] = {"generator": generator_provider, "reviewer": reviewer_provider}

    providers = raw.setdefault("providers", {})
    if not isinstance(providers, dict):
        providers = {}
        raw["providers"] = providers

    generator_model = str(payload.get("generator_model", "")).strip()
    reviewer_model = str(payload.get("reviewer_model", "")).strip()
    for provider_key, model in [(generator_provider, generator_model), (reviewer_provider, reviewer_model)]:
        if provider_key and model:
            provider_item = providers.setdefault(provider_key, {})
            if isinstance(provider_item, dict):
                provider_item["model"] = model
                provider_item.setdefault("enabled", True)
                provider_item.setdefault("prefix", provider_key)
                if provider_key == "chatgpt":
                    provider_item.setdefault("base_url", "https://api.openai.com/v1")
                    provider_item.setdefault("endpoint", "/responses")
                    provider_item.setdefault("reasoning_effort", "minimal")
                    provider_item.setdefault("max_tokens", 16000)
                elif provider_key == "deepseek":
                    provider_item.setdefault("base_url", "https://api.deepseek.com")
                    provider_item.setdefault("endpoint", "/chat/completions")
                    provider_item.setdefault("thinking", "disabled")
                    provider_item.setdefault("max_tokens", 8000)
            _ensure_model_in_catalog(raw, provider_key, model)

    review = raw.setdefault("review", {})
    if not isinstance(review, dict):
        review = {}
        raw["review"] = review
    review["enabled"] = bool(payload.get("review_enabled", True))
    review["max_rewrite_attempts"] = max(0, min(int(payload.get("max_rewrite_attempts", 1)), 5))
    review["save_review_files"] = True

    review_test = raw.setdefault("review_test", {})
    if not isinstance(review_test, dict):
        review_test = {}
        raw["review_test"] = review_test
    review_test.setdefault("enabled", False)
    review_test.setdefault("forced_bad_questions_count", 0)

    raw["excel_columns"] = QUESTION_COLUMNS
    raw.setdefault("ui", {"theme": "kuzbasscot", "preview_max_chars": 12000, "auto_open_browser": True})

    _write_yaml(CONFIG_PATH, raw)
    return raw


def _save_job_files(
    *,
    config: AppConfig,
    timestamp: str,
    doc_base: str,
    generator_provider: ProviderConfig,
    reviewer_provider: ProviderConfig | None,
    raw_items: list[dict[str, Any]],
    review_items: list[dict[str, Any]],
    rejected_items: list[dict[str, Any]],
    final_questions: list[dict[str, str]],
    errors: list[str],
) -> dict[str, str]:
    reviewer_prefix = reviewer_provider.prefix if reviewer_provider else "none"
    base_out = f"{timestamp}_gen-{generator_provider.prefix}_review-{reviewer_prefix}_{doc_base}"

    raw_path = config.output_dir / f"{base_out}_raw.json"
    review_path = config.output_dir / f"{base_out}_review.json"
    rejected_path = config.output_dir / f"{base_out}_rejected.json"
    xlsx_path = config.output_dir / f"{base_out}_questions.xlsx"
    error_path = config.output_dir / f"{base_out}_errors.txt"

    save_json(
        raw_path,
        {
            "generator_provider": generator_provider.prefix,
            "generator_model": generator_provider.model,
            "reviewer_provider": reviewer_prefix,
            "reviewer_model": reviewer_provider.model if reviewer_provider else "",
            "target_questions": config.question_count_per_document,
            "normal_questions_target": normal_questions_target(config),
            "forced_bad_questions_target": effective_forced_bad_questions_count(config),
            "items": raw_items,
            "questions": final_questions,
        },
    )
    save_json(review_path, {"items": review_items})
    save_json(rejected_path, {"items": rejected_items})
    export_questions_xlsx(xlsx_path, final_questions, config.excel_columns)
    if errors:
        save_text(error_path, "\n\n".join(errors))

    return {
        "xlsx": xlsx_path.name,
        "raw_json": raw_path.name,
        "review_json": review_path.name,
        "rejected_json": rejected_path.name,
        "errors_txt": error_path.name if errors else "",
    }


def _run_question_job(job: Job) -> None:
    try:
        config = load_config(CONFIG_PATH)
        config.output_dir.mkdir(parents=True, exist_ok=True)
        secrets = load_secrets(config.secrets_file)

        provider_index = build_provider_index(config.providers)
        generator_provider = get_provider_or_raise(provider_index, config.llm_roles.generator, "generator")
        reviewer_provider = None
        if config.review.enabled:
            reviewer_provider = get_provider_or_raise(provider_index, config.llm_roles.reviewer, "reviewer")

        generator_api_key = provider_api_key(secrets, generator_provider)
        if not generator_api_key:
            raise ValueError(f"Нет API-ключа для генератора '{generator_provider.prefix}' в secrets.yaml")

        reviewer_api_key = ""
        if reviewer_provider:
            reviewer_api_key = provider_api_key(secrets, reviewer_provider)
            if not reviewer_api_key:
                raise ValueError(f"Нет API-ключа для проверяющего '{reviewer_provider.prefix}' в secrets.yaml")

        generator_template = read_text(config.prompts_dir / "question_generator.md")
        reviewer_template = read_text(config.prompts_dir / "question_reviewer.md")
        rewriter_template = read_text(config.prompts_dir / "question_rewriter.md")

        blocks = read_docx_blocks(job.file_path)
        if not blocks:
            raise ValueError("Не удалось извлечь текстовые блоки из Word-файла")

        chunks = build_generation_chunks(blocks, config)
        forced_bad_total = effective_forced_bad_questions_count(config)
        generated_bad_total = 0
        normal_target = normal_questions_target(config)
        per_chunk_counts = distribute_questions(normal_target, len(chunks), config.max_questions_per_llm_call)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        doc_base = safe_name(job.file_path.stem)

        _log(job, f"Файл прочитан: {job.file_path.name}; блоков: {len(blocks)}; чанков: {len(chunks)}")
        _log(job, f"Генератор: {generator_provider.prefix} / {generator_provider.model}")
        if reviewer_provider:
            _log(job, f"Проверяющий: {reviewer_provider.prefix} / {reviewer_provider.model}")
        if forced_bad_total > 0:
            _log(job, f"Тест reviewer включён: обычных вопросов {normal_target}, искусственно плохих {forced_bad_total}", "warning")

        raw_items: list[dict[str, Any]] = []
        raw_review_items: list[dict[str, Any]] = []
        rejected_items: list[dict[str, Any]] = []
        final_questions: list[dict[str, str]] = []
        errors: list[str] = []
        question_counter = 0

        for chunk_index, chunk in enumerate(chunks, start=1):
            if len(final_questions) >= config.question_count_per_document:
                break
            requested_count = per_chunk_counts[chunk_index - 1] if chunk_index - 1 < len(per_chunk_counts) else 0
            bad_needed_this_chunk = max(0, forced_bad_total - generated_bad_total)
            if requested_count <= 0 and bad_needed_this_chunk <= 0:
                continue
            requested_count = min(requested_count, max(0, config.question_count_per_document - len(final_questions)))
            document_text = blocks_to_text(chunk)

            generated_questions: list[dict[str, str]] = []
            if requested_count > 0:
                _log(job, f"Чанк {chunk_index}/{len(chunks)}: генерация {requested_count} вопрос(ов)")
                gen_result = call_chat_completions(
                    generator_provider,
                    generator_api_key,
                    SYSTEM_PROMPT,
                    build_user_prompt(generator_template, document_text, requested_count),
                    timeout_sec=config.timeout_sec,
                    temperature=config.temperature,
                    max_tokens=generator_provider.max_tokens or config.max_tokens,
                    use_json_mode=config.use_json_mode,
                )
                raw_items.append(
                    {
                        "type": "generation",
                        "chunk_index": chunk_index,
                        "requested_questions": requested_count,
                        "ok": gen_result.ok,
                        "content": gen_result.content,
                        "data": gen_result.data,
                        "error": gen_result.error,
                    }
                )
                if not gen_result.ok or gen_result.data is None:
                    err = f"Чанк {chunk_index}: ошибка генерации: {gen_result.error}"
                    errors.append(err)
                    _log(job, err, "error")
                    continue

                try:
                    generated_questions = collect_questions(gen_result.data, config.excel_columns)
                except Exception as exc:
                    err = f"Чанк {chunk_index}: ошибка JSON-схемы генератора: {type(exc).__name__}: {exc}"
                    errors.append(err)
                    _log(job, err, "error")
                    continue

            if bad_needed_this_chunk > 0:
                injected = make_forced_bad_questions(bad_needed_this_chunk, config.excel_columns)
                generated_bad_total += len(injected)
                generated_questions.extend(injected)
                raw_items.append(
                    {
                        "type": "forced_bad_questions",
                        "chunk_index": chunk_index,
                        "requested_questions": len(injected),
                        "ok": True,
                        "content": "",
                        "data": {"questions": injected},
                        "error": "",
                    }
                )
                _log(job, f"Чанк {chunk_index}/{len(chunks)}: добавлено плохих тестовых вопросов {len(injected)}", "warning")

            pending: list[tuple[str, dict[str, str]]] = []
            for q in generated_questions[: requested_count + bad_needed_this_chunk]:
                question_counter += 1
                qid = f"q_{question_counter:04d}"
                pending.append((qid, q))
                _emit(job, "question_add", {"id": qid, "status": "generated", "question": q, "problems": []})

            if not reviewer_provider or not config.review.enabled:
                for qid, q in pending:
                    final_questions.append(q)
                    _emit(job, "question_update", {"id": qid, "status": "accepted", "problems": []})
                final_questions = deduplicate_questions(final_questions)[: config.question_count_per_document]
                continue

            for attempt in range(config.review.max_rewrite_attempts + 1):
                if not pending:
                    break

                for qid, _q in pending:
                    _emit(job, "question_update", {"id": qid, "status": "reviewing", "problems": []})
                _log(job, f"Чанк {chunk_index}: проверка {len(pending)} вопрос(ов), попытка {attempt + 1}")

                questions_only = [q for _qid, q in pending]
                review_result = call_chat_completions(
                    reviewer_provider,
                    reviewer_api_key,
                    REVIEW_SYSTEM_PROMPT,
                    build_review_prompt(reviewer_template, document_text, questions_only),
                    timeout_sec=config.timeout_sec,
                    temperature=config.temperature,
                    max_tokens=reviewer_provider.max_tokens or config.max_tokens,
                    use_json_mode=config.use_json_mode,
                )
                raw_review_items.append(
                    {
                        "type": "review",
                        "chunk_index": chunk_index,
                        "attempt": attempt,
                        "ok": review_result.ok,
                        "content": review_result.content,
                        "data": review_result.data,
                        "error": review_result.error,
                        "input_questions": questions_only,
                    }
                )
                if not review_result.ok or review_result.data is None:
                    err = f"Чанк {chunk_index}: ошибка проверки: {review_result.error}"
                    errors.append(err)
                    _log(job, err, "error")
                    for qid, q in pending:
                        rejected_items.append({"status": "review_error", "question": q, "problems": [review_result.error or "review error"]})
                        _emit(job, "question_update", {"id": qid, "status": "review_error", "problems": [review_result.error or "review error"]})
                    pending = []
                    break

                try:
                    review_items = collect_review_items(review_result.data, config.excel_columns)
                except Exception as exc:
                    err = f"Чанк {chunk_index}: ошибка JSON-схемы проверяющего: {type(exc).__name__}: {exc}"
                    errors.append(err)
                    _log(job, err, "error")
                    for qid, q in pending:
                        rejected_items.append({"status": "review_schema_error", "question": q, "problems": [str(exc)]})
                        _emit(job, "question_update", {"id": qid, "status": "review_error", "problems": [str(exc)]})
                    pending = []
                    break

                next_pending: list[tuple[str, dict[str, str]]] = []
                for item_index, (qid, original_q) in enumerate(pending):
                    if item_index < len(review_items):
                        review_item = review_items[item_index]
                    else:
                        review_item = {
                            "status": "needs_rewrite",
                            "problems": ["Проверяющий не вернул результат по этому вопросу"],
                            "rewrite_instruction": "Переписать вопрос строго по инструкции.",
                            "question": original_q,
                        }

                    status = str(review_item.get("status", "needs_rewrite"))
                    problems = review_item.get("problems", []) or []
                    reviewed_q = review_item.get("question") if isinstance(review_item.get("question"), dict) else original_q
                    if not reviewed_q.get("Вопрос"):
                        reviewed_q = original_q

                    if status == "accepted":
                        final_status = "accepted_after_rewrite" if attempt > 0 else "accepted"
                        final_questions.append(reviewed_q)
                        final_questions = deduplicate_questions(final_questions)[: config.question_count_per_document]
                        _emit(job, "question_replace", {"id": qid, "status": final_status, "question": reviewed_q, "problems": []})
                        continue

                    _emit(job, "question_update", {"id": qid, "status": status, "problems": problems})

                    if attempt >= config.review.max_rewrite_attempts:
                        rejected_items.append(
                            {
                                "status": status,
                                "question": reviewed_q,
                                "problems": problems,
                                "rewrite_instruction": review_item.get("rewrite_instruction", ""),
                            }
                        )
                        _emit(job, "question_update", {"id": qid, "status": "final_rejected", "problems": problems})
                        continue

                    _emit(job, "question_update", {"id": qid, "status": "rewriting", "problems": problems})
                    _log(job, f"Вопрос {qid}: отправлен на переделку")
                    rewrite_result = call_chat_completions(
                        generator_provider,
                        generator_api_key,
                        REWRITE_SYSTEM_PROMPT,
                        build_rewrite_prompt(rewriter_template, document_text, reviewed_q, review_item),
                        timeout_sec=config.timeout_sec,
                        temperature=config.temperature,
                        max_tokens=generator_provider.max_tokens or config.max_tokens,
                        use_json_mode=config.use_json_mode,
                    )
                    raw_review_items.append(
                        {
                            "type": "rewrite",
                            "chunk_index": chunk_index,
                            "attempt": attempt,
                            "question_id": qid,
                            "ok": rewrite_result.ok,
                            "content": rewrite_result.content,
                            "data": rewrite_result.data,
                            "error": rewrite_result.error,
                            "review_item": review_item,
                        }
                    )
                    if not rewrite_result.ok or rewrite_result.data is None:
                        problem = rewrite_result.error or "rewrite error"
                        errors.append(f"{qid}: ошибка переделки: {problem}")
                        rejected_items.append({"status": "rewrite_error", "question": reviewed_q, "problems": problems, "rewrite_error": problem})
                        _emit(job, "question_update", {"id": qid, "status": "rewrite_error", "problems": [problem]})
                        continue

                    try:
                        rewritten = collect_questions(rewrite_result.data, config.excel_columns)
                    except Exception as exc:
                        problem = f"{type(exc).__name__}: {exc}"
                        errors.append(f"{qid}: ошибка JSON-схемы переделки: {problem}")
                        rejected_items.append({"status": "rewrite_schema_error", "question": reviewed_q, "problems": [problem]})
                        _emit(job, "question_update", {"id": qid, "status": "rewrite_error", "problems": [problem]})
                        continue

                    if rewritten:
                        new_q = rewritten[0]
                        _emit(job, "question_replace", {"id": qid, "status": "rewritten", "question": new_q, "problems": []})
                        next_pending.append((qid, new_q))
                    else:
                        rejected_items.append({"status": "rewrite_empty", "question": reviewed_q, "problems": ["Переделка не вернула вопрос"]})
                        _emit(job, "question_update", {"id": qid, "status": "rewrite_error", "problems": ["Переделка не вернула вопрос"]})

                pending = next_pending

        final_questions = deduplicate_questions(final_questions)[: config.question_count_per_document]
        files = _save_job_files(
            config=config,
            timestamp=timestamp,
            doc_base=doc_base,
            generator_provider=generator_provider,
            reviewer_provider=reviewer_provider,
            raw_items=raw_items,
            review_items=raw_review_items,
            rejected_items=rejected_items,
            final_questions=final_questions,
            errors=errors,
        )
        result = {"questions_count": len(final_questions), "files": files, "errors": errors}
        job.result = result
        _log(job, f"Готово. Принято вопросов: {len(final_questions)}. Excel сохранён: {files['xlsx']}", "success")
        _emit(job, "done", result)
    except Exception as exc:
        job.error = f"{type(exc).__name__}: {exc}"
        _log(job, job.error, "error")
        _emit(job, "error", {"error": job.error})
    finally:
        job.done = True
        job.events.put({"type": "__close__", "payload": {}})


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return HTML


@app.get("/api/config")
def api_get_config() -> JSONResponse:
    raw = _read_yaml(CONFIG_PATH)
    providers = _provider_map(raw)
    roles = raw.get("llm_roles") if isinstance(raw.get("llm_roles"), dict) else {}
    review = raw.get("review") if isinstance(raw.get("review"), dict) else {}
    result = {
        "question_count_per_document": int(raw.get("question_count_per_document", 10)),
        "generator_provider": str(roles.get("generator", "chatgpt")),
        "reviewer_provider": str(roles.get("reviewer", "deepseek")),
        "generator_model": str(providers.get(str(roles.get("generator", "chatgpt")), {}).get("model", "gpt-5.5")),
        "reviewer_model": str(providers.get(str(roles.get("reviewer", "deepseek")), {}).get("model", "deepseek-v4-flash")),
        "review_enabled": bool(review.get("enabled", True)),
        "max_rewrite_attempts": int(review.get("max_rewrite_attempts", 1)),
        "review_test": raw.get("review_test", {"enabled": False, "forced_bad_questions_count": 0}) if isinstance(raw.get("review_test", {}), dict) else {"enabled": False, "forced_bad_questions_count": 0},
        "providers": sorted(list(providers.keys() or ["chatgpt", "deepseek"])),
        "model_catalog": _model_catalog(raw),
        "status_labels": STATUS_LABELS,
    }
    return JSONResponse(result)


@app.post("/api/config")
async def api_save_config(request: Request) -> JSONResponse:
    payload = await request.json()
    raw = _update_config_from_ui(payload)
    return JSONResponse({"ok": True, "config": raw})


@app.get("/api/models/{provider_key}")
def api_models(provider_key: str) -> JSONResponse:
    try:
        models = _fetch_provider_models(provider_key)
        return JSONResponse({"ok": True, "provider": provider_key, "models": models})
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"{type(exc).__name__}: {exc}")


@app.post("/api/upload")
async def api_upload(file: UploadFile = File(...)) -> JSONResponse:
    if not file.filename or not file.filename.lower().endswith(".docx"):
        raise HTTPException(status_code=400, detail="Нужен файл Word .docx")
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    filename = datetime.now().strftime("%Y%m%d_%H%M%S_") + _safe_upload_name(file.filename)
    path = UPLOAD_DIR / filename
    content = await file.read()
    path.write_bytes(content)

    try:
        blocks = read_docx_blocks(path)
        preview = _preview_text(blocks)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Не удалось прочитать DOCX: {type(exc).__name__}: {exc}")

    return JSONResponse(
        {
            "ok": True,
            "file_id": filename,
            "filename": file.filename,
            "saved_as": filename,
            "size_bytes": len(content),
            "blocks_count": len(blocks),
            "preview": preview,
        }
    )


@app.post("/api/jobs/start")
async def api_start_job(request: Request) -> JSONResponse:
    payload = await request.json()
    file_id = str(payload.get("file_id", "")).strip()
    if not file_id:
        raise HTTPException(status_code=400, detail="file_id is required")
    path = (UPLOAD_DIR / file_id).resolve()
    if not path.exists() or UPLOAD_DIR.resolve() not in path.parents:
        raise HTTPException(status_code=404, detail="Файл не найден")

    job_id = uuid.uuid4().hex[:12]
    job = Job(id=job_id, file_path=path)
    jobs[job_id] = job
    t = threading.Thread(target=_run_question_job, args=(job,), daemon=True)
    t.start()
    return JSONResponse({"ok": True, "job_id": job_id})


@app.get("/api/jobs/{job_id}/events")
def api_job_events(job_id: str) -> StreamingResponse:
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    def gen():
        yield _sse({"type": "hello", "payload": {"job_id": job_id}})
        while True:
            try:
                event = job.events.get(timeout=15)
            except queue.Empty:
                yield ": ping\n\n"
                continue
            if event["type"] == "__close__":
                yield _sse({"type": "closed", "payload": {"job_id": job_id}})
                break
            yield _sse(event)

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/download/{filename}")
def download(filename: str) -> FileResponse:
    if not re.match(r"^[A-Za-zА-Яа-я0-9_.\-]+$", filename):
        raise HTTPException(status_code=400, detail="Bad filename")
    path = (APP_DIR / "output" / filename).resolve()
    if not path.exists() or (APP_DIR / "output").resolve() not in path.parents:
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path, filename=filename)


HTML = r'''
<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>ИИ-агент вопросов по охране труда</title>
  <style>
    :root {
      --ink: #172033;
      --blue: #12365f;
      --blue2: #0c2748;
      --orange: #f28c28;
      --green: #22a06b;
      --soft-green: #e6f6ee;
      --pink: #fde7ee;
      --red: #b42318;
      --yellow: #fff6d6;
      --orange-soft: #fff0df;
      --gray: #f4f6f8;
      --line: #d9e1ea;
      --muted: #637083;
      --white: #ffffff;
      --shadow: 0 14px 30px rgba(18, 54, 95, 0.12);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      color: var(--ink);
      font-family: "Segoe UI", Arial, sans-serif;
      background: linear-gradient(135deg, #f5f8fb 0%, #eef3f7 100%);
    }
    .topbar {
      background: linear-gradient(135deg, var(--blue2), var(--blue));
      color: white;
      padding: 22px 34px;
      box-shadow: var(--shadow);
    }
    .topbar h1 { margin: 0; font-size: 24px; font-weight: 700; }
    .topbar p { margin: 7px 0 0; color: #cfe1f5; }
    .wrap { max-width: 1500px; margin: 22px auto 60px; padding: 0 18px; }
    .grid { display: grid; grid-template-columns: 420px 1fr; gap: 18px; align-items: start; }
    .card {
      background: var(--white);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 18px;
      box-shadow: var(--shadow);
    }
    .card h2 { margin: 0 0 14px; font-size: 18px; color: var(--blue2); }
    label { display: block; margin: 12px 0 6px; font-weight: 650; color: #25344d; }
    input[type="number"], select, input[type="file"] {
      width: 100%; padding: 11px 12px; border: 1px solid var(--line); border-radius: 12px; background: white;
      font-size: 14px; outline: none;
    }
    input:focus, select:focus { border-color: #7ba6d6; box-shadow: 0 0 0 3px rgba(18, 54, 95, .08); }
    .row2 { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
    .btns { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 16px; }
    button {
      border: 0; border-radius: 12px; padding: 11px 15px; font-weight: 700; cursor: pointer;
      background: var(--blue); color: white; transition: .15s; box-shadow: 0 8px 18px rgba(18,54,95,.16);
    }
    button:hover { transform: translateY(-1px); filter: brightness(1.03); }
    button.secondary { background: #e9eef5; color: var(--blue2); box-shadow: none; }
    button.orange { background: var(--orange); color: #241400; }
    button.green { background: var(--green); }
    button:disabled { opacity: .55; cursor: not-allowed; transform: none; }
    .hint { font-size: 12px; color: var(--muted); line-height: 1.35; margin-top: 6px; }
    .filebox { margin-top: 12px; padding: 12px; border-radius: 14px; background: var(--gray); border: 1px dashed #b8c4d1; }
    .preview {
      max-height: 360px; overflow: auto; white-space: pre-wrap; font: 12px/1.5 Consolas, monospace;
      background: #101a2a; color: #d9ecff; padding: 14px; border-radius: 14px;
    }
    .log {
      height: 210px; overflow: auto; background: #0f1b2d; color: #d6e7fb; border-radius: 14px; padding: 12px;
      font: 13px/1.45 Consolas, monospace;
    }
    .log div { margin: 0 0 5px; }
    .log .error { color: #ffb4ab; }
    .log .success { color: #99f6c8; }
    .summary { display: flex; gap: 10px; flex-wrap: wrap; margin: 12px 0; }
    .pill { padding: 7px 10px; border-radius: 999px; background: #eef4fb; color: var(--blue2); font-size: 13px; font-weight: 650; }
    .tablewrap { overflow: auto; max-height: 680px; border: 1px solid var(--line); border-radius: 14px; }
    table { border-collapse: collapse; width: 100%; min-width: 1320px; font-size: 13px; }
    thead th { position: sticky; top: 0; background: var(--blue); color: white; z-index: 1; }
    th, td { border-bottom: 1px solid var(--line); padding: 9px 10px; vertical-align: top; text-align: left; }
    tr.status-generated { background: #fff; }
    tr.status-reviewing { background: var(--yellow); }
    tr.status-accepted { background: var(--soft-green); }
    tr.status-accepted_after_rewrite { background: #d8f8e8; }
    tr.status-needs_rewrite, tr.status-rejected, tr.status-final_rejected { background: var(--pink); }
    tr.status-rewriting, tr.status-rewritten { background: var(--orange-soft); }
    tr.status-review_error, tr.status-rewrite_error, tr.status-generation_error { background: #ffe4e1; }
    .status-badge { display: inline-block; padding: 4px 8px; border-radius: 999px; font-weight: 700; background: white; color: var(--blue2); border: 1px solid rgba(0,0,0,.08); }
    .problems { color: #7a271a; max-width: 280px; }
    .downloads a { display: inline-block; margin: 6px 8px 0 0; color: var(--blue); font-weight: 700; text-decoration: none; }
    .downloads a:hover { text-decoration: underline; }
    @media (max-width: 1050px) { .grid { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <div class="topbar">
    <h1>ИИ-агент генерации обучающих вопросов по охране труда</h1>
    <p>Word-инструкция → вопросы → проверка второй LLM → переделка плохих → Excel</p>
  </div>

  <div class="wrap">
    <div class="grid">
      <div class="card">
        <h2>Настройки</h2>
        <label>Количество вопросов</label>
        <input id="questionCount" type="number" min="1" max="200" value="10" />
        <div class="hint">По умолчанию 10. Параметр сохраняется в config.yaml.</div>

        <div class="row2">
          <div>
            <label>LLM генератор</label>
            <select id="generatorProvider"></select>
          </div>
          <div>
            <label>Модель генератора</label>
            <select id="generatorModel"></select>
          </div>
        </div>

        <div class="row2">
          <div>
            <label>LLM проверяющий</label>
            <select id="reviewerProvider"></select>
          </div>
          <div>
            <label>Модель проверяющего</label>
            <select id="reviewerModel"></select>
          </div>
        </div>

        <label>Попытки переделки плохого вопроса</label>
        <input id="rewriteAttempts" type="number" min="0" max="5" value="1" />

        <div class="btns">
          <button id="saveConfigBtn" class="secondary">Сохранить настройки</button>
          <button id="refreshModelsBtn" class="secondary">Обновить модели API</button>
        </div>
        <div class="hint">Ключи API не показываются в интерфейсе и остаются в secrets.yaml.</div>

        <h2 style="margin-top:22px">Загрузка инструкции</h2>
        <input id="docxFile" type="file" accept=".docx" />
        <div class="btns">
          <button id="uploadBtn" class="orange">Загрузить Word</button>
          <button id="startBtn" class="green" disabled>Сгенерировать</button>
        </div>
        <div id="fileInfo" class="filebox">Файл не загружен.</div>
      </div>

      <div class="card">
        <h2>Предпросмотр инструкции</h2>
        <div id="preview" class="preview">Загрузите .docx файл, чтобы увидеть извлечённый текст и пункты.</div>
      </div>
    </div>

    <div class="card" style="margin-top:18px">
      <h2>Работа программы</h2>
      <div class="summary">
        <span class="pill" id="statGenerated">Сгенерировано: 0</span>
        <span class="pill" id="statAccepted">Хороших: 0</span>
        <span class="pill" id="statBad">Плохих/на переделке: 0</span>
        <span class="pill" id="statFinal">Итог: 0</span>
      </div>
      <div id="log" class="log"></div>
      <div id="downloads" class="downloads"></div>
    </div>

    <div class="card" style="margin-top:18px">
      <h2>Таблица вопросов</h2>
      <div class="tablewrap">
        <table>
          <thead>
            <tr>
              <th>№</th><th>Статус</th><th>Вопрос</th><th>Правильный ответ</th><th>Неправильный ответ 1</th><th>Неправильный ответ 2</th><th>Раздел</th><th>Пункт</th><th>Пояснение правильного ответа</th><th>Тема</th><th>Почему плохой</th>
            </tr>
          </thead>
          <tbody id="questionsBody"></tbody>
        </table>
      </div>
    </div>
  </div>

<script>
let config = null;
let uploadedFileId = null;
let rows = new Map();
let counters = {generated:0, accepted:0, bad:0, final:0};

const $ = (id) => document.getElementById(id);
const statusLabels = () => (config && config.status_labels) || {};

function esc(s) {
  return String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}
function log(message, level='info') {
  const div = document.createElement('div');
  div.className = level;
  div.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
  $('log').appendChild(div);
  $('log').scrollTop = $('log').scrollHeight;
}
function providerTitle(p) {
  if (p === 'chatgpt') return 'ChatGPT / OpenAI';
  if (p === 'deepseek') return 'DeepSeek';
  return p;
}
function fillProviderSelect(el, selected) {
  const providers = config.providers && config.providers.length ? config.providers : ['chatgpt','deepseek'];
  el.innerHTML = providers.map(p => `<option value="${esc(p)}">${esc(providerTitle(p))}</option>`).join('');
  el.value = selected || providers[0];
}
function fillModelSelect(el, provider, selected) {
  const catalog = (config.model_catalog && config.model_catalog[provider]) || [];
  let items = catalog.slice();
  if (selected && !items.some(x => x.id === selected)) items.unshift({id:selected, description:'из config.yaml'});
  el.innerHTML = items.map(m => `<option value="${esc(m.id)}">${esc(m.id)} (${esc(m.description || 'доступна')})</option>`).join('');
  if (selected) el.value = selected;
}
async function loadConfig() {
  const r = await fetch('/api/config');
  config = await r.json();
  $('questionCount').value = config.question_count_per_document || 10;
  $('rewriteAttempts').value = config.max_rewrite_attempts ?? 1;
  fillProviderSelect($('generatorProvider'), config.generator_provider);
  fillProviderSelect($('reviewerProvider'), config.reviewer_provider);
  fillModelSelect($('generatorModel'), $('generatorProvider').value, config.generator_model);
  fillModelSelect($('reviewerModel'), $('reviewerProvider').value, config.reviewer_model);
}
async function saveConfig() {
  const payload = {
    question_count_per_document: Number($('questionCount').value || 10),
    generator_provider: $('generatorProvider').value,
    generator_model: $('generatorModel').value,
    reviewer_provider: $('reviewerProvider').value,
    reviewer_model: $('reviewerModel').value,
    review_enabled: true,
    max_rewrite_attempts: Number($('rewriteAttempts').value || 1)
  };
  const r = await fetch('/api/config', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload)});
  if (!r.ok) throw new Error(await r.text());
  log('Настройки сохранены в config.yaml', 'success');
  await loadConfig();
}
async function refreshModels() {
  await saveConfig();
  for (const [provider, selectId, current] of [
    [$('generatorProvider').value, 'generatorModel', $('generatorModel').value],
    [$('reviewerProvider').value, 'reviewerModel', $('reviewerModel').value]
  ]) {
    try {
      const r = await fetch('/api/models/' + encodeURIComponent(provider));
      if (!r.ok) throw new Error(await r.text());
      const data = await r.json();
      config.model_catalog[provider] = data.models;
      fillModelSelect($(selectId), provider, current);
      log(`Список моделей ${providerTitle(provider)} обновлён: ${data.models.length}`, 'success');
    } catch (e) {
      log(`Не удалось обновить модели ${provider}: ${e.message}`, 'error');
    }
  }
}
async function uploadFile() {
  const f = $('docxFile').files[0];
  if (!f) { alert('Выберите .docx файл'); return; }
  const fd = new FormData();
  fd.append('file', f);
  const r = await fetch('/api/upload', {method:'POST', body:fd});
  if (!r.ok) throw new Error(await r.text());
  const data = await r.json();
  uploadedFileId = data.file_id;
  $('fileInfo').innerHTML = `<b>${esc(data.filename)}</b><br>Размер: ${data.size_bytes} байт<br>Блоков инструкции: ${data.blocks_count}<br>Сохранён как: ${esc(data.saved_as)}`;
  $('preview').textContent = data.preview || '';
  $('startBtn').disabled = false;
  log(`Файл загружен: ${data.filename}; блоков: ${data.blocks_count}`, 'success');
}
function resetTable() {
  rows.clear();
  $('questionsBody').innerHTML = '';
  $('downloads').innerHTML = '';
  counters = {generated:0, accepted:0, bad:0, final:0};
  updateCounters();
}
function updateCounters() {
  let generated=0, accepted=0, bad=0, final=0;
  rows.forEach(({status}) => {
    generated++;
    if (status === 'accepted' || status === 'accepted_after_rewrite') accepted++;
    if (['needs_rewrite','rejected','rewriting','rewritten','final_rejected','review_error','rewrite_error'].includes(status)) bad++;
    if (status === 'accepted' || status === 'accepted_after_rewrite') final++;
  });
  $('statGenerated').textContent = `Сгенерировано: ${generated}`;
  $('statAccepted').textContent = `Хороших: ${accepted}`;
  $('statBad').textContent = `Плохих/на переделке: ${bad}`;
  $('statFinal').textContent = `Итог: ${final}`;
}
function rowHtml(id, n, q, status, problems) {
  const label = statusLabels()[status] || status;
  return `<tr id="row_${id}" class="status-${esc(status)}">
    <td>${n}</td><td><span class="status-badge">${esc(label)}</span></td>
    <td>${esc(q['Вопрос'])}</td><td>${esc(q['Правильный ответ'])}</td><td>${esc(q['Неправильный ответ 1'])}</td><td>${esc(q['Неправильный ответ 2'])}</td>
    <td>${esc(q['Раздел'])}</td><td>${esc(q['Пункт'])}</td><td>${esc(q['Пояснение правильного ответа'])}</td><td>${esc(q['Тема'])}</td>
    <td class="problems">${esc((problems||[]).join('; '))}</td>
  </tr>`;
}
function addQuestion(ev) {
  const n = rows.size + 1;
  rows.set(ev.id, {n, question: ev.question, status: ev.status, problems: ev.problems || []});
  $('questionsBody').insertAdjacentHTML('beforeend', rowHtml(ev.id, n, ev.question, ev.status, ev.problems));
  updateCounters();
}
function updateQuestion(ev, replace=false) {
  const old = rows.get(ev.id);
  if (!old) return;
  const q = replace && ev.question ? ev.question : old.question;
  const status = ev.status || old.status;
  const problems = ev.problems || [];
  rows.set(ev.id, {n: old.n, question: q, status, problems});
  const el = $('row_' + ev.id);
  if (el) el.outerHTML = rowHtml(ev.id, old.n, q, status, problems);
  updateCounters();
}
async function startJob() {
  if (!uploadedFileId) { alert('Сначала загрузите Word-файл'); return; }
  await saveConfig();
  resetTable();
  $('startBtn').disabled = true;
  const r = await fetch('/api/jobs/start', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({file_id:uploadedFileId})});
  if (!r.ok) throw new Error(await r.text());
  const data = await r.json();
  log('Запущена генерация: job ' + data.job_id, 'success');
  const es = new EventSource('/api/jobs/' + data.job_id + '/events');
  es.addEventListener('log', e => { const d = JSON.parse(e.data); log(d.message, d.level); });
  es.addEventListener('question_add', e => addQuestion(JSON.parse(e.data)));
  es.addEventListener('question_update', e => updateQuestion(JSON.parse(e.data), false));
  es.addEventListener('question_replace', e => updateQuestion(JSON.parse(e.data), true));
  es.addEventListener('done', e => {
    const d = JSON.parse(e.data);
    const files = d.files || {};
    let html = '<b>Файлы:</b> ';
    for (const [k,v] of Object.entries(files)) {
      if (v) html += `<a href="/download/${encodeURIComponent(v)}" target="_blank">${esc(k)}</a>`;
    }
    $('downloads').innerHTML = html;
    $('startBtn').disabled = false;
  });
  es.addEventListener('error', e => { try { log(JSON.parse(e.data).error, 'error'); } catch{} $('startBtn').disabled = false; });
  es.addEventListener('closed', () => { es.close(); $('startBtn').disabled = false; });
}

$('generatorProvider').addEventListener('change', () => fillModelSelect($('generatorModel'), $('generatorProvider').value, null));
$('reviewerProvider').addEventListener('change', () => fillModelSelect($('reviewerModel'), $('reviewerProvider').value, null));
$('saveConfigBtn').addEventListener('click', () => saveConfig().catch(e => log(e.message, 'error')));
$('refreshModelsBtn').addEventListener('click', () => refreshModels().catch(e => log(e.message, 'error')));
$('uploadBtn').addEventListener('click', () => uploadFile().catch(e => log(e.message, 'error')));
$('startBtn').addEventListener('click', () => startJob().catch(e => { log(e.message, 'error'); $('startBtn').disabled = false; }));

loadConfig().then(() => log('Интерфейс готов. Загрузите Word-инструкцию.')).catch(e => log(e.message, 'error'));
</script>
</body>
</html>
'''
