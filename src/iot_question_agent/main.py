from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
import re
from typing import Any

from .docx_reader import blocks_to_text, read_docx_blocks, split_blocks_by_chars
from .excel_exporter import export_questions_xlsx
from .llm_client import call_chat_completions
from .questions import collect_questions, deduplicate_questions, normalize_question
from .settings import AppConfig, ProviderConfig, load_config, load_secrets


SYSTEM_PROMPT = (
    "Ты помогаешь создавать обучающие вопросы по охране труда. "
    "Работай строго по предоставленной инструкции и возвращай только валидный JSON."
)

REVIEW_SYSTEM_PROMPT = (
    "Ты независимо проверяешь качество обучающих вопросов по охране труда. "
    "Работай строго по предоставленной инструкции и возвращай только валидный JSON."
)

REWRITE_SYSTEM_PROMPT = (
    "Ты исправляешь обучающие вопросы по охране труда по замечаниям проверяющего. "
    "Работай строго по предоставленной инструкции и возвращай только валидный JSON."
)


def safe_name(name: str) -> str:
    name = re.sub(r"[^A-Za-zА-Яа-я0-9_.-]+", "_", name, flags=re.UNICODE)
    return name.strip("._") or "document"


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def to_json_text(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)


def provider_api_key(secrets: dict[str, str], provider: ProviderConfig) -> str:
    return secrets.get(provider.name) or secrets.get(provider.prefix) or ""


def build_provider_index(providers: list[ProviderConfig]) -> dict[str, ProviderConfig]:
    result: dict[str, ProviderConfig] = {}
    for p in providers:
        result[p.name] = p
        result[p.prefix] = p
    return result


def get_provider_or_raise(provider_index: dict[str, ProviderConfig], key: str, role_name: str) -> ProviderConfig:
    provider = provider_index.get(key)
    if provider is None:
        known = ", ".join(sorted(provider_index.keys()))
        raise ValueError(f"Unknown {role_name} provider '{key}' in config.yaml. Known providers: {known}")
    if not provider.model:
        raise ValueError(f"Provider '{key}' has empty model in config.yaml")
    if not provider.base_url:
        raise ValueError(f"Provider '{key}' has empty base_url in config.yaml")
    return provider


def distribute_questions(total: int, chunks_count: int, max_per_call: int) -> list[int]:
    """
    Возвращает список длиной chunks_count.
    Сумма значений не превышает total, а значение на чанк не превышает max_per_call.
    Если чанков больше, чем вопросов, часть чанков получает 0 и будет пропущена.
    """
    if chunks_count <= 0 or total <= 0:
        return [0] * max(0, chunks_count)

    max_per_call = max(1, max_per_call)
    result = [0] * chunks_count
    remaining = total
    idx = 0

    while remaining > 0 and any(v < max_per_call for v in result):
        if result[idx % chunks_count] < max_per_call:
            result[idx % chunks_count] += 1
            remaining -= 1
        idx += 1

    return result




def effective_forced_bad_questions_count(config: AppConfig) -> int:
    """
    Количество заведомо плохих вопросов, которые входят в общий лимит вопросов.
    Например: question_count_per_document=10 и forced_bad_questions_count=1
    означает 9 обычных вопросов + 1 тестовый плохой вопрос.
    """
    if not config.review.enabled or not config.review_test.enabled:
        return 0
    return max(0, min(config.review_test.forced_bad_questions_count, config.question_count_per_document))


def normal_questions_target(config: AppConfig) -> int:
    return max(0, config.question_count_per_document - effective_forced_bad_questions_count(config))


def make_forced_bad_questions(count: int, columns: list[str]) -> list[dict[str, str]]:
    """
    Создаёт искусственные плохие вопросы для проверки reviewer.
    Они намеренно не основаны на инструкции или имеют методические дефекты.
    Эти вопросы нужны только для контроля работы проверяющей LLM.
    """
    samples = [
        {
            "Вопрос": "Какого цвета должна быть каска работника согласно инструкции?",
            "Правильный ответ": "Красного цвета.",
            "Неправильный ответ 1": "Синего цвета.",
            "Неправильный ответ 2": "Зелёного цвета.",
            "Раздел": "ТЕСТ REVIEWER — ИСКУССТВЕННО ПЛОХОЙ ВОПРОС",
            "Пункт": "TEST-1",
            "Пояснение правильного ответа": "Это специально добавленное требование, которого обычно нет в инструкции. Проверяющий должен отклонить вопрос как не основанный на тексте.",
            "Тема": "Тест проверки reviewer: придуманное требование",
        },
        {
            "Вопрос": "Что безопаснее сделать работнику?",
            "Правильный ответ": "Соблюдать инструкцию и не получать травму.",
            "Неправильный ответ 1": "Нарушить инструкцию и получить травму.",
            "Неправильный ответ 2": "Игнорировать любые требования безопасности.",
            "Раздел": "ТЕСТ REVIEWER — ИСКУССТВЕННО ПЛОХОЙ ВОПРОС",
            "Пункт": "TEST-2",
            "Пояснение правильного ответа": "Вопрос слишком очевидный и не проверяет конкретный пункт инструкции.",
            "Тема": "Тест проверки reviewer: слишком очевидный вопрос",
        },
        {
            "Вопрос": "Какую температуру должна иметь рабочая зона перед началом работ?",
            "Правильный ответ": "Ровно 22 °C.",
            "Неправильный ответ 1": "Ровно 18 °C.",
            "Неправильный ответ 2": "Ровно 30 °C.",
            "Раздел": "ТЕСТ REVIEWER — ИСКУССТВЕННО ПЛОХОЙ ВОПРОС",
            "Пункт": "TEST-3",
            "Пояснение правильного ответа": "Это числовое требование специально придумано для проверки того, что reviewer сверяет вопрос с текстом инструкции.",
            "Тема": "Тест проверки reviewer: вымышленное числовое требование",
        },
        {
            "Вопрос": "Что должен сделать работник перед началом работы?",
            "Правильный ответ": "Получить задание у руководителя.",
            "Неправильный ответ 1": "Проверить исправность оборудования и инструмента.",
            "Неправильный ответ 2": "Ознакомиться с замечаниями предыдущей смены.",
            "Раздел": "ТЕСТ REVIEWER — ИСКУССТВЕННО ПЛОХОЙ ВОПРОС",
            "Пункт": "TEST-4",
            "Пояснение правильного ответа": "В вопросе несколько вариантов могут быть правильными для типовой инструкции; reviewer должен отметить неоднозначность.",
            "Тема": "Тест проверки reviewer: несколько потенциально правильных ответов",
        },
        {
            "Вопрос": "Какой пароль должен использовать работник для входа в систему охраны труда?",
            "Правильный ответ": "Пароль должен состоять из 12 символов.",
            "Неправильный ответ 1": "Пароль должен состоять из 6 символов.",
            "Неправильный ответ 2": "Пароль не нужен.",
            "Раздел": "ТЕСТ REVIEWER — ИСКУССТВЕННО ПЛОХОЙ ВОПРОС",
            "Пункт": "TEST-5",
            "Пояснение правильного ответа": "Вопрос не относится к инструкции по охране труда и добавляет внешнее IT-требование.",
            "Тема": "Тест проверки reviewer: не по теме инструкции",
        },
    ]
    result: list[dict[str, str]] = []
    for idx in range(count):
        q = dict(samples[idx % len(samples)])
        if count > len(samples):
            q["Пункт"] = f"TEST-{idx + 1}"
        result.append(normalize_question(q, columns))
    return result

def build_generation_chunks(blocks: list[Any], config: AppConfig) -> list[list[Any]]:
    """
    Делим документ так, чтобы один API-вызов не просил слишком много вопросов.
    Это защищает DeepSeek от обрезанного JSON, а OpenAI GPT-5 от расходования
    всего max_completion_tokens на hidden reasoning без видимого ответа.
    """
    desired_calls = max(
        1,
        math.ceil(max(1, normal_questions_target(config)) / max(1, config.max_questions_per_llm_call)),
    )
    full_text_len = max(1, len(blocks_to_text(blocks)))
    adaptive_chars = min(
        config.max_chars_per_llm_call,
        max(1200, math.ceil(full_text_len / desired_calls * 1.15)),
    )

    for candidate in [adaptive_chars, 6000, 4500, 3200, 2200, 1500, 1200]:
        candidate = min(candidate, config.max_chars_per_llm_call)
        chunks = split_blocks_by_chars(blocks, candidate)
        if len(chunks) >= desired_calls or candidate <= 1200:
            return chunks
    return split_blocks_by_chars(blocks, adaptive_chars)


def build_user_prompt(template: str, document_text: str, question_count: int) -> str:
    return (
        template.replace("{DOCUMENT_TEXT}", document_text)
        .replace("{QUESTION_COUNT}", str(question_count))
    )


def build_review_prompt(template: str, document_text: str, questions: list[dict[str, str]]) -> str:
    return (
        template.replace("{DOCUMENT_TEXT}", document_text)
        .replace("{QUESTIONS_JSON}", to_json_text({"questions": questions}))
    )


def build_rewrite_prompt(
    template: str,
    document_text: str,
    bad_question: dict[str, str],
    review_item: dict[str, Any],
) -> str:
    return (
        template.replace("{DOCUMENT_TEXT}", document_text)
        .replace("{BAD_QUESTION_JSON}", to_json_text(bad_question))
        .replace("{REVIEW_JSON}", to_json_text(review_item))
    )


def collect_review_items(data: dict[str, Any], columns: list[str]) -> list[dict[str, Any]]:
    raw_items = data.get("items", [])
    if not isinstance(raw_items, list):
        raise ValueError("JSON field 'items' must be a list")

    result: list[dict[str, Any]] = []
    for raw in raw_items:
        if not isinstance(raw, dict):
            continue
        status = str(raw.get("status", "")).strip().lower()
        if status not in {"accepted", "needs_rewrite", "rejected"}:
            status = "needs_rewrite"

        question_raw = raw.get("question", {})
        question = normalize_question(question_raw if isinstance(question_raw, dict) else {}, columns)

        problems_raw = raw.get("problems", [])
        if isinstance(problems_raw, list):
            problems = [str(p).strip() for p in problems_raw if str(p).strip()]
        elif problems_raw:
            problems = [str(problems_raw).strip()]
        else:
            problems = []

        result.append(
            {
                "status": status,
                "problems": problems,
                "rewrite_instruction": str(raw.get("rewrite_instruction", "")).strip(),
                "question": question,
            }
        )
    return result


def review_and_rewrite_questions(
    *,
    config: AppConfig,
    generator_provider: ProviderConfig,
    generator_api_key: str,
    reviewer_provider: ProviderConfig,
    reviewer_api_key: str,
    document_text: str,
    questions: list[dict[str, str]],
    reviewer_template: str,
    rewriter_template: str,
    chunk_index: int,
    raw_review_items: list[dict[str, Any]],
    rejected_items: list[dict[str, Any]],
    errors: list[str],
) -> list[dict[str, str]]:
    """
    Проверку выполняет отдельная LLM из config.llm_roles.reviewer.
    Переделку выполняет исходная LLM-генератор, затем результат снова проверяет reviewer.
    """
    accepted: list[dict[str, str]] = []
    pending = list(questions)

    for attempt in range(config.review.max_rewrite_attempts + 1):
        if not pending:
            break

        review_prompt = build_review_prompt(reviewer_template, document_text, pending)
        review_result = call_chat_completions(
            reviewer_provider,
            reviewer_api_key,
            REVIEW_SYSTEM_PROMPT,
            review_prompt,
            timeout_sec=config.timeout_sec,
            temperature=config.temperature,
            max_tokens=reviewer_provider.max_tokens or config.max_tokens,
            use_json_mode=config.use_json_mode,
        )

        raw_review_items.append(
            {
                "chunk_index": chunk_index,
                "attempt": attempt,
                "type": "review",
                "generator_provider": generator_provider.prefix,
                "reviewer_provider": reviewer_provider.prefix,
                "reviewer_model": reviewer_provider.model,
                "ok": review_result.ok,
                "content": review_result.content,
                "data": review_result.data,
                "error": review_result.error,
                "input_questions": pending,
            }
        )

        if not review_result.ok or review_result.data is None:
            errors.append(
                f"chunk {chunk_index}: review attempt {attempt}: "
                f"{reviewer_provider.prefix} error: {review_result.error}"
            )
            rejected_items.extend(
                {
                    "chunk_index": chunk_index,
                    "attempt": attempt,
                    "status": "review_failed",
                    "question": q,
                    "problems": [review_result.error or "review failed"],
                }
                for q in pending
            )
            break

        try:
            review_items = collect_review_items(review_result.data, config.excel_columns)
        except Exception as exc:
            errors.append(f"chunk {chunk_index}: review schema error: {type(exc).__name__}: {exc}")
            rejected_items.extend(
                {
                    "chunk_index": chunk_index,
                    "attempt": attempt,
                    "status": "review_schema_error",
                    "question": q,
                    "problems": [str(exc)],
                }
                for q in pending
            )
            break

        bad_items: list[dict[str, Any]] = []
        for item_index, item in enumerate(review_items):
            q = item.get("question", {})
            if item.get("status") == "accepted" and q.get("Вопрос") and q.get("Правильный ответ"):
                accepted.append(q)
            else:
                if not q.get("Вопрос") and item_index < len(pending):
                    item["question"] = pending[item_index]
                bad_items.append(item)

        if len(review_items) < len(pending):
            for missed in pending[len(review_items) :]:
                bad_items.append(
                    {
                        "status": "needs_rewrite",
                        "problems": ["Reviewer did not return this question"],
                        "rewrite_instruction": "Переписать вопрос строго по инструкции и вернуть полный набор полей.",
                        "question": missed,
                    }
                )

        if not bad_items:
            break

        if attempt >= config.review.max_rewrite_attempts:
            for item in bad_items:
                rejected_items.append(
                    {
                        "chunk_index": chunk_index,
                        "attempt": attempt,
                        "status": item.get("status"),
                        "question": item.get("question", {}),
                        "problems": item.get("problems", []),
                        "rewrite_instruction": item.get("rewrite_instruction", ""),
                    }
                )
            break

        next_pending: list[dict[str, str]] = []
        for bad_index, bad_item in enumerate(bad_items, start=1):
            bad_question = bad_item.get("question", {})
            rewrite_prompt = build_rewrite_prompt(rewriter_template, document_text, bad_question, bad_item)
            rewrite_result = call_chat_completions(
                generator_provider,
                generator_api_key,
                REWRITE_SYSTEM_PROMPT,
                rewrite_prompt,
                timeout_sec=config.timeout_sec,
                temperature=config.temperature,
                max_tokens=generator_provider.max_tokens or config.max_tokens,
                use_json_mode=config.use_json_mode,
            )
            raw_review_items.append(
                {
                    "chunk_index": chunk_index,
                    "attempt": attempt,
                    "bad_index": bad_index,
                    "type": "rewrite",
                    "generator_provider": generator_provider.prefix,
                    "generator_model": generator_provider.model,
                    "ok": rewrite_result.ok,
                    "content": rewrite_result.content,
                    "data": rewrite_result.data,
                    "error": rewrite_result.error,
                    "bad_item": bad_item,
                }
            )

            if not rewrite_result.ok or rewrite_result.data is None:
                errors.append(
                    f"chunk {chunk_index}: rewrite attempt {attempt}: "
                    f"{generator_provider.prefix} error: {rewrite_result.error}"
                )
                rejected_items.append(
                    {
                        "chunk_index": chunk_index,
                        "attempt": attempt,
                        "status": "rewrite_failed",
                        "question": bad_question,
                        "problems": bad_item.get("problems", []),
                        "rewrite_instruction": bad_item.get("rewrite_instruction", ""),
                        "rewrite_error": rewrite_result.error,
                    }
                )
                continue

            try:
                rewritten = collect_questions(rewrite_result.data, config.excel_columns)
            except Exception as exc:
                errors.append(f"chunk {chunk_index}: rewrite schema error: {type(exc).__name__}: {exc}")
                rejected_items.append(
                    {
                        "chunk_index": chunk_index,
                        "attempt": attempt,
                        "status": "rewrite_schema_error",
                        "question": bad_question,
                        "problems": [str(exc)],
                    }
                )
                continue

            if rewritten:
                next_pending.append(rewritten[0])
            else:
                rejected_items.append(
                    {
                        "chunk_index": chunk_index,
                        "attempt": attempt,
                        "status": "rewrite_empty",
                        "question": bad_question,
                        "problems": ["Rewriter returned no valid question"],
                    }
                )

        pending = next_pending

    return deduplicate_questions(accepted)


def generate_document_questions(
    *,
    config: AppConfig,
    secrets: dict[str, str],
    generator_provider: ProviderConfig,
    reviewer_provider: ProviderConfig | None,
    docx_path: Path,
    chunks: list[list[Any]],
    generator_template: str,
    reviewer_template: str,
    rewriter_template: str,
    timestamp: str,
) -> dict[str, Any]:
    generator_api_key = provider_api_key(secrets, generator_provider)
    if not generator_api_key:
        raise ValueError(f"Generator API key not found for provider '{generator_provider.prefix}' in secrets.yaml")

    reviewer_api_key = ""
    if config.review.enabled:
        if reviewer_provider is None:
            raise ValueError("Review is enabled, but reviewer provider is not configured")
        if reviewer_provider.name == generator_provider.name or reviewer_provider.prefix == generator_provider.prefix:
            raise ValueError("Generator and reviewer must be different providers")
        reviewer_api_key = provider_api_key(secrets, reviewer_provider)
        if not reviewer_api_key:
            raise ValueError(f"Reviewer API key not found for provider '{reviewer_provider.prefix}' in secrets.yaml")

    doc_base = safe_name(docx_path.stem)
    raw_items: list[dict[str, Any]] = []
    raw_review_items: list[dict[str, Any]] = []
    rejected_items: list[dict[str, Any]] = []
    all_questions: list[dict[str, str]] = []
    errors: list[str] = []

    forced_bad_total = effective_forced_bad_questions_count(config)
    generated_bad_total = 0
    normal_target = normal_questions_target(config)
    per_chunk_counts = distribute_questions(
        normal_target,
        len(chunks),
        config.max_questions_per_llm_call,
    )

    print(
        f"[GEN:{generator_provider.prefix}] {docx_path.name}: "
        f"chunks={len(chunks)}, model={generator_provider.model}, target_questions={config.question_count_per_document}, "
        f"normal_questions={normal_target}, forced_bad_questions={forced_bad_total}"
    )
    if config.review.enabled and reviewer_provider is not None:
        print(f"[REV:{reviewer_provider.prefix}] model={reviewer_provider.model}")

    for chunk_index, chunk in enumerate(chunks, start=1):
        requested_count = per_chunk_counts[chunk_index - 1]
        bad_needed_this_chunk = max(0, forced_bad_total - generated_bad_total)
        # Тестовые плохие вопросы добавляем в первый доступный чанк и считаем их частью общего лимита.
        if requested_count <= 0 and bad_needed_this_chunk <= 0:
            continue
        if len(all_questions) >= config.question_count_per_document:
            break

        document_text = blocks_to_text(chunk)
        questions: list[dict[str, str]] = []
        if requested_count > 0:
            user_prompt = build_user_prompt(
                generator_template,
                document_text=document_text,
                question_count=requested_count,
            )
            result = call_chat_completions(
                generator_provider,
                generator_api_key,
                SYSTEM_PROMPT,
                user_prompt,
                timeout_sec=config.timeout_sec,
                temperature=config.temperature,
                max_tokens=generator_provider.max_tokens or config.max_tokens,
                use_json_mode=config.use_json_mode,
            )

            raw_items.append(
                {
                    "provider": generator_provider.prefix,
                    "model": generator_provider.model,
                    "document": docx_path.name,
                    "chunk_index": chunk_index,
                    "chunks_total": len(chunks),
                    "requested_questions": requested_count,
                    "ok": result.ok,
                    "content": result.content,
                    "data": result.data,
                    "error": result.error,
                }
            )

            if not result.ok or result.data is None:
                errors.append(f"chunk {chunk_index}: generation error: {result.error}")
                continue

            try:
                questions = collect_questions(result.data, config.excel_columns)
                print(
                    f"[GEN:{generator_provider.prefix}] chunk {chunk_index}/{len(chunks)}: "
                    f"requested={requested_count}, generated={len(questions)}"
                )
            except Exception as exc:
                errors.append(f"chunk {chunk_index}: JSON schema error: {type(exc).__name__}: {exc}")
                continue

        if bad_needed_this_chunk > 0:
            injected = make_forced_bad_questions(bad_needed_this_chunk, config.excel_columns)
            generated_bad_total += len(injected)
            questions.extend(injected)
            raw_items.append(
                {
                    "provider": "internal_test",
                    "model": "forced_bad_questions",
                    "document": docx_path.name,
                    "chunk_index": chunk_index,
                    "chunks_total": len(chunks),
                    "requested_questions": len(injected),
                    "ok": True,
                    "content": "",
                    "data": {"questions": injected},
                    "error": "",
                }
            )
            print(f"[TEST] chunk {chunk_index}/{len(chunks)}: injected_bad_questions={len(injected)}")

        if config.review.enabled and reviewer_provider is not None and reviewer_api_key:
            accepted_questions = review_and_rewrite_questions(
                config=config,
                generator_provider=generator_provider,
                generator_api_key=generator_api_key,
                reviewer_provider=reviewer_provider,
                reviewer_api_key=reviewer_api_key,
                document_text=document_text,
                questions=questions,
                reviewer_template=reviewer_template,
                rewriter_template=rewriter_template,
                chunk_index=chunk_index,
                raw_review_items=raw_review_items,
                rejected_items=rejected_items,
                errors=errors,
            )
            all_questions.extend(accepted_questions)
            print(
                f"[REV:{reviewer_provider.prefix}] chunk {chunk_index}/{len(chunks)}: "
                f"accepted_after_review={len(accepted_questions)}"
            )
        else:
            all_questions.extend(questions)

        all_questions = deduplicate_questions(all_questions)[: config.question_count_per_document]

    all_questions = deduplicate_questions(all_questions)[: config.question_count_per_document]

    reviewer_prefix = reviewer_provider.prefix if config.review.enabled and reviewer_provider else "none"
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
            "normal_questions_target": normal_target,
            "forced_bad_questions_target": forced_bad_total,
            "items": raw_items,
            "questions": all_questions,
        },
    )

    if config.review.enabled and config.review.save_review_files:
        save_json(
            review_path,
            {
                "generator_provider": generator_provider.prefix,
                "reviewer_provider": reviewer_prefix,
                "items": raw_review_items,
            },
        )
        save_json(rejected_path, {"items": rejected_items})

    export_questions_xlsx(xlsx_path, all_questions, config.excel_columns)

    if errors:
        save_text(error_path, "\n\n".join(errors))

    print(f"[DONE] saved: {xlsx_path.name}, questions={len(all_questions)}")
    if config.review.enabled and config.review.save_review_files:
        print(f"[DONE] review saved: {review_path.name}, rejected={len(rejected_items)}")
    if errors:
        print(f"[WARN] warnings/errors saved: {error_path.name}")

    return {
        "document": docx_path.name,
        "generator": generator_provider.prefix,
        "reviewer": reviewer_prefix,
        "questions_count": len(all_questions),
        "xlsx": str(xlsx_path),
        "raw_json": str(raw_path),
        "review_json": str(review_path) if config.review.enabled else "",
        "rejected_json": str(rejected_path) if config.review.enabled else "",
        "errors": errors,
        "errors_file": str(error_path) if errors else "",
    }


def process_document(config: AppConfig, secrets: dict[str, str], docx_path: Path) -> dict[str, Any]:
    generator_template = read_text(config.prompts_dir / "question_generator.md")
    reviewer_template = read_text(config.prompts_dir / "question_reviewer.md")
    rewriter_template = read_text(config.prompts_dir / "question_rewriter.md")

    blocks = read_docx_blocks(docx_path)
    if not blocks:
        raise ValueError(f"No text blocks extracted from {docx_path.name}")

    chunks = build_generation_chunks(blocks, config)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    provider_index = build_provider_index(config.providers)
    generator_provider = get_provider_or_raise(provider_index, config.llm_roles.generator, "generator")
    reviewer_provider = None
    if config.review.enabled:
        reviewer_provider = get_provider_or_raise(provider_index, config.llm_roles.reviewer, "reviewer")

    return generate_document_questions(
        config=config,
        secrets=secrets,
        generator_provider=generator_provider,
        reviewer_provider=reviewer_provider,
        docx_path=docx_path,
        chunks=chunks,
        generator_template=generator_template,
        reviewer_template=reviewer_template,
        rewriter_template=rewriter_template,
        timestamp=timestamp,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate safety training questions from DOCX instructions")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
        config.output_dir.mkdir(parents=True, exist_ok=True)
        config.input_dir.mkdir(parents=True, exist_ok=True)

        if not config.secrets_file.exists():
            raise FileNotFoundError(
                f"secrets.yaml not found: {config.secrets_file}\n"
                "Copy secrets.example.yaml to secrets.yaml and paste API keys."
            )
        secrets = load_secrets(config.secrets_file)

        docx_files = sorted(config.input_dir.glob("*.docx"))
        if not docx_files:
            raise FileNotFoundError(f"No .docx files found in {config.input_dir}")

        results = []
        for docx_path in docx_files:
            results.append(process_document(config, secrets, docx_path))

        summary_path = config.output_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_summary.json"
        save_json(summary_path, {"results": results})
        print(f"[DONE] summary: {summary_path.name}")
        return 0
    except Exception as exc:
        print(f"[ERROR] {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
