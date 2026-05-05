from __future__ import annotations

from typing import Any


REQUIRED_COLUMNS = [
    "Вопрос",
    "Правильный ответ",
    "Неправильный ответ 1",
    "Неправильный ответ 2",
    "Раздел",
    "Пункт",
    "Пояснение правильного ответа",
    "Тема",
]


def normalize_question(item: dict[str, Any], columns: list[str] | None = None) -> dict[str, str]:
    cols = columns or REQUIRED_COLUMNS
    result: dict[str, str] = {}
    for col in cols:
        value = item.get(col, "")
        if value is None:
            value = ""
        result[col] = " ".join(str(value).split()).strip()
    return result


def collect_questions(data: dict[str, Any], columns: list[str] | None = None) -> list[dict[str, str]]:
    raw_questions = data.get("questions", [])
    if not isinstance(raw_questions, list):
        raise ValueError("JSON field 'questions' must be a list")

    questions: list[dict[str, str]] = []
    for raw in raw_questions:
        if not isinstance(raw, dict):
            continue
        q = normalize_question(raw, columns)
        if q.get("Вопрос") and q.get("Правильный ответ"):
            questions.append(q)
    return deduplicate_questions(questions)


def deduplicate_questions(questions: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[str] = set()
    result: list[dict[str, str]] = []
    for q in questions:
        key = q.get("Вопрос", "").lower().strip()
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(q)
    return result
