# ИИ-агент генерации обучающих вопросов по охране труда

Локальный Python-инструмент для Windows. Принимает Word-инструкцию `.docx`, генерирует обучающие вопросы, проверяет их второй LLM, отправляет плохие вопросы на переделку и сохраняет итоговый Excel.

## Новое в этой версии: web-интерфейс

Запуск локального интерфейса:

```bat
run_ui.bat
```

После запуска откроется браузер:

```text
http://127.0.0.1:8000
```

Интерфейс содержит:

- поле количества вопросов, по умолчанию `10`;
- выбор LLM-генератора и LLM-проверяющего;
- выбор моделей ChatGPT/OpenAI и DeepSeek;
- сохранение изменяемых параметров в `config.yaml`;
- загрузку Word `.docx`;
- preview извлечённого текста инструкции;
- live-зону «Работа программы»;
- таблицу вопросов, которая обновляется во время работы;
- подсветку статусов:
  - хороший вопрос — зелёный;
  - плохой вопрос — розовый;
  - вопрос на переделке — оранжевый;
  - исправленный вопрос — насыщенный зелёный;
- ссылку на скачивание итогового Excel и служебных JSON-файлов.

## Архитектура LLM

По умолчанию:

```yaml
llm_roles:
  generator: chatgpt
  reviewer: deepseek
```

То есть:

```text
ChatGPT генерирует вопросы
DeepSeek проверяет вопросы
ChatGPT переделывает плохие вопросы
DeepSeek повторно проверяет исправленные вопросы
```

Можно поменять роли в интерфейсе или напрямую в `config.yaml`.

## API-ключи

Ключи доступа хранятся только в отдельном файле `secrets.yaml`.

Скопируйте:

```text
secrets.example.yaml -> secrets.yaml
```

И заполните:

```yaml
chatgpt:
  api_key: "sk-..."

deepseek:
  api_key: "sk-..."
```

`secrets.yaml` не должен попадать в git.

## Выходной Excel

Итоговый Excel содержит колонки:

```text
Вопрос
Правильный ответ
Неправильный ответ 1
Неправильный ответ 2
Раздел
Пункт
Пояснение правильного ответа
Тема
```

## Служебные файлы

После генерации в `output/` сохраняются:

```text
*_questions.xlsx      итоговый Excel
*_raw.json           сырые ответы генератора
*_review.json        результаты проверки и переделки
*_rejected.json      вопросы, которые не удалось принять
*_errors.txt         ошибки, если были
```

## Старый CLI-запуск

CLI-режим сохранён:

```bat
run.bat
```

Он берёт `.docx` из папки `input/` и сохраняет результат в `output/`.

## Основные файлы

```text
config.yaml                         настройки
secrets.yaml                        API-ключи, создаётся пользователем
prompts/question_generator.md       редактируемый промпт генерации
prompts/question_reviewer.md        редактируемый промпт проверки
prompts/question_rewriter.md        редактируемый промпт переделки
src/iot_question_agent/ui_app.py    web-интерфейс и live-обработка
```

## Если модели не отображаются

В интерфейсе есть кнопка «Обновить модели API». Она пытается получить модели через `/models` у соответствующего провайдера. Если ключа нет или API недоступен, интерфейс использует список моделей из `config.yaml`.

## Тест reviewer искусственно плохими вопросами

Чтобы проверить, действительно ли LLM-проверяющий бракует плохие вопросы, включите режим в `config.yaml`:

```yaml
review_test:
  enabled: true
  forced_bad_questions_count: 1
```

Плохие тестовые вопросы входят в общее количество вопросов. Например:

```yaml
question_count_per_document: 10
review_test:
  enabled: true
  forced_bad_questions_count: 1
```

Это означает: агент попросит генератор сделать 9 обычных вопросов и добавит 1 искусственно плохой вопрос для проверки reviewer. В интерфейсе такой вопрос должен сначала стать розовым/оранжевым и получить причину браковки. Если `review.max_rewrite_attempts > 0`, вопрос затем отправится генератору на переделку и может быть заменён хорошей версией.

Для обычной работы выключите режим:

```yaml
review_test:
  enabled: false
  forced_bad_questions_count: 0
```


## fix9: пустой ответ LLM из-за thinking/reasoning

Если в ошибке видно `Raw assistant content is empty`, а в usage все выходные токены ушли в `reasoning_tokens`, это означает, что модель сгенерировала скрытое рассуждение, но не выдала финальный JSON.

В fix9 сделаны две защиты:

1. Для ChatGPT/OpenAI по умолчанию используется `endpoint: /responses`, потому что для GPT-5.x OpenAI рекомендует Responses API. Клиент извлекает `output_text`/`output[].content[].text` и использует `max_output_tokens`.
2. Для DeepSeek V4 по умолчанию добавлен параметр `thinking: disabled`. Это отключает thinking mode, чтобы DeepSeek возвращал обычный `content` с JSON, а не только `reasoning_content`.

Ключевой фрагмент `config.yaml`:

```yaml
providers:
  chatgpt:
    base_url: https://api.openai.com/v1
    endpoint: /responses
    model: gpt-5.5
    reasoning_effort: minimal
    max_tokens: 16000

  deepseek:
    base_url: https://api.deepseek.com
    endpoint: /chat/completions
    model: deepseek-v4-flash
    thinking: disabled
    max_tokens: 8000
```
