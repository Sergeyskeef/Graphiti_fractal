# Fractal Memory v2 — Graphiti-first (единый entrypoint)

Проект использует Graphiti как единственную точку входа к Neo4j и собирает три слоя памяти (L1–L3) с визуализацией и бенчмарками. Все операции запускаются через один CLI `main.py`.

## Структура
- `core/` — обёртки Graphiti, кастомные сущности.
- `layers/` — L1/L2/L3 построение контекста.
- `queries/` — стратегии поиска, quality check, context builder.
- `visualization/` — экспорт графа и HTML c D3.js.
- `benchmarks/` — скрипты производительности.
- `tests/` — smoke-тесты импорта/структуры.

## Требования
- Python 3.10+
- Docker Neo4j с включёнными индексами
- Переменные окружения: `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`, `OPENAI_API_KEY`, опционально `NEO4J_DATABASE` (см. `.env.example`)

## Установка
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Основные команды
```bash
# Инициализация индексов/констрейнтов
python main.py setup

# Загрузить демо-эпизоды и сущности
python main.py seed

# Отчёт по качеству графа
python main.py quality

# Демонстрация стратегий поиска
python main.py search-demo

# Контекст для сущности
python main.py context "Fractal Memory" --size full

# Слои памяти
python main.py l1 --query "Fractal Memory" --hours 24
python main.py l2 "Sergey"
python main.py l3 "Fractal Memory"

# Визуализация и бенчмарки
python main.py viz-export --output visualization/graph_data.json
python main.py benchmark

# Очистка графа
python main.py clear

# Web API + статический чат
python -m uvicorn app:app --host 0.0.0.0 --port 8000
# UI: http://localhost:8000/static/index.html
```

## MCP (Cursor) — локальный запуск на Windows (stdio)
Важно: **stdio MCP запускается на том компьютере, где запущен Cursor**. Поэтому если Cursor у тебя на Windows‑ПК, то и MCP‑процесс будет стартовать на Windows (даже если раньше ты открывал проект по SSH).

### Как подключить MCP в Cursor
- **1) Подготовь окружение в корне проекта**:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

- **2) Укажи Cursor запуск через `run_mcp_server.cmd`**:
  - Файл‑шаблон: `mcp.json.example`
  - В Cursor → **Tools & MCP** → Add MCP Server → stdio:
    - `command`: `cmd.exe`
    - `args`: `["/c", "C:\\PATH\\TO\\Graphiti_fractal\\run_mcp_server.cmd"]`

### Нужно ли держать PowerShell открытым?
- **Нет**, в обычной работе **не нужно**. MCP‑процесс поднимает и контролирует **сам Cursor**.
- Вручную запускать `run_mcp_server.cmd` имеет смысл только для быстрой диагностики “стартует ли сервер”.

HTML для D3.js: `visualization/visualization.html` (использует `visualization/graph_data.json`).

## Docker (Neo4j + app)
```bash
# собрать контейнеры
docker compose build

# поднять Neo4j и app-контейнер (app ждёт, можно exec внутрь)
docker compose up -d

# войти в app-контейнер и выполнить команды
docker compose exec app bash
python main.py setup
python main.py seed
python main.py quality
# запустить веб-API
python -m uvicorn app:app --host 0.0.0.0 --port 8000
# открыть http://localhost:8000/static/index.html
```

Порты:
- Neo4j Browser: http://localhost:7474 (bolt 7687)
- Web chat: http://localhost:8000/static/index.html

Данные Neo4j сохраняются в `./neo4j/data`.

## Maintenance Scripts
Скрипты для обслуживания графа и исправления данных:

*   `scripts/audit_authorship.py` — Проверка целостности связей авторства (`[:AUTHORED]`) и идентификации (`[:IS]`).
*   `scripts/backfill_authored_edges.py` — Автоматическое создание связей авторства для эпизодов, потерявших их (например, при сбоях).
*   `scripts/seed_identity.py` — Создание семантической сущности пользователя и связывание её с User-аккаунтом.

## Связность графа (SAME_AS)
Для объединения слоёв (Personal/Project/Knowledge) используются связи `[:SAME_AS]`, которые создаются автоматически при импорте.

### Полезные Cypher-запросы для мониторинга:

**А. Общая статистика связности:**
```cypher
MATCH (e:Entity)
WITH count(e) as entities
MATCH ()-[r:SAME_AS]->()
WITH entities, count(r) as bridges
MATCH ()-[r2:RELATES_TO]->()
RETURN entities, bridges, count(r2) as semantic_relations
```

**Б. Топ "мостовых" сущностей:**
```cypher
MATCH (e:Entity)-[r:SAME_AS]-(other)
RETURN e.name, e.group_id, count(r) as connections, collect(other.group_id) as linked_groups
ORDER BY connections DESC
LIMIT 10
```

**В. Проверка cross-layer retrieval (1-hop):**
```cypher
MATCH (e:Entity {name: "Graphiti"}) 
MATCH (e)-[:SAME_AS]-(neighbor)
OPTIONAL MATCH (neighbor)-[r:RELATES_TO]->(target)
RETURN e.group_id, neighbor.group_id, type(r), target.name
LIMIT 20
```

Для полной диагностики используйте `python scripts/graph_health.py`.
