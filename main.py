import argparse
import asyncio
from datetime import datetime, timezone
import sys
from pathlib import Path

# Apply library patches before importing core logic
sys.path.append(str(Path(__file__).parent / "scripts"))
try:
    from apply_patches import apply_patches
    apply_patches()
except ImportError:
    pass

from core import (
    DecisionEntity,
    ProjectEntity,
    TeamEntity,
    TechnicalConceptEntity,
    L3Summary, # Добавлено
    get_graphiti_client,
)
from core.graphiti_client import get_write_semaphore
from queries.context_builder import build_agent_context
from queries.quality_check import check_graph_quality
from queries.search_strategies import test_search_strategies
from layers.l1_consolidation import get_l1_context
from layers.l2_semantic import get_l2_semantic_context
from layers.l3_fractal import get_l3_fractal_context
from visualization.visualization_export import export_to_file
from benchmarks.benchmark import run_benchmark
from core.migrations import apply_migrations
from scripts.consolidate import run_consolidate as cmd_consolidate # Импортируем скрипт консолидации
from queries.dedupe_entities import main as dedupe_entities_main


async def ensure_graphiti():
    client = get_graphiti_client()
    graphiti = await client.ensure_ready()
    return graphiti


async def cmd_setup(args):
    graphiti = await ensure_graphiti()
    mig = await apply_migrations(graphiti)
    print("✅ Graphiti/Neo4j готов, индексы созданы")
    if mig["total"] > 0:
        print(f"✅ Migrations: applied={mig['applied']} skipped={mig['skipped']} total={mig['total']}")


async def cmd_seed(args):
    graphiti = await ensure_graphiti()
    episodes = [
        dict(
            name="Project Overview",
            body="""
            Sergey and Natasha are working on a Fractal Memory project.

            The project has three main components:
            1. Graph Engine - built with Neo4j for knowledge representation
            2. LLM Integration - using GPT-4 for entity extraction and reasoning
            3. Temporal Processing - maintaining bi-temporal data (valid_from, valid_to)

            The project status is in Development phase.
            Sergey is the primary developer.
            Priority is High - this is a core research initiative.

            Key concepts involved:
            - Fractal Architecture: a hierarchical representation system
            - Knowledge Graph: semantic network of entities and relationships
            - Temporal Logic: maintaining contradictions over time

            These concepts are at Advanced abstraction level (3-4).
            """,
            source="project_documentation",
        ),
        dict(
            name="Strategic Decision - Vanilla First",
            body="""
            Decision made on 2025-12-10:

            "We decided to simplify the Fractal Memory implementation by starting with 
            vanilla Graphiti instead of building custom Redis buffer layer."

            Made by: Natasha
            Rationale: Reduce complexity, avoid Integration Hell, focus on core value.
            Dependencies: This affects L0 optimization, L1 consolidation logic.
            Status: Active - this is our current strategy.
            """,
            source="decision_log",
        ),
        dict(
            name="Team Structure",
            body="""
            The development team consists of:
            - Sergey: Senior Developer, specializing in AI/ML and Python
            - Natasha: Technical Lead and Business Advisor, strategic guidance

            Team Name: Fractal Memory Core Team
            Focus: Building production-grade memory system for AI agents
            Communication: Primarily Telegram for async discussions
            """,
            source="team_documentation",
        ),
    ]

    write_semaphore = get_write_semaphore()
    for ep in episodes:
        async with write_semaphore:
            await graphiti.add_episode(
                name=ep["name"],
                episode_body=ep["body"],
                source_description=ep["source"],
                reference_time=datetime.now(timezone.utc),
            )
        print(f"✅ Added episode: {ep['name']}")

    # Link user to person entity
    from knowledge.ingest import link_user_to_person_entity
    await link_user_to_person_entity(graphiti, "sergey", "Sergey")

    if args.with_search:
        await cmd_search_demo(args)

    print("✨ Demo data loaded")


async def cmd_clear(args):
    graphiti = await ensure_graphiti()
    # Полная очистка графа
    await graphiti.driver.execute_query("MATCH (n) DETACH DELETE n")
    await graphiti.build_indices_and_constraints()
    await apply_migrations(graphiti)
    print("🧹 Graph cleared and indices recreated")


async def cmd_migrate(args):
    graphiti = await ensure_graphiti()
    mig = await apply_migrations(graphiti)
    print(f"✅ Migrations: applied={mig['applied']} skipped={mig['skipped']} total={mig['total']}")


async def cmd_quality(args):
    await check_graph_quality()


async def cmd_search_demo(args):
    await test_search_strategies()


async def cmd_context(args):
    graphiti = await ensure_graphiti()
    context = await build_agent_context(
        graphiti,
        entity_name=args.entity,
        context_size=args.size,
    )
    if context:
        print(context)
    else:
        print("⚠️  Сущность не найдена")


async def cmd_l1(args):
    graphiti = await ensure_graphiti()
    summary = await get_l1_context(
        graphiti,
        user_context=args.query,
        hours_back=args.hours,
    )
    print(summary)


async def cmd_l2(args):
    graphiti = await ensure_graphiti()
    summary = await get_l2_semantic_context(graphiti, args.entity)
    if summary:
        print(summary)
    else:
        print("⚠️  Сущность не найдена")


async def cmd_l3(args):
    graphiti = await ensure_graphiti()
    summary = await get_l3_fractal_context(graphiti, args.entity)
    if summary:
        print(summary)
    else:
        print("⚠️  Сущность не найдена")


async def cmd_viz_export(args):
    graphiti = await ensure_graphiti()
    await export_to_file(graphiti, filename=args.output)


async def cmd_benchmark(args):
    await run_benchmark()


def build_parser():
    parser = argparse.ArgumentParser(description="Fractal Memory / Graphiti CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("setup", help="Создать индексы/констрейнты Graphiti").set_defaults(
        func=cmd_setup
    )

    seed_p = subparsers.add_parser("seed", help="Создать демо-эпизоды и сущности")
    seed_p.add_argument("--with-search", action="store_true", help="Запустить поисковую демонстрацию после загрузки")
    seed_p.set_defaults(func=cmd_seed)

    subparsers.add_parser("quality", help="Отчёт по качеству графа").set_defaults(func=cmd_quality)
    subparsers.add_parser("clear", help="Очистить граф (reset) и пересоздать индексы").set_defaults(func=cmd_clear)
    subparsers.add_parser("migrate", help="Применить миграции из папки migrations/").set_defaults(func=cmd_migrate)
    subparsers.add_parser("search-demo", help="Демонстрация стратегий поиска").set_defaults(func=cmd_search_demo)

    ctx_p = subparsers.add_parser("context", help="Построить контекст для сущности")
    ctx_p.add_argument("entity", help="Имя сущности")
    ctx_p.add_argument("--size", choices=["minimal", "medium", "full"], default="full")
    ctx_p.set_defaults(func=cmd_context)

    l1_p = subparsers.add_parser("l1", help="L1 recent context summary")
    l1_p.add_argument("--query", default="Fractal Memory", help="Поисковая фраза для недавних эпизодов")
    l1_p.add_argument("--hours", type=int, default=24, help="Часов назад для выборки")
    l1_p.set_defaults(func=cmd_l1)

    l2_p = subparsers.add_parser("l2", help="L2 semantic patterns")
    l2_p.add_argument("entity", help="Имя сущности")
    l2_p.set_defaults(func=cmd_l2)

    l3_p = subparsers.add_parser("l3", help="L3 fractal abstraction")
    l3_p.add_argument("entity", help="Имя сущности")
    l3_p.set_defaults(func=cmd_l3)

    viz_p = subparsers.add_parser("viz-export", help="Экспорт графа в JSON для D3")
    viz_p.add_argument(
        "--output",
        default="visualization/graph_data.json",
        help="Путь к JSON (по умолчанию visualization/graph_data.json)",
    )
    viz_p.set_defaults(func=cmd_viz_export)

    subparsers.add_parser("benchmark", help="Перфоманс add_episode/search").set_defaults(func=cmd_benchmark)
    
    # Добавляем команду для консолидации L3
    consolidate_p = subparsers.add_parser("consolidate", help="Запустить консолидацию L3 памяти")
    consolidate_p.add_argument("--hours", type=int, default=24*7, help="Количество часов для консолидации")
    consolidate_p.set_defaults(func=lambda args: cmd_consolidate(ensure_graphiti(), hours_back=args.hours))

    # Добавляем команду для дедупликации Entity узлов
    dedupe_p = subparsers.add_parser("dedupe-entities", help="Дедуплицировать Entity узлы по имени")
    dedupe_p.add_argument("--dry-run", action="store_true", help="Анализ без применения изменений")
    dedupe_p.set_defaults(func=lambda args: dedupe_entities_main(dry_run=args.dry_run))


    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    asyncio.run(args.func(args))


if __name__ == "__main__":
    main()

