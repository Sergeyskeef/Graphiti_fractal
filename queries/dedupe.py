#!/usr/bin/env python3
"""
Утилита дедупликации эпизодов.
- Проставляет fingerprint для всех Episodic (sha256 нормализованного summary/content).
- Помечает дубликаты: оставляет первый, остальным ставит duplicate_of и deleted=true.
"""

import asyncio
import hashlib
import re
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Optional

from core.graphiti_client import get_graphiti_client


def normalize(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text.strip())
    return cleaned.lower()


def fingerprint(text: str) -> str:
    return hashlib.sha256(normalize(text).encode("utf-8")).hexdigest()


async def fetch_episodes(driver) -> list[dict]:
    res = await driver.execute_query(
        """
        MATCH (e:Episodic)
        RETURN e.uuid AS uuid, coalesce(e.summary, e.content, '') AS text
        """
    )
    out = []
    for rec in res.records:
        out.append({"uuid": rec["uuid"], "text": rec["text"] or ""})
    return out


async def set_fingerprint(driver, uuid: str, fp: str):
    await driver.execute_query(
        "MATCH (e:Episodic {uuid:$uuid}) SET e.fingerprint=$fp", uuid=uuid, fp=fp
    )


async def mark_duplicate(driver, uuid: str, master_uuid: str):
    await driver.execute_query(
        """
        MATCH (e:Episodic {uuid:$uuid})
        SET e.deleted=true, e.duplicate_of=$master, e.deleted_at=$deleted_at
        """,
        uuid=uuid,
        master=master_uuid,
        deleted_at=datetime.now(timezone.utc).isoformat(),
    )


async def purge_deleted(driver, days: int = 3) -> int:
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    res = await driver.execute_query(
        """
        MATCH (e:Episodic)
        WHERE e.deleted = true
          AND exists(e.deleted_at)
          AND datetime(e.deleted_at) < datetime($cutoff)
        DETACH DELETE e
        RETURN count(*) AS purged
        """,
        cutoff=cutoff.isoformat(),
    )
    return res.records[0]["purged"] if res.records else 0


async def main():
    graphiti = await get_graphiti_client().ensure_ready()
    driver = graphiti.driver

    episodes = await fetch_episodes(driver)
    groups = defaultdict(list)
    for ep in episodes:
        fp = fingerprint(ep["text"])
        ep["fp"] = fp
        groups[fp].append(ep)

    updated_fp = 0
    duplicates = 0

    for fp, items in groups.items():
        master = items[0]["uuid"]
        for ep in items:
            await set_fingerprint(driver, ep["uuid"], fp)
            updated_fp += 1
        if len(items) > 1:
            for dup in items[1:]:
                await mark_duplicate(driver, dup["uuid"], master)
                duplicates += 1

    purged = await purge_deleted(driver, days=3)

    print(f"Fingerprints set: {updated_fp}")
    print(f"Duplicates marked (soft): {duplicates}")
    print(f"Purged deleted older than 3 days: {purged}")


if __name__ == "__main__":
    asyncio.run(main())

