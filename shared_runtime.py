"""Shared runtime helpers for logging, retry, metrics, deduplication, and local object storage."""

from __future__ import annotations

import functools
import logging
import random
import sqlite3
import json
import time
from pathlib import Path
from typing import Callable, TypeVar


T = TypeVar("T")


def setup_logging(name: str) -> logging.Logger:
    """Return a consistently configured logger for console apps."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


def retry_with_backoff(
    attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    exceptions: tuple[type[BaseException], ...] = (Exception,),
):
    """Retry decorator with exponential backoff and jitter."""

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_error: BaseException | None = None
            for attempt in range(1, attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as error:
                    last_error = error
                    if attempt == attempts:
                        break
                    delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
                    delay += random.uniform(0, 0.25 * delay)
                    time.sleep(delay)
            assert last_error is not None
            raise last_error

        return wrapper

    return decorator


class DedupStore:
    """Very small SQLite-backed seen-key store for idempotent ingestion."""

    def __init__(self, db_path: str = ".rag_state/processed_items.sqlite"):
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path, timeout=30, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA busy_timeout=30000")
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS seen_items (
                item_key TEXT PRIMARY KEY,
                seen_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self.conn.commit()

    def has(self, item_key: str) -> bool:
        cur = self.conn.execute("SELECT 1 FROM seen_items WHERE item_key = ? LIMIT 1", (item_key,))
        return cur.fetchone() is not None

    def add(self, item_key: str) -> None:
        self.conn.execute("INSERT OR IGNORE INTO seen_items(item_key) VALUES (?)", (item_key,))
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()


class MetricsStore:
    """SQLite-backed counters for lightweight telemetry."""

    def __init__(self, db_path: str = ".rag_state/metrics.sqlite"):
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path, timeout=30, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA busy_timeout=30000")
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS metrics (
                metric_name TEXT PRIMARY KEY,
                metric_value REAL NOT NULL DEFAULT 0,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self.conn.commit()

    def increment(self, metric_name: str, amount: float = 1.0) -> float:
        self.conn.execute(
            """
            INSERT INTO metrics(metric_name, metric_value)
            VALUES(?, ?)
            ON CONFLICT(metric_name) DO UPDATE SET
                metric_value = metric_value + excluded.metric_value,
                updated_at = CURRENT_TIMESTAMP
            """,
            (metric_name, amount),
        )
        self.conn.commit()
        return self.get(metric_name)

    def set(self, metric_name: str, value: float) -> None:
        self.conn.execute(
            """
            INSERT INTO metrics(metric_name, metric_value)
            VALUES(?, ?)
            ON CONFLICT(metric_name) DO UPDATE SET
                metric_value = excluded.metric_value,
                updated_at = CURRENT_TIMESTAMP
            """,
            (metric_name, value),
        )
        self.conn.commit()

    def get(self, metric_name: str, default: float = 0.0) -> float:
        cur = self.conn.execute("SELECT metric_value FROM metrics WHERE metric_name = ?", (metric_name,))
        row = cur.fetchone()
        return float(row[0]) if row else default

    def all(self) -> dict[str, float]:
        cur = self.conn.execute("SELECT metric_name, metric_value FROM metrics")
        return {name: float(value) for name, value in cur.fetchall()}

    def close(self) -> None:
        self.conn.close()


class LocalObjectStore:
    """Tiny filesystem-backed object store for downloaded assets."""

    def __init__(self, root_dir: str = ".rag_state/object_store"):
        self.root = Path(root_dir)
        self.root.mkdir(parents=True, exist_ok=True)

    def save_bytes(self, content: bytes, key: str, suffix: str = "") -> str:
        safe_key = key.replace("/", "_").replace(":", "_")
        path = self.root / f"{safe_key}{suffix}"
        path.write_bytes(content)
        return str(path)


class DeadLetterStore:
    """Append-only JSONL dead-letter sink for malformed records."""

    def __init__(self, file_path: str = ".rag_state/dead_letters.jsonl"):
        self.path = Path(file_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, payload: dict, reason: str, stage: str) -> None:
        entry = {
            "reason": reason,
            "stage": stage,
            "ts": time.time(),
            "payload": payload,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")