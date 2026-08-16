"""Shared runtime helpers for logging, retry, and simple deduplication."""

from __future__ import annotations

import functools
import logging
import random
import sqlite3
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
        self.conn = sqlite3.connect(self.path)
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