"""Nightly NVD cache refresh — run from cron or CI.

Usage:
    uv run examples/scripts/refresh_cache.py

Exit codes:
    0 — success
    1 — error
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

CACHE_PATH = Path("data/library.parquet")
MAX_AGE = "1d"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    from vulnflow.helpers import CacheBuilder

    logger.info("Ensuring cache at %s (max_age=%s)", CACHE_PATH, MAX_AGE)

    ds = CacheBuilder(quiet=True).ensure(CACHE_PATH, max_age=MAX_AGE)

    prov = ds.provenance
    logger.info(
        "Done: %d records, date range %s .. %s",
        len(ds),
        prov.get("date_min", "?"),
        prov.get("date_max", "?"),
    )


if __name__ == "__main__":
    try:
        main()
        sys.exit(0)
    except Exception:
        logger.exception("Cache refresh failed")
        sys.exit(1)
