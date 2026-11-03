"""CI/CD ETL pipeline — build enriched NVD library via VulnPipeline.

Usage:
    uv run examples/scripts/etl_pipeline.py

Exit codes:
    0 — success
    1 — error (check logs)
"""

from __future__ import annotations

import logging
import sys

YEARS = [2024, 2025, 2026]
OUTPUT_PATH = "data/ci_library.parquet"
STRICT = True

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> int:
    from vulnflow.extractors.nvd import NVDExtractor
    from vulnflow.helpers import CacheBuilder
    from vulnflow.loaders.files import ParquetLoader
    from vulnflow.pipeline import VulnPipeline
    from vulnflow.transforms.enrichers.cwe import CWEEnricher
    from vulnflow.transforms.enrichers.epss import EPSSEnricher
    from vulnflow.transforms.enrichers.kev import KEVEnricher
    from vulnflow.transforms.flatten.cpe import CPEFlattenTransform
    from vulnflow.transforms.flatten.cvss import CVSSFlattenTransform
    from vulnflow.transforms.flatten.cwe import CWEFlattenTransform
    from vulnflow.transforms.flatten.epss import EPSSFlattenTransform
    from vulnflow.transforms.flatten.kev import KEVFlattenTransform

    extractors = [NVDExtractor(CacheBuilder.feed_url(y)) for y in YEARS]

    transformers = [
        CVSSFlattenTransform(),
        EPSSEnricher(),
        EPSSFlattenTransform(),
        KEVEnricher(),
        KEVFlattenTransform(),
        CWEEnricher(),
        CWEFlattenTransform(),
        CPEFlattenTransform(),
    ]

    loader = ParquetLoader(OUTPUT_PATH)

    pipeline = VulnPipeline(
        extractors=extractors,
        transformers=transformers,
        loaders=[loader],
    )

    logger.info(
        "Running pipeline: %d extractors, %d transforms, output=%s",
        len(extractors),
        len(transformers),
        OUTPUT_PATH,
    )

    result = pipeline.run(strict=STRICT, quiet=True)

    logger.info("Duration           : %.1fs", result.duration)
    logger.info("Successful stages  : %s", result.successful_stages)
    logger.info("Records            : %d", len(result.dataset) if result.dataset else 0)

    if result.errors:
        for err in result.errors:
            logger.error("Pipeline error: %s", err)
        return 1

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        logger.exception("ETL pipeline failed")
        sys.exit(1)
