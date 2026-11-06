"""Alert pipeline — detect critical KEV CVEs published in the last 24 hours.

Usage:
    uv run examples/scripts/critical_alert.py

Exit codes:
    0 — no critical KEV CVEs found
    1 — error
    2 — critical KEV CVEs detected (output written)
"""

from __future__ import annotations

import logging
import sys
from datetime import UTC, datetime, timedelta

OUTPUT_PATH = "data/critical_kev_alerts.json"
LOOKBACK_HOURS = 24

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> int:
    from vulnflow.extractors.nvd_api import NVDAPIExtractor
    from vulnflow.transforms.enrichers.epss import EPSSEnricher
    from vulnflow.transforms.enrichers.kev import KEVEnricher
    from vulnflow.transforms.flatten.cvss import CVSSFlattenTransform
    from vulnflow.transforms.flatten.epss import EPSSFlattenTransform
    from vulnflow.transforms.flatten.kev import KEVFlattenTransform

    now = datetime.now(UTC)
    start = now - timedelta(hours=LOOKBACK_HOURS)
    pub_start = start.isoformat()
    pub_end = now.isoformat()

    logger.info("Querying NVD API: CRITICAL CVEs from %s to %s", pub_start, pub_end)

    extractor = NVDAPIExtractor(
        cvss_v3_severity="CRITICAL",
        pub_start_date=pub_start,
        pub_end_date=pub_end,
        no_rejected=True,
    )
    ds = extractor.extract()
    logger.info("Extracted %d critical CVEs", len(ds))

    if len(ds) == 0:
        logger.info("No critical CVEs in the last %d hours", LOOKBACK_HOURS)
        return 0

    ds.transform([
        KEVEnricher(),
        KEVFlattenTransform(),
        CVSSFlattenTransform(),
        EPSSEnricher(),
        EPSSFlattenTransform(),
    ], quiet=True)

    # Filter to KEV-listed CVEs
    import numpy as np

    is_kev = ds.get_column("is_kev")
    mask = np.array([v is True or v == True for v in is_kev], dtype=bool)  # noqa: E712
    kev_ds = ds.filter(mask)

    if len(kev_ds) > 0:
        kev_ds.to_json(OUTPUT_PATH)
        logger.info(
            "ALERT: %d critical KEV CVEs found — written to %s",
            len(kev_ds),
            OUTPUT_PATH,
        )
        return 2

    logger.info("No critical KEV CVEs in the last %d hours", LOOKBACK_HOURS)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        logger.exception("Alert pipeline failed")
        sys.exit(1)
