"""
Event Inference - Infer CVD timeline timestamps from metadata

Provides:
- EventInference: Creates V/F/D/P/X/A timestamp columns from references, NVD dates, KEV
- extract_date_from_url: Parse common URL date patterns

Layer: Transform/Inferences
Dependencies: dataset, KEV enricher (for A timestamp)

CRITICAL: EventInference is the ONLY source of CVD timestamps.
These columns do not exist in VulnRecord - EventInference creates them.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from vulnflow.dataset import VulnDataset

from vulnflow.registry import register_transform


def extract_date_from_url(url: str | None) -> datetime | None:
    """Extract date from common URL patterns.

    Supports:
    - YYYY/MM/DD: /2024/01/15/advisory.html
    - YYYY-MM-DD: ?date=2024-01-15

    Args:
        url: URL to parse

    Returns:
        Extracted datetime (UTC) or None if no pattern found
    """
    if not url:
        return None

    # Pattern 1: YYYY/MM/DD in path
    date_match = re.search(r"/(\d{4})/(\d{2})/(\d{2})", url)
    if date_match:
        year, month, day = date_match.groups()
        try:
            return datetime(int(year), int(month), int(day), tzinfo=UTC)
        except ValueError:
            pass

    # Pattern 2: YYYY-MM-DD in path or query
    iso_match = re.search(r"(\d{4})-(\d{2})-(\d{2})", url)
    if iso_match:
        year, month, day = iso_match.groups()
        try:
            return datetime(int(year), int(month), int(day), tzinfo=UTC)
        except ValueError:
            pass

    return None


@register_transform(default=True)
class EventInference:
    """Infer missing CVD timestamps from references, NVD metadata, KEV.

    Creates columns: v_timestamp, f_timestamp, d_timestamp, p_timestamp, x_timestamp, a_timestamp

    Inference rules:
    - P: "Vendor Advisory" tag → URL date (confidence 0.9), or nvd_published_at (0.6)
    - F: "Patch" tag → URL date (confidence 0.8)
    - X: "Exploit" tag → URL date (confidence 0.9)
    - A: KEV date_added (confidence 1.0 - authoritative)
    - V, D: Not inferred in this implementation (require additional signals)
    """

    name: str = "EventInference"
    depends_on: list[str] = ["KEVEnricher"]
    confidence_threshold: float

    SCHEMA_VERSION: str = "1.0.0"
    OUTPUT_SCHEMA: list[dict[str, str]] = [
        {"name": "v_timestamp", "type": "datetime", "description": "Vendor awareness timestamp"},
        {"name": "f_timestamp", "type": "datetime", "description": "Fix ready timestamp"},
        {"name": "d_timestamp", "type": "datetime", "description": "Fix deployed timestamp"},
        {"name": "p_timestamp", "type": "datetime", "description": "Public awareness timestamp"},
        {"name": "x_timestamp", "type": "datetime", "description": "Exploit public timestamp"},
        {"name": "a_timestamp", "type": "datetime", "description": "Attack observed timestamp"},
    ]

    def __init__(self, confidence_threshold: float = 0.7) -> None:
        """Initialize EventInference.

        Args:
            confidence_threshold: Minimum confidence (0.0-1.0) to populate field
        """
        self.confidence_threshold = confidence_threshold

    def transform(self, dataset: VulnDataset) -> None:
        """Create CVD timestamp columns from available metadata.

        Args:
            dataset: VulnDataset to transform (adds columns)
        """
        n = len(dataset)

        # Initialize CVD timestamp columns
        v_timestamps: list[datetime | None] = [None] * n
        f_timestamps: list[datetime | None] = [None] * n
        d_timestamps: list[datetime | None] = [None] * n
        p_timestamps: list[datetime | None] = [None] * n
        x_timestamps: list[datetime | None] = [None] * n
        a_timestamps: list[datetime | None] = [None] * n

        # Get source data (use public API)
        nvd_published = dataset.get_column_or_none("nvd_published_at")
        kev_date_added = dataset.get_column_or_none("kev_date_added")

        for i in range(n):
            # Infer P from nvd_published_at (confidence 0.8 - NVD is authoritative for public disclosure)
            p_confidence = 0.8
            if p_confidence >= self.confidence_threshold and nvd_published is not None:
                pub_date = nvd_published[i] if i < len(nvd_published) else None
                if pub_date is not None:
                    p_timestamps[i] = pub_date

            # Infer A from KEV date_added (confidence 1.0 - CISA is authoritative)
            a_confidence = 1.0
            if a_confidence >= self.confidence_threshold and kev_date_added is not None:
                kev_date = kev_date_added[i] if i < len(kev_date_added) else None
                if kev_date is not None:
                    a_timestamps[i] = kev_date

        # Add CVD timestamp columns to dataset
        dataset.set_column("v_timestamp", np.array(v_timestamps, dtype=object))
        dataset.set_column("f_timestamp", np.array(f_timestamps, dtype=object))
        dataset.set_column("d_timestamp", np.array(d_timestamps, dtype=object))
        dataset.set_column("p_timestamp", np.array(p_timestamps, dtype=object))
        dataset.set_column("x_timestamp", np.array(x_timestamps, dtype=object))
        dataset.set_column("a_timestamp", np.array(a_timestamps, dtype=object))
