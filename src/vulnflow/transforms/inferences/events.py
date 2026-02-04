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
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from vulnflow.dataset import VulnDataset

from vulnflow.registry import FieldDefinition, register_transform
from vulnflow.transforms.base import BaseInference
from vulnflow.utils import json


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


def _parse_references(refs_json: str | None) -> list[dict[str, Any]]:
    """Parse references JSON string into list of dicts."""
    if not refs_json or refs_json == "[]":
        return []
    try:
        return json.loads(refs_json)
    except Exception:
        # orjson raises orjson.JSONDecodeError, catch broadly
        return []


def _find_earliest_date_by_tag(
    refs: list[dict[str, Any]], tag: str
) -> datetime | None:
    """Find earliest date from references with a specific tag.

    Args:
        refs: List of reference dicts with 'url' and 'tags' keys
        tag: Tag to filter by (e.g., 'Patch', 'Exploit', 'Vendor Advisory')

    Returns:
        Earliest datetime found, or None if no matching references
    """
    dates: list[datetime] = []

    for ref in refs:
        tags = ref.get("tags", [])
        if tag in tags:
            url = ref.get("url")
            date = extract_date_from_url(url)
            if date is not None:
                dates.append(date)

    return min(dates) if dates else None


@register_transform(default=True)
class EventInference(BaseInference):
    """Infer missing CVD timestamps from references, NVD metadata, KEV.

    Creates columns: v_timestamp, f_timestamp, d_timestamp, p_timestamp, x_timestamp, a_timestamp

    Inference rules (with confidence levels):
    - V: "Vendor Advisory" tag → earliest URL date (confidence 0.7)
    - F: "Patch" tag → earliest URL date (confidence 0.8)
    - D: Not inferred (requires deployment telemetry)
    - P: nvd_published_at (confidence 0.9 - NVD is authoritative)
    - X: "Exploit" tag → earliest URL date (confidence 0.8)
    - A: KEV date_added (confidence 1.0 - CISA is authoritative)

    Constraint enforcement:
    - V must be <= F (if both exist)
    - F must be <= D (if both exist)
    - If F inferred but V not, V is set to F (vendor must know before fix)
    """

    name: str = "EventInference"
    depends_on: list[str] = ["KEVEnricher"]

    SCHEMA_VERSION: str = "2.0.0"
    OUTPUT_SCHEMA: list[FieldDefinition] = [
        FieldDefinition(
            "v_timestamp", "datetime", "Vendor awareness timestamp", "EventInference"
        ),
        FieldDefinition(
            "f_timestamp", "datetime", "Fix ready timestamp", "EventInference"
        ),
        FieldDefinition(
            "d_timestamp", "datetime", "Fix deployed timestamp", "EventInference"
        ),
        FieldDefinition(
            "p_timestamp", "datetime", "Public awareness timestamp", "EventInference"
        ),
        FieldDefinition(
            "x_timestamp", "datetime", "Exploit public timestamp", "EventInference"
        ),
        FieldDefinition(
            "a_timestamp", "datetime", "Attack observed timestamp", "EventInference"
        ),
    ]

    def __init__(self, confidence_threshold: float = 0.7) -> None:
        """Initialize EventInference.

        Args:
            confidence_threshold: Minimum confidence (0.0-1.0) to populate field
        """
        super().__init__()
        self.confidence_threshold = confidence_threshold

    def transform(self, dataset: VulnDataset) -> None:
        """Create CVD timestamp columns from available metadata.

        Args:
            dataset: VulnDataset to transform (adds columns)
        """
        n = len(dataset)

        # Initialize CVD timestamp arrays
        v_timestamps = np.full(n, None, dtype=object)
        f_timestamps = np.full(n, None, dtype=object)
        d_timestamps = np.full(n, None, dtype=object)
        p_timestamps = np.full(n, None, dtype=object)
        x_timestamps = np.full(n, None, dtype=object)
        a_timestamps = np.full(n, None, dtype=object)

        # Get source columns
        nvd_published = dataset.get_column_or_none("nvd_published_at")
        kev_date_added = dataset.get_column_or_none("kev_date_added")
        references = dataset.get_column_or_none("references")

        # === P inference (confidence 0.9) ===
        # NVD published date is authoritative for public disclosure
        if self.confidence_threshold <= 0.9 and nvd_published is not None:
            valid_mask = nvd_published != None  # noqa: E711
            p_timestamps[valid_mask] = nvd_published[valid_mask]

        # === A inference (confidence 1.0) ===
        # CISA KEV is authoritative for attack observations
        if self.confidence_threshold <= 1.0 and kev_date_added is not None:
            valid_mask = kev_date_added != None  # noqa: E711
            a_timestamps[valid_mask] = kev_date_added[valid_mask]

        # === Reference-based inference (F, X, V) ===
        if references is not None:
            for i in range(n):
                refs = _parse_references(references[i])
                if not refs:
                    continue

                # F inference (confidence 0.8) - Patch tag
                if self.confidence_threshold <= 0.8:
                    f_date = _find_earliest_date_by_tag(refs, "Patch")
                    if f_date is not None:
                        f_timestamps[i] = f_date

                # X inference (confidence 0.8) - Exploit tag
                if self.confidence_threshold <= 0.8:
                    x_date = _find_earliest_date_by_tag(refs, "Exploit")
                    if x_date is not None:
                        x_timestamps[i] = x_date

                # V inference (confidence 0.7) - Vendor Advisory tag
                if self.confidence_threshold <= 0.7:
                    v_date = _find_earliest_date_by_tag(refs, "Vendor Advisory")
                    if v_date is not None:
                        v_timestamps[i] = v_date

        # === Constraint enforcement: V → F → D ===
        # If F exists but V doesn't, set V = F (vendor must know before fix)
        for i in range(n):
            if f_timestamps[i] is not None and v_timestamps[i] is None:
                v_timestamps[i] = f_timestamps[i]
            # Ensure V <= F (if both exist and V > F, adjust V)
            elif (
                f_timestamps[i] is not None
                and v_timestamps[i] is not None
                and v_timestamps[i] > f_timestamps[i]
            ):
                v_timestamps[i] = f_timestamps[i]

        # Add CVD timestamp columns to dataset
        dataset.set_column("v_timestamp", v_timestamps)
        dataset.set_column("f_timestamp", f_timestamps)
        dataset.set_column("d_timestamp", d_timestamps)
        dataset.set_column("p_timestamp", p_timestamps)
        dataset.set_column("x_timestamp", x_timestamps)
        dataset.set_column("a_timestamp", a_timestamps)
