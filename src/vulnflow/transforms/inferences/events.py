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

import logging
import re
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from vulnflow.dataset import VulnDataset

from vulnflow.registry import FieldDefinition, register_transform
from vulnflow.transforms.base import BaseInference
from vulnflow.utils import json

logger = logging.getLogger(__name__)

logger = logging.getLogger(__name__)

# Pre-compiled regexes for performance
_URL_DATE_SLASH = re.compile(r"/(\d{4})/(\d{2})/(\d{2})")
_URL_DATE_ISO = re.compile(r"(\d{4})-(\d{2})-(\d{2})")
_EXPLOIT_URL_PATTERN = re.compile(
    r"exploit-db\.com|exploitdb|metasploit|/poc/|/proof-of-concept|nuclei-templates",
    re.IGNORECASE,
)

# Tier confidence thresholds
TIER_THRESHOLDS = {
    1: 0.95,  # Authoritative sources
    2: 0.80,  # URL date extraction
    3: 0.60,  # Tag presence
    4: 0.50,  # Structural signals
    5: 0.30,  # Age-based (spectrum down to 0.05)
}

# Age-based inference spectrum (confidence -> minimum age)
AGE_SPECTRUM = [
    (0.30, timedelta(days=5 * 365)),  # 5 years
    (0.25, timedelta(days=3 * 365)),  # 3 years
    (0.20, timedelta(days=2 * 365)),  # 2 years
    (0.15, timedelta(days=365)),  # 1 year
    (0.10, timedelta(days=90)),  # 3 months
    (0.05, timedelta(days=30)),  # 1 month
]

# Statistical estimates for age-based inference
AVG_V_BEFORE_P = timedelta(days=90)  # Vendor knows ~90 days before public
AVG_F_BEFORE_P = timedelta(days=30)  # Fix ready ~30 days before public


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
    date_match = _URL_DATE_SLASH.search(url)
    if date_match:
        year, month, day = date_match.groups()
        try:
            return datetime(int(year), int(month), int(day), tzinfo=UTC)
        except ValueError:
            pass

    # Pattern 2: YYYY-MM-DD in path or query
    iso_match = _URL_DATE_ISO.search(url)
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
        result = json.loads(refs_json)
        return result if isinstance(result, list) else []
    except Exception:
        logger.warning("Failed to parse references: %s", refs_json[:100])
        return []


def _has_tag(refs: list[dict[str, Any]], tag: str) -> bool:
    """Check if any reference has the specified tag."""
    for ref in refs:
        if tag in ref.get("tags", []):
            return True
    return False


def _has_exploit_url(refs: list[dict[str, Any]]) -> bool:
    """Check if any reference URL matches exploit patterns."""
    for ref in refs:
        url = ref.get("url", "")
        if _EXPLOIT_URL_PATTERN.search(url):
            return True
    return False


def _find_earliest_date_by_tag(refs: list[dict[str, Any]], tag: str) -> datetime | None:
    """Find earliest date from references with a specific tag."""
    dates: list[datetime] = []
    for ref in refs:
        if tag in ref.get("tags", []):
            date = extract_date_from_url(ref.get("url"))
            if date is not None:
                dates.append(date)
    return min(dates) if dates else None


@register_transform(default=False)
class EventInference(BaseInference):
    """Infer CVD timestamps using tiered confidence rules.

    Uses a 5-tier inference system where lower confidence thresholds enable
    more aggressive inference rules. Each tier adds coverage at the cost of
    precision.

    Tier 1 (≥0.95) - Authoritative Sources:
        P = nvd_published_at, A = kev_date_added

    Tier 2 (≥0.80) - URL Date Extraction:
        V/F/X from tagged reference URLs containing dates

    Tier 3 (≥0.60) - Tag Presence:
        V/F/X = P when relevant tag exists (Vendor Advisory, Patch, Exploit)

    Tier 4 (≥0.50) - Structural Signals:
        F from CPE versionEndExcluding, X from exploit URL patterns,
        V = P (coordinated disclosure assumption)

    Tier 5 (0.30→0.05) - Age-Based Assumptions:
        V and F estimated for old CVEs using statistical averages.
        Confidence determines age threshold (0.30=5yr, 0.05=1mo).
        X is NOT inferred by age alone - requires evidence.

    Attributes:
        confidence_threshold: Minimum confidence to run inference rules.
            Higher values = more conservative, lower coverage.
            Lower values = more aggressive, higher coverage.
        overwrite: If True, replace existing timestamps with fresh inference.
            Default False preserves existing values.
        tiers: Optional list of specific tiers to run (e.g., [5] for age-only).
            Default None runs all tiers meeting confidence threshold.

    Example:
        # Default: tiers 1-3 (balanced)
        EventInference().transform(dataset)

        # Conservative: only authoritative sources
        EventInference(confidence_threshold=0.95).transform(dataset)

        # Aggressive: all tiers including age-based
        EventInference(confidence_threshold=0.1).transform(dataset)

        # Manual: only age-based for old CVEs
        EventInference(tiers=[5], confidence_threshold=0.2).transform(dataset)

        # Refresh with better data
        EventInference(overwrite=True).transform(dataset)
    """

    name: str = "EventInference"
    depends_on: list[str] = ["KEVFlattenTransform"]

    SCHEMA_VERSION: str = "3.0.0"
    OUTPUT_SCHEMA: list[FieldDefinition] = [
        # Timestamps
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
        # Confidence scores
        FieldDefinition(
            "v_confidence",
            "float",
            "Confidence level for V inference",
            "EventInference",
        ),
        FieldDefinition(
            "f_confidence",
            "float",
            "Confidence level for F inference",
            "EventInference",
        ),
        FieldDefinition(
            "d_confidence",
            "float",
            "Confidence level for D inference",
            "EventInference",
        ),
        FieldDefinition(
            "p_confidence",
            "float",
            "Confidence level for P inference",
            "EventInference",
        ),
        FieldDefinition(
            "x_confidence",
            "float",
            "Confidence level for X inference",
            "EventInference",
        ),
        FieldDefinition(
            "a_confidence",
            "float",
            "Confidence level for A inference",
            "EventInference",
        ),
    ]

    def __init__(
        self,
        confidence_threshold: float = 0.7,
        overwrite: bool = False,
        tiers: list[int] | None = None,
    ) -> None:
        """Initialize EventInference.

        Args:
            confidence_threshold: Minimum confidence (0.0-1.0) to run a tier.
                Default 0.7 runs tiers 1-3. Use 0.1 for maximum coverage.
            overwrite: Replace existing timestamps. Default False preserves
                values from previous inference runs.
            tiers: Specific tiers to run (1-5). Default None runs all tiers
                meeting the confidence threshold. Use [5] for age-only.
        """
        super().__init__()
        self.confidence_threshold = confidence_threshold
        self.overwrite = overwrite
        self.tiers = tiers

    def _should_run(self, tier: int) -> bool:
        """Check if a tier should run based on configuration."""
        if self.tiers is not None:
            return tier in self.tiers
        return self.confidence_threshold <= TIER_THRESHOLDS[tier]

    def _assign(
        self,
        timestamps: np.ndarray,
        confidences: np.ndarray,
        idx: int,
        value: datetime,
        confidence: float,
    ) -> bool:
        """Assign timestamp if not already set (or overwrite enabled).

        Returns True if assignment was made.
        """
        if self.overwrite or timestamps[idx] is None:
            timestamps[idx] = value
            confidences[idx] = confidence
            return True
        return False

    def _tier1_authoritative(
        self,
        dataset: VulnDataset,
        p_ts: np.ndarray,
        p_conf: np.ndarray,
        a_ts: np.ndarray,
        a_conf: np.ndarray,
    ) -> dict[str, int]:
        """Tier 1: Authoritative sources (P from NVD, A from KEV)."""
        stats = {"p": 0, "a": 0}

        nvd_published = dataset.get_column_or_none("nvd_published_at")
        kev_date_added = dataset.get_column_or_none("kev_date_added")

        if nvd_published is not None:
            for i, val in enumerate(nvd_published):
                if val is not None and self._assign(p_ts, p_conf, i, val, 0.95):
                    stats["p"] += 1

        if kev_date_added is not None:
            for i, val in enumerate(kev_date_added):
                if val is not None and self._assign(a_ts, a_conf, i, val, 0.95):
                    stats["a"] += 1

        return stats

    def _tier2_url_dates(
        self,
        parsed_refs: list[list[dict[str, Any]]],
        v_ts: np.ndarray,
        v_conf: np.ndarray,
        f_ts: np.ndarray,
        f_conf: np.ndarray,
        x_ts: np.ndarray,
        x_conf: np.ndarray,
    ) -> dict[str, int]:
        """Tier 2: Extract dates from tagged reference URLs."""
        stats = {"v": 0, "f": 0, "x": 0}

        for i, refs in enumerate(parsed_refs):
            if not refs:
                continue

            # V from Vendor Advisory URL date
            v_date = _find_earliest_date_by_tag(refs, "Vendor Advisory")
            if v_date and self._assign(v_ts, v_conf, i, v_date, 0.80):
                stats["v"] += 1

            # F from Patch URL date
            f_date = _find_earliest_date_by_tag(refs, "Patch")
            if f_date and self._assign(f_ts, f_conf, i, f_date, 0.80):
                stats["f"] += 1

            # X from Exploit URL date
            x_date = _find_earliest_date_by_tag(refs, "Exploit")
            if x_date and self._assign(x_ts, x_conf, i, x_date, 0.80):
                stats["x"] += 1

        return stats

    def _tier3_tag_presence(
        self,
        parsed_refs: list[list[dict[str, Any]]],
        p_ts: np.ndarray,
        v_ts: np.ndarray,
        v_conf: np.ndarray,
        f_ts: np.ndarray,
        f_conf: np.ndarray,
        x_ts: np.ndarray,
        x_conf: np.ndarray,
    ) -> dict[str, int]:
        """Tier 3: Infer from tag presence (timestamp = P)."""
        stats = {"v": 0, "f": 0, "x": 0}

        for i, refs in enumerate(parsed_refs):
            if not refs or p_ts[i] is None:
                continue

            # V = P when Vendor Advisory tag exists
            if _has_tag(refs, "Vendor Advisory"):
                if self._assign(v_ts, v_conf, i, p_ts[i], 0.60):
                    stats["v"] += 1

            # F = P when Patch tag exists
            if _has_tag(refs, "Patch"):
                if self._assign(f_ts, f_conf, i, p_ts[i], 0.60):
                    stats["f"] += 1

            # X = P when Exploit tag exists
            if _has_tag(refs, "Exploit"):
                if self._assign(x_ts, x_conf, i, p_ts[i], 0.60):
                    stats["x"] += 1

        return stats

    def _tier4_structural(
        self,
        dataset: VulnDataset,
        parsed_refs: list[list[dict[str, Any]]],
        p_ts: np.ndarray,
        v_ts: np.ndarray,
        v_conf: np.ndarray,
        f_ts: np.ndarray,
        f_conf: np.ndarray,
        x_ts: np.ndarray,
        x_conf: np.ndarray,
        a_ts: np.ndarray,
    ) -> dict[str, int]:
        """Tier 4: Structural signals (CPE, URL patterns, coordinated disclosure)."""
        stats = {"v": 0, "f": 0, "x": 0}
        n = len(dataset)

        # Check for CPE fix version indicator
        cpes = dataset.get_column_or_none("cpes")

        for i in range(n):
            if p_ts[i] is None:
                continue

            # F from CPE versionEndExcluding (fix version identified)
            if cpes is not None and cpes[i]:
                try:
                    cpe_data = (
                        json.loads(cpes[i]) if isinstance(cpes[i], str) else cpes[i]
                    )
                    # If there's version info, a fix likely exists
                    if isinstance(cpe_data, list) and len(cpe_data) > 0:
                        if self._assign(f_ts, f_conf, i, p_ts[i], 0.50):
                            stats["f"] += 1
                except Exception:
                    logger.warning("Failed to parse CPE data for record %d", i)

            # X from exploit URL patterns
            refs = parsed_refs[i] if i < len(parsed_refs) else []
            if refs and _has_exploit_url(refs):
                if self._assign(x_ts, x_conf, i, p_ts[i], 0.50):
                    stats["x"] += 1

            # X from A (attack implies exploit existed)
            if a_ts[i] is not None:
                if self._assign(x_ts, x_conf, i, a_ts[i], 0.50):
                    stats["x"] += 1

            # V = P (coordinated disclosure assumption)
            if self._assign(v_ts, v_conf, i, p_ts[i], 0.50):
                stats["v"] += 1

        return stats

    def _tier5_age_based(
        self,
        p_ts: np.ndarray,
        v_ts: np.ndarray,
        v_conf: np.ndarray,
        f_ts: np.ndarray,
        f_conf: np.ndarray,
    ) -> dict[str, int]:
        """Tier 5: Age-based assumptions for V and F only."""
        stats = {"v": 0, "f": 0}
        now = datetime.now(UTC)

        # Find applicable age threshold based on confidence
        age_threshold: timedelta | None = None
        conf_value: float = 0.30  # Default to lowest tier 5 confidence
        for conf, age in AGE_SPECTRUM:
            if self.confidence_threshold <= conf:
                age_threshold = age
                conf_value = conf
                break

        if age_threshold is None:
            return stats

        for i, p in enumerate(p_ts):
            if p is None:
                continue

            # Normalize timezone for comparison
            p_aware = p if p.tzinfo else p.replace(tzinfo=UTC)
            age = now - p_aware

            if age >= age_threshold:
                # V = P - 90 days (average vendor awareness lead time)
                v_estimate = p_aware - AVG_V_BEFORE_P
                if self._assign(v_ts, v_conf, i, v_estimate, conf_value):
                    stats["v"] += 1

                # F = P - 30 days (average fix lead time)
                f_estimate = p_aware - AVG_F_BEFORE_P
                if self._assign(f_ts, f_conf, i, f_estimate, conf_value):
                    stats["f"] += 1

        return stats

    def _enforce_constraints(
        self,
        v_ts: np.ndarray,
        v_conf: np.ndarray,
        f_ts: np.ndarray,
        f_conf: np.ndarray,
        x_ts: np.ndarray,
        x_conf: np.ndarray,
        a_ts: np.ndarray,
    ) -> None:
        """Enforce CVD timeline constraints."""
        n = len(v_ts)

        for i in range(n):
            # If F exists but V doesn't, set V = F
            if f_ts[i] is not None and v_ts[i] is None:
                v_ts[i] = f_ts[i]
                v_conf[i] = f_conf[i] if f_conf[i] is not None else 0.50

            # Ensure V <= F (if V > F, adjust V)
            elif f_ts[i] is not None and v_ts[i] is not None:
                v_tz = v_ts[i] if v_ts[i].tzinfo else v_ts[i].replace(tzinfo=UTC)
                f_tz = f_ts[i] if f_ts[i].tzinfo else f_ts[i].replace(tzinfo=UTC)
                if v_tz > f_tz:
                    v_ts[i] = f_ts[i]

            # If A exists but X doesn't, set X = A (attack implies exploit)
            if a_ts[i] is not None and x_ts[i] is None:
                x_ts[i] = a_ts[i]
                x_conf[i] = 0.50

    def _log_stats(
        self,
        n: int,
        tier_stats: dict[int, dict[str, int]],
        v_ts: np.ndarray,
        f_ts: np.ndarray,
        p_ts: np.ndarray,
        x_ts: np.ndarray,
        a_ts: np.ndarray,
    ) -> None:
        """Log inference coverage statistics."""
        # Per-tier breakdown (DEBUG)
        for tier, stats in sorted(tier_stats.items()):
            parts = [f"{k.upper()}+={v}" for k, v in stats.items() if v > 0]
            if parts:
                tier_name = {
                    1: "authoritative",
                    2: "url_dates",
                    3: "tag_presence",
                    4: "structural",
                    5: "age_based",
                }.get(tier, str(tier))
                logger.debug(f"Tier {tier} ({tier_name}): {', '.join(parts)}")

        # Final coverage (INFO)
        def pct(arr: np.ndarray) -> str:
            count = np.sum([x is not None for x in arr])
            return f"{count / n * 100:.1f}%"

        logger.info(
            f"EventInference coverage: V={pct(v_ts)} F={pct(f_ts)} "
            f"P={pct(p_ts)} X={pct(x_ts)} A={pct(a_ts)}"
        )

    def transform(self, dataset: VulnDataset) -> None:
        """Create CVD timestamp columns from available metadata.

        Runs applicable inference tiers based on confidence_threshold,
        then enforces CVD timeline constraints.

        Args:
            dataset: VulnDataset to transform (adds timestamp and confidence columns)
        """
        n = len(dataset)

        # Initialize arrays
        v_ts = np.full(n, None, dtype=object)
        f_ts = np.full(n, None, dtype=object)
        d_ts = np.full(n, None, dtype=object)  # Not inferred
        p_ts = np.full(n, None, dtype=object)
        x_ts = np.full(n, None, dtype=object)
        a_ts = np.full(n, None, dtype=object)

        v_conf = np.full(n, None, dtype=object)
        f_conf = np.full(n, None, dtype=object)
        d_conf = np.full(n, None, dtype=object)
        p_conf = np.full(n, None, dtype=object)
        x_conf = np.full(n, None, dtype=object)
        a_conf = np.full(n, None, dtype=object)

        # Load existing values if not overwriting
        if not self.overwrite:
            for col, arr in [
                ("v_timestamp", v_ts),
                ("f_timestamp", f_ts),
                ("d_timestamp", d_ts),
                ("p_timestamp", p_ts),
                ("x_timestamp", x_ts),
                ("a_timestamp", a_ts),
            ]:
                existing = dataset.get_column_or_none(col)
                if existing is not None:
                    arr[:] = existing

            for col, arr in [
                ("v_confidence", v_conf),
                ("f_confidence", f_conf),
                ("d_confidence", d_conf),
                ("p_confidence", p_conf),
                ("x_confidence", x_conf),
                ("a_confidence", a_conf),
            ]:
                existing = dataset.get_column_or_none(col)
                if existing is not None:
                    arr[:] = existing

        # Parse references once for reuse across tiers
        references = dataset.get_column_or_none("references")
        parsed_refs: list[list[dict[str, Any]]] = []
        if references is not None:
            parsed_refs = [_parse_references(r) for r in references]

        tier_stats: dict[int, dict[str, int]] = {}

        # Run tiers
        if self._should_run(1):
            tier_stats[1] = self._tier1_authoritative(
                dataset, p_ts, p_conf, a_ts, a_conf
            )

        if self._should_run(2):
            tier_stats[2] = self._tier2_url_dates(
                parsed_refs, v_ts, v_conf, f_ts, f_conf, x_ts, x_conf
            )

        if self._should_run(3):
            tier_stats[3] = self._tier3_tag_presence(
                parsed_refs, p_ts, v_ts, v_conf, f_ts, f_conf, x_ts, x_conf
            )

        if self._should_run(4):
            tier_stats[4] = self._tier4_structural(
                dataset,
                parsed_refs,
                p_ts,
                v_ts,
                v_conf,
                f_ts,
                f_conf,
                x_ts,
                x_conf,
                a_ts,
            )

        if self._should_run(5):
            tier_stats[5] = self._tier5_age_based(p_ts, v_ts, v_conf, f_ts, f_conf)

        # Enforce constraints
        self._enforce_constraints(v_ts, v_conf, f_ts, f_conf, x_ts, x_conf, a_ts)

        # Log statistics
        self._log_stats(n, tier_stats, v_ts, f_ts, p_ts, x_ts, a_ts)

        # Set columns
        dataset.set_column("v_timestamp", v_ts)
        dataset.set_column("f_timestamp", f_ts)
        dataset.set_column("d_timestamp", d_ts)
        dataset.set_column("p_timestamp", p_ts)
        dataset.set_column("x_timestamp", x_ts)
        dataset.set_column("a_timestamp", a_ts)

        dataset.set_column("v_confidence", v_conf)
        dataset.set_column("f_confidence", f_conf)
        dataset.set_column("d_confidence", d_conf)
        dataset.set_column("p_confidence", p_conf)
        dataset.set_column("x_confidence", x_conf)
        dataset.set_column("a_confidence", a_conf)
