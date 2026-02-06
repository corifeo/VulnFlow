"""
NVD Extractor - NVD JSON file reader

Provides:
- NVDExtractor: Extracts vulnerability records from NVD JSON/JSONL files or URLs

Layer: Extract
Dependencies: schema, utils
Used by: dataset.from_nvd(), pipeline
"""

from __future__ import annotations

import gzip
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import numpy as np

from vulnflow.extractors.base import BaseExtractor
from vulnflow.registry import Registry
from vulnflow.schema import CPEEntry, CVSSEntry, CWEEntry, ReferenceEntry
from vulnflow.utils import json
from vulnflow.utils.cpe import parse_cpe23
from vulnflow.utils.cvss import extract_cvss_metadata
from vulnflow.utils.http import HttpFetcher, is_url
from vulnflow.utils.schema import get_record_columns

if TYPE_CHECKING:
    from vulnflow.dataset import VulnDataset


@Registry.register_extractor("nvd_json")
class NVDExtractor(BaseExtractor):
    """Extract from NVD JSON/JSONL files or URLs → VulnDataset.

    Parses NVD 2.0 JSON format and converts directly to columnar storage.

    Supported formats:
    - .json: Standard NVD JSON with "vulnerabilities" array
    - .json.gz: Gzip-compressed JSON
    - .jsonl/.ndjson: Newline-delimited JSON (one CVE per line)

    Conversion logic:
    - Single-pass: JSON → dataclasses → columnar storage
    - Uses dataclasses.asdict() for type-safe serialization
    - Pre-allocates arrays for efficiency
    - Encodes complex fields (lists) as JSON strings

    Does NOT populate CVD timestamps (v/f/d/p/x/a) - those come from EventInference.
    Does NOT flatten CVSS/CWE/CPE - use FlattenTransforms for that.
    """

    name: str = "nvd"
    format: str = "nvd_json"

    def __init__(
        self,
        path: str,
        cache_dir: Path | str | None = ".cache/nvd",
        cache_ttl: int | None = None,
        timeout: float = 120.0,
    ) -> None:
        """Initialize NVD extractor.

        Args:
            path: File path or HTTP/HTTPS URL to NVD data
            cache_dir: Directory for HTTP response caching (None to disable)
            cache_ttl: HTTP cache time-to-live in seconds (None disables caching)
            timeout: HTTP request timeout in seconds
        """
        super().__init__()
        self.path = path
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.cache_ttl = cache_ttl
        self.timeout = timeout

    def extract(self) -> VulnDataset:
        """Extract from NVD JSON/JSONL → VulnDataset.

        Single-pass conversion: JSON → dataclasses → columnar storage.
        Uses dataclasses.asdict() for type-safe serialization.

        Returns:
            VulnDataset with columnar storage populated

        Raises:
            FileNotFoundError: If NVD JSON file doesn't exist
            httpx.HTTPError: On network or HTTP errors for URLs
        """
        from dataclasses import asdict

        from vulnflow.dataset import VulnDataset
        from vulnflow.utils.progress import progress_bar

        content = self._load_content()
        vulnerabilities = self._parse_content(content)

        # Empty dataset
        n = len(vulnerabilities)
        if n == 0:
            return VulnDataset()

        # Pre-allocate columnar arrays
        record_cols = get_record_columns()
        arrays = {col: np.empty(n, dtype=object) for col in record_cols}

        # Single-pass: Parse JSON → dataclasses → serialize inline
        with progress_bar(
            total=n,
            desc="Parsing CVEs",
            unit="CVEs",
        ) as pbar:
            for i, vuln in enumerate(vulnerabilities):
                cve_data = vuln.get("cve", {})

                # Identity
                arrays["id"][i] = str(uuid4())
                arrays["cve_id"][i] = cve_data.get("id")

                # Description (extract English)
                description = None
                for desc in cve_data.get("descriptions", []):
                    if desc.get("lang") == "en":
                        description = desc.get("value")
                        break
                arrays["description"][i] = description

                # NVD Temporal
                arrays["nvd_published_at"][i] = self._parse_timestamp(
                    cve_data.get("published")
                )
                arrays["nvd_modified_at"][i] = self._parse_timestamp(
                    cve_data.get("lastModified")
                )
                arrays["nvd_status"][i] = (
                    self._extract_nvd_status(cve_data) or "unknown"
                )

                # Complex fields: Parse to dataclasses → asdict() → JSON
                # Type-safe but still efficient (single pass)

                # CVSS scores
                cvss_entries = self._parse_cvss(cve_data.get("metrics", {}))
                cvss_dicts = [asdict(entry) for entry in cvss_entries]
                arrays["cvss_scores"][i] = (
                    json.dumps(cvss_dicts) if cvss_dicts else "[]"
                )

                # CWEs
                cwe_entries = self._parse_cwes(cve_data.get("weaknesses", []))
                cwe_dicts = [asdict(entry) for entry in cwe_entries]
                arrays["cwes"][i] = json.dumps(cwe_dicts) if cwe_dicts else "[]"

                # CPEs
                cpe_entries = self._parse_cpes(cve_data.get("configurations", []))
                cpe_dicts = [asdict(entry) for entry in cpe_entries]
                arrays["cpes"][i] = json.dumps(cpe_dicts) if cpe_dicts else "[]"

                # References
                ref_entries = self._parse_references(cve_data.get("references", []))
                ref_dicts = [asdict(entry) for entry in ref_entries]
                arrays["references"][i] = json.dumps(ref_dicts) if ref_dicts else "[]"

                # Empty fields (not in NVD)
                arrays["epss_scores"][i] = "[]"
                arrays["kev"][i] = "null"
                arrays["exploits"][i] = "[]"
                arrays["tags"][i] = "{}"
                arrays["metadata"][i] = "{}"

                pbar.update(1)

        # Build dataset
        dataset = VulnDataset()
        dataset._data = arrays
        dataset._record_provenance(extractor=self.provenance_label)
        return dataset

    def _load_content(self) -> bytes:
        """Load content from file or URL.

        Handles:
        - Local files (with gzip support for .gz extension)
        - HTTP/HTTPS URLs (with automatic gzip decompression)

        Returns:
            Raw or decompressed content bytes

        Raises:
            FileNotFoundError: If local file doesn't exist
            httpx.HTTPError: On network or HTTP errors
        """
        if is_url(self.path):
            return self._load_from_url(self.path)
        else:
            return self._load_from_file(Path(self.path))

    def _load_from_file(self, path: Path) -> bytes:
        """Load content from local file.

        Args:
            path: Path to file (supports .gz extension)

        Returns:
            File content bytes (decompressed if .gz)

        Raises:
            FileNotFoundError: If file doesn't exist
        """
        content = path.read_bytes()

        # Auto-decompress gzip by extension or magic bytes
        if path.suffix.lower() == ".gz" or content[:2] == b"\x1f\x8b":
            content = gzip.decompress(content)

        return content

    def _load_from_url(self, url: str) -> bytes:
        """Load content from HTTP/HTTPS URL.

        Args:
            url: URL to fetch

        Returns:
            Response content bytes (auto-decompressed if gzip)

        Raises:
            httpx.HTTPError: On network or HTTP errors
        """
        fetcher = HttpFetcher(
            cache_dir=self.cache_dir or Path(".cache/nvd"),
            timeout=self.timeout,
        )
        content, _ = fetcher.fetch_sync(url, self.cache_ttl)
        return content

    def _parse_content(self, content: bytes) -> list[dict[str, Any]]:
        """Parse content into list of vulnerability dicts.

        Handles:
        - Standard JSON with "vulnerabilities" array
        - JSONL/NDJSON (newline-delimited JSON)

        Args:
            content: Raw content bytes

        Returns:
            List of vulnerability dicts
        """
        # Detect format from path
        path_lower = self.path.lower()
        if path_lower.endswith(".gz"):
            path_lower = path_lower[:-3]

        is_jsonl = path_lower.endswith((".jsonl", ".ndjson"))

        if is_jsonl:
            return self._parse_jsonl(content)
        else:
            return self._parse_json(content)

    def _parse_json(self, content: bytes) -> list[dict[str, Any]]:
        """Parse standard NVD JSON format.

        Args:
            content: JSON content bytes

        Returns:
            List of vulnerability dicts from "vulnerabilities" array
        """
        data = json.loads(content)
        return data.get("vulnerabilities", [])  # type: ignore[no-any-return]

    def _parse_jsonl(self, content: bytes) -> list[dict[str, Any]]:
        """Parse JSONL/NDJSON format (one CVE per line).

        Args:
            content: JSONL content bytes

        Returns:
            List of vulnerability dicts
        """
        vulnerabilities: list[dict[str, Any]] = []
        for line in content.decode("utf-8").splitlines():
            line = line.strip()
            if line:
                obj = json.loads(line)
                # Wrap in vulnerability structure if needed
                if "cve" not in obj:
                    obj = {"cve": obj}
                vulnerabilities.append(obj)
        return vulnerabilities

    def _parse_timestamp(self, ts: str | None) -> datetime | None:
        """Parse ISO timestamp string to datetime."""
        if not ts:
            return None
        # Handle Z suffix
        ts = ts.replace("Z", "+00:00")
        return datetime.fromisoformat(ts)

    def _extract_nvd_status(self, cve_data: dict[str, Any]) -> str | None:
        """Extract and normalize NVD status from vulnStatus field.

        Args:
            cve_data: The CVE data dict

        Returns:
            Normalized status: analyzed, awaiting, modified, rejected, disputed, unknown
        """
        vuln_status = cve_data.get("vulnStatus", "")
        if not vuln_status:
            return None

        status_lower = vuln_status.lower()

        # Map NVD status values to normalized form
        if "analyzed" in status_lower:
            return "analyzed"
        elif "awaiting" in status_lower:
            return "awaiting"
        elif "modified" in status_lower:
            return "modified"
        elif "rejected" in status_lower:
            return "rejected"
        elif "disputed" in status_lower:
            return "disputed"
        elif "received" in status_lower:
            return "awaiting"  # Treat "Received" as awaiting analysis
        elif "undergoing" in status_lower:
            return "awaiting"  # Treat "Undergoing Analysis" as awaiting
        else:
            return "unknown"

    def _parse_cvss(self, metrics: dict[str, list[dict]]) -> list[CVSSEntry]:  # type: ignore[type-arg]
        """Parse CVSS metrics from NVD format.

        Handles version-specific field locations:
        - v3.1/v3.0: baseSeverity in cvssData
        - v2.0: baseSeverity at metric level

        Returns:
            List of CVSSEntry dataclass instances
        """
        cvss_scores: list[CVSSEntry] = []

        # CVSS 3.1 - baseSeverity in cvssData
        for metric in metrics.get("cvssMetricV31", []):
            cvss_data = metric.get("cvssData", {})
            metadata = extract_cvss_metadata(metric)  # Common fields from utility

            cvss_scores.append(
                CVSSEntry(
                    version=cvss_data.get("version", "3.1"),
                    vector_string=cvss_data.get("vectorString", ""),
                    base_score=cvss_data.get("baseScore", 0.0),
                    base_severity=cvss_data.get("baseSeverity", "NONE"),
                    **metadata,  # Unpack: source, is_authoritative, scores
                )
            )

        # CVSS 3.0 - same structure as 3.1
        for metric in metrics.get("cvssMetricV30", []):
            cvss_data = metric.get("cvssData", {})
            metadata = extract_cvss_metadata(metric)

            cvss_scores.append(
                CVSSEntry(
                    version=cvss_data.get("version", "3.0"),
                    vector_string=cvss_data.get("vectorString", ""),
                    base_score=cvss_data.get("baseScore", 0.0),
                    base_severity=cvss_data.get("baseSeverity", "NONE"),
                    **metadata,
                )
            )

        # CVSS 2.0 - baseSeverity at metric level (not in cvssData)
        for metric in metrics.get("cvssMetricV2", []):
            cvss_data = metric.get("cvssData", {})
            metadata = extract_cvss_metadata(metric)

            cvss_scores.append(
                CVSSEntry(
                    version=cvss_data.get("version", "2.0"),
                    vector_string=cvss_data.get("vectorString", ""),
                    base_score=cvss_data.get("baseScore", 0.0),
                    base_severity=metric.get(
                        "baseSeverity", "NONE"
                    ),  # Different location!
                    **metadata,
                )
            )

        return cvss_scores

    def _parse_cwes(self, weaknesses: list[dict[str, object]]) -> list[CWEEntry]:
        """Parse CWE weaknesses from NVD format."""
        cwes: list[CWEEntry] = []

        for weakness in weaknesses:
            descriptions = weakness.get("description", [])
            if not descriptions or not isinstance(descriptions, list):
                continue

            for desc in descriptions:
                if not isinstance(desc, dict):
                    continue
                cwe_id = desc.get("value", "")
                if cwe_id and isinstance(cwe_id, str) and cwe_id.startswith("CWE-"):
                    cwes.append(
                        CWEEntry(
                            id=cwe_id,
                            name=None,
                            category_id=None,
                        )
                    )

        return cwes

    def _parse_cpes(self, configurations: list[dict[str, object]]) -> list[CPEEntry]:
        """Parse CPE configurations from NVD format."""
        cpes: list[CPEEntry] = []

        for config in configurations:
            nodes = config.get("nodes", [])
            if not nodes or not isinstance(nodes, list):
                continue

            for node in nodes:
                if not isinstance(node, dict):
                    continue
                cpe_matches = node.get("cpeMatch", [])
                if not cpe_matches or not isinstance(cpe_matches, list):
                    continue

                for cpe_match in cpe_matches:
                    if not isinstance(cpe_match, dict):
                        continue
                    cpe_str = cpe_match.get("criteria", "")
                    if (
                        cpe_str
                        and isinstance(cpe_str, str)
                        and cpe_str.startswith("cpe:2.3")
                    ):
                        try:
                            cpe = parse_cpe23(cpe_str)
                            # Extract versionEndExcluding as fix_version
                            fix_version = cpe_match.get("versionEndExcluding")
                            if fix_version and isinstance(fix_version, str):
                                cpe.fix_version = fix_version
                            cpes.append(cpe)
                        except (ValueError, AttributeError):
                            # Skip invalid CPE strings or malformed data
                            pass

        return cpes

    def _parse_references(
        self, references: list[dict[str, object]]
    ) -> list[ReferenceEntry]:
        """Parse references from NVD format."""
        ref_entries: list[ReferenceEntry] = []

        for ref in references:
            url = ref.get("url", "")
            if url and isinstance(url, str):
                source = ref.get("source")
                tags = ref.get("tags", [])
                ref_entries.append(
                    ReferenceEntry(
                        url=url,
                        source=source if isinstance(source, str) else None,
                        tags=tags if isinstance(tags, list) else [],
                    )
                )

        return ref_entries
