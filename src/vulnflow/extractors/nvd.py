"""
NVD Extractor - NVD JSON file reader

Provides:
- NVDExtractor: Extracts vulnerability records from NVD JSON/JSONL files or URLs

Layer: Extract
Dependencies: schema, utils
Used by: dataset.from_nvd(), pipeline

Parsing logic (CVSS, CWE, CPE, references, status) is in utils.nvd
for reuse by other extractors (e.g., future CVE.org extractor).
"""

from __future__ import annotations

import gzip
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import numpy as np

from vulnflow.extractors.base import BaseExtractor
from vulnflow.registry import Registry
from vulnflow.utils import json
from vulnflow.utils.http import HttpFetcher, is_url
from vulnflow.utils.nvd import (
    extract_nvd_status,
    parse_cpes,
    parse_cvss,
    parse_cwes,
    parse_references,
    parse_timestamp,
)
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

    @property
    def label(self) -> str:
        """Short display name derived from path.

        Examples: nvdcve-2.0-2023, nvdcve-2.0-modified
        """
        name = Path(self.path).name
        for ext in (".gz", ".json", ".jsonl", ".ndjson"):
            if name.endswith(ext):
                name = name[: -len(ext)]
        return name

    def extract(self) -> VulnDataset:
        """Extract from NVD JSON/JSONL → VulnDataset.

        Single-pass conversion: JSON → dataclasses → columnar storage.
        Uses dataclasses.asdict() for type-safe serialization.

        Populates self._timings with per-phase durations (download,
        decompress, json, populate) for progress reporting.

        Returns:
            VulnDataset with columnar storage populated

        Raises:
            FileNotFoundError: If NVD JSON file doesn't exist
            httpx.HTTPError: On network or HTTP errors for URLs
        """
        from vulnflow.dataset import VulnDataset
        from vulnflow.utils.progress import progress_bar

        timings: dict[str, float] = {}

        t0 = perf_counter()
        raw = self._fetch_raw()
        timings["download"] = perf_counter() - t0

        t0 = perf_counter()
        content = self._decompress(raw)
        timings["decompress"] = perf_counter() - t0

        t0 = perf_counter()
        vulnerabilities = self._parse_content(content)
        timings["json"] = perf_counter() - t0

        # Empty dataset
        n = len(vulnerabilities)
        if n == 0:
            self._timings = timings
            return VulnDataset()

        t0 = perf_counter()

        # Pre-allocate columnar arrays
        record_cols = get_record_columns()
        arrays = {col: np.empty(n, dtype=object) for col in record_cols}

        # Single-pass: Parse JSON → dataclasses → serialize inline
        with progress_bar(
            total=n,
            desc=f"Parsing {self.label}",
            unit="CVEs",
            mininterval=0.5,
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
                arrays["nvd_published_at"][i] = parse_timestamp(
                    cve_data.get("published")
                )
                arrays["nvd_modified_at"][i] = parse_timestamp(
                    cve_data.get("lastModified")
                )
                arrays["nvd_status"][i] = extract_nvd_status(cve_data)

                # Complex fields: Parse to dataclasses → asdict() → JSON
                # Type-safe but still efficient (single pass)

                # CVSS scores
                cvss_entries = parse_cvss(cve_data.get("metrics", {}))
                cvss_dicts = [entry.to_dict() for entry in cvss_entries]
                arrays["cvss_scores"][i] = (
                    json.dumps(cvss_dicts) if cvss_dicts else "[]"
                )

                # CWEs
                cwe_entries = parse_cwes(cve_data.get("weaknesses", []))
                cwe_dicts = [entry.to_dict() for entry in cwe_entries]
                arrays["cwes"][i] = json.dumps(cwe_dicts) if cwe_dicts else "[]"

                # CPEs
                cpe_entries = parse_cpes(cve_data.get("configurations", []))
                cpe_dicts = [entry.to_dict() for entry in cpe_entries]
                arrays["cpes"][i] = json.dumps(cpe_dicts) if cpe_dicts else "[]"

                # References
                ref_entries = parse_references(cve_data.get("references", []))
                ref_dicts = [entry.to_dict() for entry in ref_entries]
                arrays["references"][i] = json.dumps(ref_dicts) if ref_dicts else "[]"

                # Empty fields (not in NVD)
                arrays["epss_scores"][i] = "[]"
                arrays["kev"][i] = "null"
                arrays["exploits"][i] = "[]"
                arrays["tags"][i] = "{}"
                arrays["metadata"][i] = "{}"

                pbar.update(1)

        timings["populate"] = perf_counter() - t0

        # Build dataset
        dataset = VulnDataset()
        dataset._data = arrays
        dataset._record_provenance(extractor=self.provenance_label)
        self._timings = timings
        return dataset

    def _fetch_raw(self) -> bytes:
        """Fetch raw bytes without decompression.

        For URLs, uses HttpFetcher with decompress=False to allow
        separate timing of download vs decompression.

        Returns:
            Raw content bytes (may be gzip-compressed)

        Raises:
            FileNotFoundError: If local file doesn't exist
            HttpError: On network or HTTP errors for URLs
        """
        if is_url(self.path):
            fetcher = HttpFetcher(
                cache_dir=self.cache_dir or Path(".cache/nvd"),
                timeout=self.timeout,
            )
            content, _ = fetcher.fetch_sync(
                self.path,
                self.cache_ttl,
                decompress=False,
                progress_desc=f"Downloading {self.label}",
            )
            return content
        return Path(self.path).read_bytes()

    @staticmethod
    def _decompress(raw: bytes) -> bytes:
        """Decompress gzip if detected by magic bytes.

        Args:
            raw: Raw content bytes (possibly gzip-compressed)

        Returns:
            Decompressed content, or original bytes if not gzip
        """
        if raw[:2] == b"\x1f\x8b":
            return gzip.decompress(raw)
        return raw

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
