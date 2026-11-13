"""
cvelistV5 Extractor - CVE JSON 5.x records from CVEProject/cvelistV5

Provides:
- CVEListExtractor: Extracts records from a release zip (URL or path, nested
  zips included) or a directory of CVE-*.json files (registered as "cvelist")

Layer: Extract
Dependencies: utils.cvelist, utils.http, utils.json, registry
Used by: CacheBuilder(source="cvelist"), pipeline

Mapping logic (CVSS, CWE, CPE, references, metadata) is in utils.cvelist.
"""

from __future__ import annotations

import io
import zipfile
from collections.abc import Iterator
from contextlib import contextmanager
from itertools import islice
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

import numpy as np

from vulnflow.exceptions import ExtractorError
from vulnflow.extractors.base import BaseExtractor
from vulnflow.registry import Registry
from vulnflow.utils import json
from vulnflow.utils.cvelist import Validation, populate_cvelist_columns
from vulnflow.utils.http import HttpError, HttpFetcher, is_url

if TYPE_CHECKING:
    from vulnflow.dataset import VulnDataset

INSTALL_HINT = (
    "validate=True needs cve-validation-library. Install the extra: "
    'uv add "vulnflow[cvelist]" (or: pip install "cve-validation-library[fast] '
    '@ git+https://github.com/corifeo/cve-validation-library-py")'
)


def _first_message(result: dict[str, Any]) -> str | None:
    """First detailed error from a validation result, else its summary message."""
    details = result.get("details")
    errors = details.get("errors") if isinstance(details, dict) else details
    if isinstance(errors, list) and errors:
        first = errors[0]
        if isinstance(first, dict):
            where = first.get("instancePath") or first.get("param") or ""
            text = first.get("message") or first.get("msg") or ""
            return f"{where}: {text}" if where else str(text)
        return str(first)
    message = result.get("message")
    return str(message) if message else None


@Registry.register_extractor("cvelist")
class CVEListExtractor(BaseExtractor):
    """Extract cvelistV5 (CVE JSON 5.x) records → VulnDataset.

    Sources:
    - URL of a release asset (``<date>_all_CVEs_at_midnight.zip.zip`` or
      ``<date>_delta_CVEs_at_<HHMM>Z.zip``), downloaded in memory
    - Local path to such a zip (a zip wrapping one inner zip is opened in place)
    - Local directory, searched recursively for ``CVE-*.json``

    Records are read, optionally validated, and mapped in batches to keep memory
    flat. Dates and status use the ``nvd_*`` columns; ``metadata.source`` is
    ``"cvelistV5"``.

    Validation (``validate=True``) runs ``cve_validation_library`` on each record
    before mapping and adds ``cve_valid`` and ``cve_validation_error`` columns.
    The validator trims whitespace in strings in place, so mapped values are the
    trimmed ones. Needs the ``vulnflow[cvelist]`` extra.

    Example:
        >>> ds = CVEListExtractor("cves/").extract()
        >>> ds = CVEListExtractor("2026-11-13_all_CVEs_at_midnight.zip.zip",
        ...                       validate=True).extract()
    """

    name: str = "cvelist"
    format: str = "cvelist"
    BATCH_SIZE: int = 20_000

    def __init__(
        self,
        path: str | Path,
        validate: bool = False,
        cache_dir: Path | str | None = ".cache/cvelist",
        cache_ttl: int | None = None,
        timeout: float = 600.0,
    ) -> None:
        """Initialize cvelistV5 extractor.

        Args:
            path: Release zip URL, local zip path, or directory of CVE-*.json.
            validate: Validate each record with cve-validation-library.
            cache_dir: Directory for HTTP response caching (None for default).
            cache_ttl: HTTP cache time-to-live in seconds (None disables caching).
            timeout: HTTP request timeout in seconds (the full zip is ~660 MB).
        """
        super().__init__()
        self.path = str(path)
        self.validate = validate
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.cache_ttl = cache_ttl
        self.timeout = timeout

    def extract(self) -> VulnDataset:
        """Extract records → VulnDataset.

        Populates self._timings with read, json, validate (if enabled) and map.

        Raises:
            ExtractorError: Missing path, HTTP error, bad zip or invalid JSON.
            ImportError: validate=True without cve-validation-library.
        """
        from vulnflow.dataset import VulnDataset

        validator = self._validator() if self.validate else None
        self._timings.clear()
        with self._timed_sum("read"):
            members = self._read_members()  # lazy: bytes are read one batch at a time

        chunks: list[dict[str, np.ndarray[Any, Any]]] = []
        total = 0
        while True:
            with self._timed_sum("read"):
                batch = list(islice(members, self.BATCH_SIZE))
            if not batch:
                break
            total += len(batch)
            with self._timed_sum("json"):
                records = [self._loads(name, raw) for name, raw in batch]
            validations: list[Validation] | None = None
            if validator is not None:
                with self._timed_sum("validate"):
                    validations = [self._validate_one(validator, r) for r in records]
            with self._timed_sum("map"):
                chunks.append(populate_cvelist_columns(records, validations))

        self.logger.debug("Read %d records from %s", total, self.path)
        if not chunks:
            return VulnDataset()
        arrays = {col: np.concatenate([c[col] for c in chunks]) for col in chunks[0]}
        if validator is not None:
            self.logger.info(
                "%d of %d records failed validation",
                int((~arrays["cve_valid"]).sum()),
                len(arrays["cve_valid"]),
            )

        dataset = VulnDataset()
        dataset._data = arrays
        dataset._record_provenance(extractor=self.provenance_label)
        self._record_date_provenance(dataset, arrays)
        return dataset

    @contextmanager
    def _timed_sum(self, name: str) -> Iterator[None]:
        """Like _timed, but adds up repeated blocks (one per batch)."""
        before = self._timings.get(name, 0.0)
        with self._timed(name):
            yield
        self._timings[name] += before

    @staticmethod
    def _validator() -> Any:
        try:
            from cve_validation_library import Validate
        except ImportError:
            raise ImportError(INSTALL_HINT) from None
        return Validate()

    @staticmethod
    def _validate_one(validator: Any, record: dict[str, Any]) -> Validation:
        result = validator.validate_cve_record(record)  # trims strings in place
        if result.valid:
            return (True, None, None)
        return (False, result.error, _first_message(result.to_dict()))

    def _loads(self, name: str, raw: bytes) -> dict[str, Any]:
        try:
            record = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise ExtractorError(f"Invalid JSON in {name} ({self.path}): {e}") from None
        if not isinstance(record, dict):
            raise ExtractorError(f"{name} ({self.path}) is not a JSON object")
        return record

    def _read_members(self) -> Iterator[tuple[str, bytes]]:
        """(file name, bytes) for every CVE-*.json in the source, read lazily."""
        if is_url(self.path):
            return self._zip_members(io.BytesIO(self._download()))
        path = Path(self.path)
        if path.is_dir():
            files = sorted(path.rglob("CVE-*.json"))
            return ((p.name, p.read_bytes()) for p in files)
        if not path.is_file():
            raise ExtractorError(f"Failed to read {self.path}: no such file")
        return self._zip_members(path)

    def _download(self) -> bytes:
        fetcher = HttpFetcher(
            cache_dir=self.cache_dir or Path(".cache/cvelist"), timeout=self.timeout
        )
        try:
            content, _ = fetcher.fetch_sync(
                self.path,
                self.cache_ttl,
                decompress=False,
                progress_desc=f"Downloading {self.label}",
            )
        except HttpError as e:
            raise ExtractorError(str(e)) from None
        return content

    def _zip_members(self, source: Path | io.BytesIO) -> Iterator[tuple[str, bytes]]:
        try:
            zf = zipfile.ZipFile(source)
            names = zf.namelist()
            # Release zips wrap one inner zip; load it in memory, since seeking
            # inside a compressed member re-decompresses from the start
            while len(names) == 1 and names[0].lower().endswith(".zip"):
                zf = zipfile.ZipFile(io.BytesIO(zf.read(names[0])))
                names = zf.namelist()
        except (zipfile.BadZipFile, OSError) as e:
            raise ExtractorError(
                f"Failed to read {self.path} as a zip archive: {e}"
            ) from None
        wanted = [
            (base, name)
            for name in names
            if (base := PurePosixPath(name).name).startswith("CVE-")
            and base.endswith(".json")
        ]
        return self._iter_zip(zf, wanted)

    def _iter_zip(
        self, zf: zipfile.ZipFile, wanted: list[tuple[str, str]]
    ) -> Iterator[tuple[str, bytes]]:
        for base, name in wanted:
            try:
                yield base, zf.read(name)
            except (zipfile.BadZipFile, OSError) as e:
                raise ExtractorError(
                    f"Failed to read {name} from {self.path}: {e}"
                ) from None
