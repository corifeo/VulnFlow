"""
VulnDataset - Core container for vulnerability records

Provides:
- VulnDataset: Main class for ETL operations on vulnerability data

Layer: Core
Dependencies: schema
Used by: __init__, pipeline, user code

Storage: Columnar (dict[str, np.ndarray]) for vectorized operations at scale
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl

    from vulnflow.schema import VulnRecord


def _topological_sort(transforms: list[object]) -> list[object]:
    """Sort transforms by dependencies using simple O(n²) algorithm.

    Limitations vs. Kahn's algorithm:
    - No parallel execution groups (can't identify independent transforms)
    - Lenient on missing dependencies (assumes already satisfied)
    - Slower O(n²) vs O(n+e) for large transform lists
    - Good enough: typical usage has <10 transforms

    Args:
        transforms: List of transform instances with name and depends_on attributes

    Returns:
        Sorted list where dependencies come before dependents

    Raises:
        ValueError: If circular dependencies detected
    """
    by_name = {t.name: t for t in transforms}
    result = []
    remaining = list(transforms)

    while remaining:
        # Find transforms with all dependencies satisfied
        ready = [
            t for t in remaining
            if all(
                dep in [r.name for r in result] or dep not in by_name
                for dep in getattr(t, "depends_on", [])
            )
        ]

        if not ready:
            remaining_names = [t.name for t in remaining]
            raise ValueError(f"Circular dependency among: {remaining_names}")

        result.extend(ready)
        for t in ready:
            remaining.remove(t)

    return result


class VulnDataset:
    """Columnar storage for vulnerability records.

    Internal storage uses dict[str, np.ndarray] for vectorized operations.
    Public API provides extractors, transformers, and loaders.
    """

    def __init__(self) -> None:
        """Initialize empty dataset."""
        self._data: dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        """Return number of records."""
        if not self._data:
            return 0
        # Get length from first column
        first_key = next(iter(self._data.keys()))
        return len(self._data[first_key])

    @property
    def columns(self) -> list[str]:
        """Return list of column names.

        Returns:
            List of column names in dataset
        """
        return list(self._data.keys())

    def get_column(self, name: str) -> np.ndarray:
        """Get column data as numpy array.

        Args:
            name: Column name

        Returns:
            Numpy array with column data

        Raises:
            KeyError: If column does not exist
        """
        return self._data[name]

    def get_column_or_none(self, name: str) -> np.ndarray | None:
        """Get column by name, returning None if not present.

        Args:
            name: Column name

        Returns:
            Column data as numpy array, or None if column doesn't exist
        """
        return self._data.get(name)

    def set_column(self, name: str, data: np.ndarray) -> None:
        """Set column data (add new or update existing).

        Args:
            name: Column name
            data: Numpy array with column data

        Raises:
            ValueError: If data length doesn't match dataset length
        """
        if self._data and len(data) != len(self):
            msg = f"Column length mismatch: expected {len(self)}, got {len(data)}"
            raise ValueError(msg)

        self._data[name] = data

    def transform(self, transforms: list[object] | None = None) -> None:
        """Apply transforms and enrichers to dataset (modifies in place).

        Handles both:
        - Sync transforms (with sync transform() method)
        - Async enrichers (with async transform() method) - run concurrently via asyncio.gather()

        Args:
            transforms: List of transform/enricher instances. If None, uses registered defaults
                       (both enrichers and transforms).
        """
        import asyncio
        import inspect

        if transforms is None:
            from vulnflow.registry import Registry

            # Get both default enrichers and transforms
            enrichers = Registry.get_default_enrichers()
            computed_transforms = Registry.get_default_transforms()
            transforms = enrichers + computed_transforms

        # Sort by dependencies
        transforms = _topological_sort(transforms)

        # Separate async from sync
        async_transforms = [
            t
            for t in transforms
            if hasattr(t, "transform") and inspect.iscoroutinefunction(t.transform)
        ]
        sync_transforms = [
            t
            for t in transforms
            if hasattr(t, "transform") and not inspect.iscoroutinefunction(t.transform)
        ]

        # Run async concurrently
        if async_transforms:
            # Check if we're already in an async context
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                # No event loop running - create one with asyncio.run()
                # Use a wrapper coroutine to avoid creating gather future before loop exists
                async def run_all():
                    return await asyncio.gather(*[t.transform(self) for t in async_transforms])

                asyncio.run(run_all())
            else:
                # Event loop is running - run async code in a separate thread
                # This allows transform() to be called from async test contexts
                import threading

                exception_container = []

                def run_in_new_loop():
                    """Run coroutine in a new event loop in this thread."""
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    try:
                        # Create the gather coroutine in this thread's loop
                        async def run_all():
                            return await asyncio.gather(*[t.transform(self) for t in async_transforms])

                        new_loop.run_until_complete(run_all())
                    except Exception as e:
                        exception_container.append(e)
                    finally:
                        new_loop.close()

                thread = threading.Thread(target=run_in_new_loop)
                thread.start()
                thread.join()

                # Re-raise any exception that occurred
                if exception_container:
                    raise exception_container[0]

        # Run sync sequentially
        for t in sync_transforms:
            t.transform(self)

    @classmethod
    def _from_records(cls, records: list[VulnRecord]) -> VulnDataset:
        """Convert list of VulnRecord to columnar storage.

        Args:
            records: List of VulnRecord instances

        Returns:
            VulnDataset with columnar storage

        Note:
            Complex fields (lists, nested dataclasses) stored as JSON strings.
            Flattened columns (cvss_base_score, etc.) added for DataFrame convenience.
            CVD timestamps are NOT stored here - EventInference adds them later.
        """
        import json
        from dataclasses import is_dataclass
        from typing import Any

        from vulnflow.schema import CPEEntry, CVSSEntry, CWEEntry

        # Create JSON encoder once for reuse
        json_encoder = json.JSONEncoder(default=str)

        def dataclass_to_dict(obj: Any) -> Any:
            """Fast dataclass to dict conversion without asdict overhead."""
            if obj is None:
                return None
            if is_dataclass(obj) and not isinstance(obj, type):
                # Manually extract fields instead of using asdict()
                return {f.name: getattr(obj, f.name) for f in obj.__dataclass_fields__.values()}  # type: ignore[union-attr]
            return obj

        def encode_nested(value: Any) -> str:
            """Encode nested structure as JSON string."""
            if value is None:
                return "null"
            if isinstance(value, list):
                # Fast path for empty lists
                if not value:
                    return "[]"
                # Convert dataclasses in list to dicts
                items = [dataclass_to_dict(item) for item in value]
                return json_encoder.encode(items)
            if is_dataclass(value) and not isinstance(value, type):
                return json_encoder.encode(dataclass_to_dict(value))
            return json_encoder.encode(value)

        def flatten_cvss(
            cvss_scores: list[CVSSEntry],
        ) -> dict[str, float | str | None]:
            """Flatten list[CVSSEntry] to scalar columns.

            Selection logic:
            1. Prefer source: nvd > redhat > microsoft > other
            2. If tie, take first in list
            """
            if not cvss_scores:
                return {
                    "cvss_base_score": None,
                    "cvss_base_severity": None,
                    "cvss_version": None,
                    "cvss_vector_string": None,
                    "cvss_source": None,
                }

            # Source preference order
            source_priority = {"nvd": 0, "redhat": 1, "microsoft": 2}

            def score_key(entry: CVSSEntry) -> int:
                return source_priority.get(entry.source, 999)

            # Select preferred entry
            selected = min(cvss_scores, key=score_key)

            return {
                "cvss_base_score": selected.base_score,
                "cvss_base_severity": selected.base_severity,
                "cvss_version": selected.version,
                "cvss_vector_string": selected.vector_string,
                "cvss_source": selected.source,
            }

        def flatten_cwe(cwes: list[CWEEntry]) -> dict[str, str | None]:
            """Flatten list[CWEEntry] to scalar columns (first entry)."""
            if not cwes:
                return {
                    "primary_cwe_id": None,
                    "primary_cwe_name": None,
                }
            # Take first CWE
            first = cwes[0]
            return {
                "primary_cwe_id": first.id,
                "primary_cwe_name": first.name,
            }

        def flatten_cpe(cpes: list[CPEEntry]) -> dict[str, str | None]:
            """Flatten list[CPEEntry] to scalar columns (first entry)."""
            if not cpes:
                return {
                    "primary_vendor": None,
                    "primary_product": None,
                    "primary_version": None,
                }
            # Take first CPE
            first = cpes[0]
            return {
                "primary_vendor": first.vendor,
                "primary_product": first.product,
                "primary_version": first.version,
            }

        dataset = cls()
        if not records:
            return dataset

        n = len(records)

        # Pre-allocate arrays for single-pass population
        # Identity
        ids = np.empty(n, dtype=object)
        cve_ids = np.empty(n, dtype=object)
        aliases_arr = np.empty(n, dtype=object)
        # Description
        titles = np.empty(n, dtype=object)
        descriptions = np.empty(n, dtype=object)
        # Metadata
        first_seen = np.empty(n, dtype=object)
        last_updated = np.empty(n, dtype=object)
        sources_arr = np.empty(n, dtype=object)
        # NVD Temporal
        nvd_published = np.empty(n, dtype=object)
        nvd_modified = np.empty(n, dtype=object)
        # Scoring
        cvss_scores_arr = np.empty(n, dtype=object)
        epss_scores_arr = np.empty(n, dtype=object)
        # Flattened CVSS
        cvss_base_score = np.empty(n, dtype=object)
        cvss_base_severity = np.empty(n, dtype=object)
        cvss_version = np.empty(n, dtype=object)
        cvss_vector_string = np.empty(n, dtype=object)
        cvss_source = np.empty(n, dtype=object)
        # Exploitation
        kev_arr = np.empty(n, dtype=object)
        exploits_arr = np.empty(n, dtype=object)
        # Classification
        cwes_arr = np.empty(n, dtype=object)
        cpes_arr = np.empty(n, dtype=object)
        # Flattened CWE/CPE
        primary_cwe_id = np.empty(n, dtype=object)
        primary_cwe_name = np.empty(n, dtype=object)
        primary_vendor = np.empty(n, dtype=object)
        primary_product = np.empty(n, dtype=object)
        primary_version = np.empty(n, dtype=object)
        # References
        references_arr = np.empty(n, dtype=object)
        # Extensible
        tags_arr = np.empty(n, dtype=object)
        metadata_arr = np.empty(n, dtype=object)

        # Single pass over records
        for i, r in enumerate(records):
            # Identity
            ids[i] = str(r.id)
            cve_ids[i] = r.cve_id
            aliases_arr[i] = json.dumps(r.aliases)
            # Description
            titles[i] = r.title
            descriptions[i] = r.description
            # Metadata
            first_seen[i] = r.first_seen_at
            last_updated[i] = r.last_updated_at
            sources_arr[i] = json.dumps(r.sources)
            # NVD Temporal
            nvd_published[i] = r.nvd_published_at
            nvd_modified[i] = r.nvd_modified_at
            # Scoring
            cvss_scores_arr[i] = encode_nested(r.cvss_scores)
            epss_scores_arr[i] = encode_nested(r.epss_scores)
            # Flattened CVSS
            flattened = flatten_cvss(r.cvss_scores)
            cvss_base_score[i] = flattened["cvss_base_score"]
            cvss_base_severity[i] = flattened["cvss_base_severity"]
            cvss_version[i] = flattened["cvss_version"]
            cvss_vector_string[i] = flattened["cvss_vector_string"]
            cvss_source[i] = flattened["cvss_source"]
            # Exploitation
            kev_arr[i] = encode_nested(r.kev)
            exploits_arr[i] = encode_nested(r.exploits)
            # Classification
            cwes_arr[i] = encode_nested(r.cwes)
            cpes_arr[i] = encode_nested(r.cpes)
            # Flattened CWE/CPE
            cwe_flat = flatten_cwe(r.cwes)
            primary_cwe_id[i] = cwe_flat["primary_cwe_id"]
            primary_cwe_name[i] = cwe_flat["primary_cwe_name"]
            cpe_flat = flatten_cpe(r.cpes)
            primary_vendor[i] = cpe_flat["primary_vendor"]
            primary_product[i] = cpe_flat["primary_product"]
            primary_version[i] = cpe_flat["primary_version"]
            # References
            references_arr[i] = encode_nested(r.references)
            # Extensible
            tags_arr[i] = json.dumps(r.tags)
            metadata_arr[i] = json.dumps(r.metadata, default=str)

        dataset._data = {
            # Identity
            "id": ids,
            "cve_id": cve_ids,
            "aliases": aliases_arr,
            # Description
            "title": titles,
            "description": descriptions,
            # Metadata
            "first_seen_at": first_seen,
            "last_updated_at": last_updated,
            "sources": sources_arr,
            # NVD Temporal
            "nvd_published_at": nvd_published,
            "nvd_modified_at": nvd_modified,
            # Scoring (JSON-encoded lists for Parquet round-trip)
            "cvss_scores": cvss_scores_arr,
            "epss_scores": epss_scores_arr,
            # Flattened CVSS (for DataFrame convenience)
            "cvss_base_score": cvss_base_score,
            "cvss_base_severity": cvss_base_severity,
            "cvss_version": cvss_version,
            "cvss_vector_string": cvss_vector_string,
            "cvss_source": cvss_source,
            # Exploitation (JSON-encoded)
            "kev": kev_arr,
            "exploits": exploits_arr,
            # Classification (JSON-encoded lists)
            "cwes": cwes_arr,
            "cpes": cpes_arr,
            # Flattened CWE/CPE (for DataFrame convenience)
            "primary_cwe_id": primary_cwe_id,
            "primary_cwe_name": primary_cwe_name,
            "primary_vendor": primary_vendor,
            "primary_product": primary_product,
            "primary_version": primary_version,
            # References (JSON-encoded list)
            "references": references_arr,
            # Extensible (JSON-encoded dicts)
            "tags": tags_arr,
            "metadata": metadata_arr,
        }

        return dataset

    @classmethod
    def from_nvd(cls, path: str | Path) -> VulnDataset:
        """Load dataset from NVD JSON file.

        Args:
            path: Path to NVD JSON file

        Returns:
            VulnDataset with records from NVD file
        """
        from vulnflow.extractors.nvd import NVDExtractor

        extractor = NVDExtractor(str(path))
        records = extractor.extract()
        return cls._from_records(records)

    def to_pandas(self) -> pd.DataFrame:
        """Export dataset to pandas DataFrame.

        Returns:
            Pandas DataFrame with all columns
        """
        import pandas as pd

        return pd.DataFrame(self._data)

    def to_polars(self) -> pl.DataFrame:
        """Export dataset to polars DataFrame.

        Returns:
            Polars DataFrame with all columns
        """
        import polars as pl

        # Convert numpy arrays to lists for polars compatibility
        data = {k: v.tolist() for k, v in self._data.items()}
        return pl.from_dict(data)

    def to_parquet(self, path: str | Path) -> None:
        """Export dataset to Parquet file.

        Args:
            path: Output file path
        """
        import pyarrow as pa
        import pyarrow.parquet as pq

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to PyArrow table
        table = pa.table(self._data)
        pq.write_table(table, path)

    @classmethod
    def from_parquet(cls, path: str | Path) -> VulnDataset:
        """Load dataset from Parquet file.

        Args:
            path: Path to Parquet file

        Returns:
            VulnDataset with data from Parquet file
        """
        import pyarrow.parquet as pq

        table = pq.read_table(path)

        dataset = cls()
        # Convert PyArrow table to dict of numpy arrays
        dataset._data = {col: table[col].to_numpy() for col in table.column_names}

        return dataset

    def update_sources(self, indices: list[int], source: str) -> None:
        """Update sources list for specified records.

        Args:
            indices: List of record indices to update
            source: Source name to add (e.g., "KEV", "EPSS")
        """
        import json
        from datetime import datetime

        # Make writable copies (Parquet load returns read-only arrays)
        sources_col = self.get_column("sources").copy()
        updated_at_col = self.get_column("last_updated_at").copy()

        now = datetime.now()

        for idx in indices:
            # Deserialize, update, reserialize
            sources_list = json.loads(sources_col[idx])
            if source not in sources_list:
                sources_list.append(source)
            sources_col[idx] = json.dumps(sources_list)

            # Update timestamp
            updated_at_col[idx] = now

        # Update columns with new arrays
        self._data["sources"] = sources_col
        self._data["last_updated_at"] = updated_at_col
