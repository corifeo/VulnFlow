"""
CVEListExtractor - cvelistV5 records (directory, zip, nested release zip, URL)
Tests: sources, errors, registry, validation flag

Layer: Extract
Dependencies: tests/fixtures/cvelist (8 real cvelistV5 records)
"""

from __future__ import annotations

import io
import json
import sys
import zipfile
from pathlib import Path
from unittest.mock import patch

import pytest

from vulnflow.dataset import VulnDataset
from vulnflow.exceptions import ExtractorError
from vulnflow.extractors.cvelist import CVEListExtractor

CVELIST_DIR = Path(__file__).resolve().parents[2] / "fixtures" / "cvelist"
N_FIXTURES = 8


def _zip_bytes(nested: bool = False) -> bytes:
    """Release-style zip: cves/<year>/<bucket>/CVE-*.json, optionally wrapped."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(CVELIST_DIR.glob("CVE-*.json")):
            year = path.name.split("-")[1]
            zf.writestr(f"cves/{year}/xxx/{path.name}", path.read_bytes())
        zf.writestr("cves/delta.json", b"{}")  # not a CVE record: ignored
    if not nested:
        return buf.getvalue()
    outer = io.BytesIO()
    with zipfile.ZipFile(outer, "w") as zf:
        zf.writestr("cves.zip", buf.getvalue())
    return outer.getvalue()


class TestCVEListExtractorSources:
    def test_directory_returns_all_records(self) -> None:
        ds = CVEListExtractor(str(CVELIST_DIR)).extract()

        assert isinstance(ds, VulnDataset)
        assert len(ds) == N_FIXTURES
        assert "CVE-2025-0168" in set(ds.get_column("cve_id"))
        assert set(ds.get_column("nvd_status")) == {"published", "rejected"}

    def test_zip_file(self, tmp_path: Path) -> None:
        path = tmp_path / "2026-11-13_delta_CVEs_at_1700Z.zip"
        path.write_bytes(_zip_bytes())

        ds = CVEListExtractor(str(path)).extract()

        assert len(ds) == N_FIXTURES
        assert ds.provenance["extractors"] == [
            "CVEListExtractor(2026-11-13_delta_CVEs_at_1700Z.zip)"
        ]

    def test_nested_release_zip(self, tmp_path: Path) -> None:
        path = tmp_path / "2026-11-13_all_CVEs_at_midnight.zip.zip"
        path.write_bytes(_zip_bytes(nested=True))

        ds = CVEListExtractor(path).extract()

        assert len(ds) == N_FIXTURES

    @patch("vulnflow.extractors.cvelist.HttpFetcher")
    def test_url_downloads_zip(self, MockFetcher: object) -> None:
        instance = MockFetcher.return_value  # type: ignore[attr-defined]
        instance.fetch_sync.return_value = (_zip_bytes(nested=True), "binary")
        url = "https://github.com/x/2026-11-13_all_CVEs_at_midnight.zip.zip"

        ds = CVEListExtractor(url).extract()

        assert len(ds) == N_FIXTURES
        assert instance.fetch_sync.call_args.args[0] == url
        assert instance.fetch_sync.call_args.kwargs["decompress"] is False

    def test_timings_recorded(self) -> None:
        ext = CVEListExtractor(str(CVELIST_DIR))
        ext.extract()

        assert {"read", "json", "map"} <= set(ext._timings)

    def test_registered_as_cvelist(self) -> None:
        from vulnflow.registry import Registry

        assert Registry.get_extractor("cvelist") is CVEListExtractor
        assert CVEListExtractor.format == "cvelist"

    def test_empty_directory_returns_empty_dataset(self, tmp_path: Path) -> None:
        assert len(CVEListExtractor(str(tmp_path)).extract()) == 0


class TestCVEListExtractorErrors:
    def test_missing_path_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ExtractorError, match="Failed to read"):
            CVEListExtractor(str(tmp_path / "nope.zip")).extract()

    def test_not_a_zip_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "x.zip"
        path.write_bytes(b"not a zip")

        with pytest.raises(ExtractorError, match="zip"):
            CVEListExtractor(str(path)).extract()

    def test_invalid_json_member_raises(self, tmp_path: Path) -> None:
        (tmp_path / "CVE-2024-0001.json").write_text("{broken")

        with pytest.raises(ExtractorError, match="CVE-2024-0001.json"):
            CVEListExtractor(str(tmp_path)).extract()


class TestCVEListExtractorValidation:
    def test_validate_false_adds_no_columns(self) -> None:
        ds = CVEListExtractor(str(CVELIST_DIR)).extract()

        assert "cve_valid" not in ds.columns

    def test_validate_flags_invalid_record(self) -> None:
        pytest.importorskip("cve_validation_library")
        ext = CVEListExtractor(str(CVELIST_DIR), validate=True)

        ds = ext.extract()

        by_id = dict(zip(ds.get_column("cve_id"), range(len(ds)), strict=True))
        valid = ds.get_column("cve_valid")
        errors = ds.get_column("cve_validation_error")
        bad = by_id["CVE-2022-0353"]  # affected version " " (empty once trimmed)
        assert not valid[bad]
        assert errors[bad] == "FAILED_JSON_SCHEMA_VALIDATION"
        assert json.loads(ds.get_column("metadata")[bad])["validation_message"]
        assert valid.sum() == N_FIXTURES - 1
        assert "validate" in ext._timings

    def test_validate_without_library_raises_import_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(sys.modules, "cve_validation_library", None)

        with pytest.raises(ImportError, match=r"vulnflow\[cvelist\]"):
            CVEListExtractor(str(CVELIST_DIR), validate=True).extract()
