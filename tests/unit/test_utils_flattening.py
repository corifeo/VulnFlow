"""Tests for flattening utilities.

Tests cover:
- decode_json_column(): JSON decoding with empty lists, nulls, valid JSON
- flatten_cvss_scores(): CVSS flattening with NVD source priority
- flatten_cwes(): CWE flattening (first entry)
- flatten_cpes(): CPE flattening (first entry)
"""

from __future__ import annotations

import numpy as np

from vulnflow.utils.flattening import (
    decode_json_column,
    flatten_cpes,
    flatten_cvss_scores,
    flatten_cwes,
)


class TestDecodeJsonColumn:
    """Tests for decode_json_column()."""

    def test_empty_array(self) -> None:
        """Empty array returns empty list."""
        col = np.array([], dtype=object)
        result = decode_json_column(col)
        assert result == []

    def test_null_values(self) -> None:
        """Null values decode to None."""
        col = np.array(["null", None, ""], dtype=object)
        result = decode_json_column(col)
        assert result == [None, None, None]

    def test_empty_lists(self) -> None:
        """Empty JSON lists decode correctly."""
        col = np.array(["[]", "[]"], dtype=object)
        result = decode_json_column(col)
        assert result == [[], []]

    def test_valid_json(self) -> None:
        """Valid JSON strings decode to Python objects."""
        col = np.array(
            ['[{"a": 1}]', '[{"b": 2}, {"c": 3}]', "[]"], dtype=object
        )
        result = decode_json_column(col)
        assert result == [[{"a": 1}], [{"b": 2}, {"c": 3}], []]


class TestFlattenCvssScores:
    """Tests for flatten_cvss_scores()."""

    def test_none_input(self) -> None:
        """None input returns all None fields."""
        result = flatten_cvss_scores(None)
        assert result == {
            "cvss_version": None,
            "cvss_vector": None,
            "cvss_base_score": None,
            "cvss_source": None,
        }

    def test_empty_list(self) -> None:
        """Empty list returns all None fields."""
        result = flatten_cvss_scores([])
        assert result == {
            "cvss_version": None,
            "cvss_vector": None,
            "cvss_base_score": None,
            "cvss_source": None,
        }

    def test_single_entry(self) -> None:
        """Single entry extracts correctly."""
        cvss_list = [
            {
                "version": "3.1",
                "vector": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H",
                "base_score": 9.8,
                "source": "nvd",
            }
        ]
        result = flatten_cvss_scores(cvss_list)
        assert result == {
            "cvss_version": "3.1",
            "cvss_vector": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H",
            "cvss_base_score": 9.8,
            "cvss_source": "nvd",
        }

    def test_nvd_priority(self) -> None:
        """NVD source has priority over others."""
        cvss_list = [
            {
                "version": "3.0",
                "vector": "CVSS:3.0/AV:L/AC:L/PR:N/UI:N/S:U/C:L/I:L/A:L",
                "base_score": 6.3,
                "source": "redhat",
            },
            {
                "version": "3.1",
                "vector": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H",
                "base_score": 9.8,
                "source": "nvd",
            },
            {
                "version": "3.1",
                "vector": "CVSS:3.1/AV:N/AC:H/PR:N/UI:N/S:U/C:H/I:H/A:H",
                "base_score": 8.1,
                "source": "microsoft",
            },
        ]
        result = flatten_cvss_scores(cvss_list)
        # Should select NVD entry (9.8) not first entry (6.3)
        assert result == {
            "cvss_version": "3.1",
            "cvss_vector": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H",
            "cvss_base_score": 9.8,
            "cvss_source": "nvd",
        }

    def test_source_priority_order(self) -> None:
        """Source priority: nvd > redhat > microsoft > unknown."""
        cvss_list = [
            {
                "version": "3.1",
                "vector": "CVSS:3.1/AV:L/AC:L/PR:N/UI:N/S:U/C:L/I:L/A:L",
                "base_score": 6.3,
                "source": "unknown_vendor",
            },
            {
                "version": "3.0",
                "vector": "CVSS:3.0/AV:N/AC:H/PR:N/UI:N/S:U/C:H/I:H/A:H",
                "base_score": 8.1,
                "source": "microsoft",
            },
            {
                "version": "3.1",
                "vector": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H",
                "base_score": 9.8,
                "source": "redhat",
            },
        ]
        result = flatten_cvss_scores(cvss_list)
        # Should select redhat (priority 1) over microsoft (priority 2)
        assert result == {
            "cvss_version": "3.1",
            "cvss_vector": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H",
            "cvss_base_score": 9.8,
            "cvss_source": "redhat",
        }


class TestFlattenCwes:
    """Tests for flatten_cwes()."""

    def test_none_input(self) -> None:
        """None input returns all None fields."""
        result = flatten_cwes(None)
        assert result == {"cwe_id": None, "cwe_name": None}

    def test_empty_list(self) -> None:
        """Empty list returns all None fields."""
        result = flatten_cwes([])
        assert result == {"cwe_id": None, "cwe_name": None}

    def test_single_entry(self) -> None:
        """Single entry extracts correctly."""
        cwe_list = [{"id": "CWE-79", "name": "Cross-site Scripting (XSS)"}]
        result = flatten_cwes(cwe_list)
        assert result == {
            "cwe_id": "CWE-79",
            "cwe_name": "Cross-site Scripting (XSS)",
        }

    def test_multiple_entries_takes_first(self) -> None:
        """Multiple entries take first (primary CWE)."""
        cwe_list = [
            {"id": "CWE-79", "name": "Cross-site Scripting (XSS)"},
            {"id": "CWE-20", "name": "Improper Input Validation"},
        ]
        result = flatten_cwes(cwe_list)
        assert result == {
            "cwe_id": "CWE-79",
            "cwe_name": "Cross-site Scripting (XSS)",
        }


class TestFlattenCpes:
    """Tests for flatten_cpes()."""

    def test_none_input(self) -> None:
        """None input returns all None fields."""
        result = flatten_cpes(None)
        assert result == {
            "cpe_uri": None,
            "cpe_vendor": None,
            "cpe_product": None,
        }

    def test_empty_list(self) -> None:
        """Empty list returns all None fields."""
        result = flatten_cpes([])
        assert result == {
            "cpe_uri": None,
            "cpe_vendor": None,
            "cpe_product": None,
        }

    def test_single_entry(self) -> None:
        """Single entry extracts correctly."""
        cpe_list = [
            {
                "uri": "cpe:2.3:a:apache:log4j:2.14.1:*:*:*:*:*:*:*",
                "vendor": "apache",
                "product": "log4j",
            }
        ]
        result = flatten_cpes(cpe_list)
        assert result == {
            "cpe_uri": "cpe:2.3:a:apache:log4j:2.14.1:*:*:*:*:*:*:*",
            "cpe_vendor": "apache",
            "cpe_product": "log4j",
        }

    def test_multiple_entries_takes_first(self) -> None:
        """Multiple entries take first (primary CPE)."""
        cpe_list = [
            {
                "uri": "cpe:2.3:a:apache:log4j:2.14.1:*:*:*:*:*:*:*",
                "vendor": "apache",
                "product": "log4j",
            },
            {
                "uri": "cpe:2.3:a:apache:log4j:2.15.0:*:*:*:*:*:*:*",
                "vendor": "apache",
                "product": "log4j",
            },
        ]
        result = flatten_cpes(cpe_list)
        assert result == {
            "cpe_uri": "cpe:2.3:a:apache:log4j:2.14.1:*:*:*:*:*:*:*",
            "cpe_vendor": "apache",
            "cpe_product": "log4j",
        }
