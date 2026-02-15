"""MSRCFlattenTransform tests."""

from __future__ import annotations

import json

import numpy as np

from vulnflow.dataset import VulnDataset


def _make_dataset_with_msrc_metadata(
    exploited: bool = True,
    publicly_disclosed: bool = False,
    impact: str = "Remote Code Execution",
    severity_by_product_id: dict[str, str] | None = None,
    month_id: str = "2025-Apr",
) -> VulnDataset:
    """Create a minimal VulnDataset with MSRC metadata."""
    if severity_by_product_id is None:
        severity_by_product_id = {"11568": "Critical", "11569": "Important"}

    ds = VulnDataset()
    ds._data = {
        "cve_id": np.array(["CVE-2025-27745"], dtype=object),
        "metadata": np.array(
            [
                json.dumps(
                    {
                        "msrc": {
                            "month_id": month_id,
                            "title": "Test Vulnerability",
                            "exploited": exploited,
                            "publicly_disclosed": publicly_disclosed,
                            "exploitability": (
                                "Exploitation Detected" if exploited else "Less Likely"
                            ),
                            "impact": impact,
                            "product_ids": list(severity_by_product_id.keys()),
                            "severity_by_product_id": severity_by_product_id,
                            "remediations": [],
                            "acknowledgments": [],
                        }
                    }
                )
            ],
            dtype=object,
        ),
    }
    return ds


class TestMSRCFlattenTransform:
    def test_exploited_true(self) -> None:
        from vulnflow.transforms.flatten.msrc import MSRCFlattenTransform

        ds = _make_dataset_with_msrc_metadata(exploited=True)
        transform = MSRCFlattenTransform()
        transform.transform(ds)

        assert ds.get_column("msrc_exploited")[0] is True

    def test_exploited_false(self) -> None:
        from vulnflow.transforms.flatten.msrc import MSRCFlattenTransform

        ds = _make_dataset_with_msrc_metadata(exploited=False)
        transform = MSRCFlattenTransform()
        transform.transform(ds)

        assert ds.get_column("msrc_exploited")[0] is False

    def test_publicly_disclosed(self) -> None:
        from vulnflow.transforms.flatten.msrc import MSRCFlattenTransform

        ds = _make_dataset_with_msrc_metadata(publicly_disclosed=True)
        transform = MSRCFlattenTransform()
        transform.transform(ds)

        assert ds.get_column("msrc_publicly_disclosed")[0] is True

    def test_impact(self) -> None:
        from vulnflow.transforms.flatten.msrc import MSRCFlattenTransform

        ds = _make_dataset_with_msrc_metadata(impact="Elevation of Privilege")
        transform = MSRCFlattenTransform()
        transform.transform(ds)

        assert ds.get_column("msrc_impact")[0] == "Elevation of Privilege"

    def test_max_severity_critical(self) -> None:
        from vulnflow.transforms.flatten.msrc import MSRCFlattenTransform

        ds = _make_dataset_with_msrc_metadata(
            severity_by_product_id={"11568": "Critical", "11569": "Important"}
        )
        transform = MSRCFlattenTransform()
        transform.transform(ds)

        assert ds.get_column("msrc_max_severity")[0] == "Critical"

    def test_max_severity_important_only(self) -> None:
        from vulnflow.transforms.flatten.msrc import MSRCFlattenTransform

        ds = _make_dataset_with_msrc_metadata(
            severity_by_product_id={"11568": "Important", "11569": "Moderate"}
        )
        transform = MSRCFlattenTransform()
        transform.transform(ds)

        assert ds.get_column("msrc_max_severity")[0] == "Important"

    def test_month_id(self) -> None:
        from vulnflow.transforms.flatten.msrc import MSRCFlattenTransform

        ds = _make_dataset_with_msrc_metadata(month_id="2025-Apr")
        transform = MSRCFlattenTransform()
        transform.transform(ds)

        assert ds.get_column("msrc_month_id")[0] == "2025-Apr"

    def test_no_msrc_metadata(self) -> None:
        """Records without MSRC metadata get None/False defaults."""
        from vulnflow.transforms.flatten.msrc import MSRCFlattenTransform

        ds = VulnDataset()
        ds._data = {
            "cve_id": np.array(["CVE-2024-0001"], dtype=object),
            "metadata": np.array(["{}"], dtype=object),
        }
        transform = MSRCFlattenTransform()
        transform.transform(ds)

        assert ds.get_column("msrc_exploited")[0] is False
        assert ds.get_column("msrc_publicly_disclosed")[0] is False
        assert ds.get_column("msrc_impact")[0] is None
        assert ds.get_column("msrc_max_severity")[0] is None
        assert ds.get_column("msrc_month_id")[0] is None

    def test_output_schema_has_five_fields(self) -> None:
        from vulnflow.transforms.flatten.msrc import MSRCFlattenTransform

        transform = MSRCFlattenTransform()
        assert len(transform.OUTPUT_SCHEMA) == 5

    def test_columns_inserted_after_metadata(self) -> None:
        from vulnflow.transforms.flatten.msrc import MSRCFlattenTransform

        ds = _make_dataset_with_msrc_metadata()
        transform = MSRCFlattenTransform()
        transform.transform(ds)

        cols = ds.columns
        meta_idx = cols.index("metadata")
        assert cols[meta_idx + 1] == "msrc_exploited"

    def test_registered_as_default(self) -> None:
        """MSRCFlattenTransform is registered as default=False."""
        import importlib

        import vulnflow.transforms.flatten.msrc

        importlib.reload(vulnflow.transforms.flatten.msrc)

        from vulnflow.registry import Registry

        entry = Registry._transforms.get("MSRCFlattenTransform")
        assert entry is not None
        _cls, is_default = entry
        assert is_default is False
