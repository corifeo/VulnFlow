"""Integration tests for NVD API extractor and history enricher.

Uses mocked HTTP to test the full pipeline without real API calls.
"""

from __future__ import annotations

import json
from unittest.mock import patch

from vulnflow.dataset import VulnDataset


@patch("vulnflow.extractors.nvd_api.time.sleep")
@patch("vulnflow.extractors.nvd_api.HttpFetcher")
def test_nvd_api_extractor_pipeline(MockFetcher, _mock_sleep):
    """NVDAPIExtractor works in a VulnPipeline."""
    from vulnflow.extractors.nvd_api import NVDAPIExtractor
    from vulnflow.pipeline import VulnPipeline

    with open("tests/fixtures/nvd_sample.json") as f:
        response = json.load(f)

    instance = MockFetcher.return_value
    instance.fetch_sync.return_value = (json.dumps(response).encode(), "json")

    pipeline = VulnPipeline(
        extractors=[NVDAPIExtractor(keyword="test")],
    )
    result = pipeline.run()

    assert len(result.errors) == 0
    assert result.dataset is not None
    assert len(result.dataset) == 5


@patch("vulnflow.extractors.nvd_api.time.sleep")
@patch("vulnflow.extractors.nvd_api.HttpFetcher")
def test_nvd_api_extractor_to_parquet_roundtrip(MockFetcher, _mock_sleep, tmp_path):
    """NVDAPIExtractor output can be saved and reloaded via Parquet."""
    from vulnflow.extractors.nvd_api import NVDAPIExtractor

    with open("tests/fixtures/nvd_sample.json") as f:
        response = json.load(f)

    instance = MockFetcher.return_value
    instance.fetch_sync.return_value = (json.dumps(response).encode(), "json")

    dataset = NVDAPIExtractor(keyword="test").extract()
    parquet_path = tmp_path / "api_results.parquet"
    dataset.to_parquet(str(parquet_path))

    reloaded = VulnDataset.from_parquet(str(parquet_path))
    assert len(reloaded) == len(dataset)
    assert list(reloaded.get_column("cve_id")) == list(dataset.get_column("cve_id"))
