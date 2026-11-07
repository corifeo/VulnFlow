"""
README - The front page's code and links work
Tests: Every vulnflow import in a code block runs, links resolve, no stale claims

Layer: Integration Test
Dependencies: All layers (imports only; the full blocks need the network)
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

README = Path("README.md").read_text(encoding="utf8")
BLOCKS = re.findall(r"```python\n(.*?)```", README, re.S)


def test_readme_python_imports_run() -> None:
    # Keep README imports single-line so this can run them on their own
    lines = [
        line
        for block in BLOCKS
        for line in block.splitlines()
        if line.startswith(("from vulnflow", "import vulnflow"))
    ]
    assert lines
    for line in lines:
        exec(line, {})


def test_readme_pipeline_is_the_tested_chain() -> None:
    # The chain tests/integration/test_log4shell.py verifies end to end
    assert "KEVFlattenTransform()" in README
    assert "EventInference()" in README


def test_readme_links_point_at_real_files() -> None:
    pattern = r"\]\(((?:examples|scripts|\.github|tests|LICENSE|CITATION|CONTRIBUTING|SECURITY)[^)#]*)\)"
    links = re.findall(pattern, README) + re.findall(
        r'(?:src|srcset)="(\.github/[^"]+)"', README
    )
    missing = [link for link in links if not Path(link).exists()]
    assert missing == []
    assert "docs/" not in README


@pytest.mark.parametrize("claim", ["OSV", "SQL", "uv add vulnflow", "Mozilla"])
def test_readme_has_no_stale_claims(claim: str) -> None:
    assert claim not in README
