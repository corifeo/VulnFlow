"""
Example notebooks - Published notebooks are executed, error-free and private
Tests: Every code cell ran, no error outputs, no local paths, all listed in README

Layer: Integration Test
Dependencies: None (reads notebook JSON; no Jupyter needed)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

NOTEBOOKS = sorted(Path("examples").glob("*.ipynb"))


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_is_executed_and_clean(path: Path) -> None:
    text = path.read_text(encoding="utf8")
    cells = [c for c in json.loads(text)["cells"] if c["cell_type"] == "code"]
    assert all(c["execution_count"] is not None for c in cells)
    assert not any(o["output_type"] == "error" for c in cells for o in c["outputs"])
    assert "\\\\Users\\\\" not in text and "/Users/" not in text and "/home/" not in text


def test_examples_readme_lists_every_notebook() -> None:
    readme = Path("examples/README.md").read_text(encoding="utf8")
    assert NOTEBOOKS
    assert [p.name for p in NOTEBOOKS if p.name not in readme] == []
