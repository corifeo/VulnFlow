# Contributing to vulnflow

Issues and pull requests are welcome, especially new extractors and enrichers.

## Setup

vulnflow uses [uv](https://docs.astral.sh/uv/) and needs Python 3.11 or later.

```bash
git clone https://github.com/corifeo/vulnflow
cd vulnflow
uv sync
```

## Checks

CI runs these on every pull request, and they should all pass locally first:

```bash
uv run pytest
uv run ruff check src/ tests/
uv run mypy src/ --strict
uv run black --check src/ tests/
```

`uv run pytest` skips tests marked `slow`. Run them with `uv run pytest -m slow`.

## Pull requests

- Keep each PR to one change, and add tests for it.
- Add a line under `## Unreleased` in `CHANGELOG.md` for any change a user would notice.
- A new extractor, enricher or loader is one small class. Copy the shape of an existing one in `src/vulnflow/extractors/`, `src/vulnflow/transforms/enrichers/` or `src/vulnflow/loaders/`, and export it from that package's `__init__.py`.
- Tests must not call the real NVD, KEV, EPSS or MSRC services. Use the fixtures in `tests/fixtures/`.

## Security issues

Don't open a public issue. See [`SECURITY.md`](SECURITY.md).
