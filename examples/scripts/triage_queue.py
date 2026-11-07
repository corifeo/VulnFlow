"""Triage queue export: NVD's Recent feed as a ranked SSVC queue for VulnTriage.

Usage:
    uv run examples/scripts/triage_queue.py [output.json] [--kev-only] [--min-epss 0.01]

Writes a ``vulntriage.queue/v1`` JSON file: KEV CVEs first, then by EPSS and
CVSS, each with suggested SSVC values (exploitation, automatable, technical
impact) and the evidence behind them. Import the file in VulnTriage.
"""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    from vulnflow.helpers import CacheBuilder
    from vulnflow.loaders import TriageLoader

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("output", nargs="?", default="data/triage-queue.json")
    parser.add_argument("--kev-only", action="store_true")
    parser.add_argument("--min-epss", type=float, default=None)
    parser.add_argument("--min-cvss", type=float, default=None)
    parser.add_argument("--limit", type=int, default=500)
    args = parser.parse_args()

    dataset = CacheBuilder(recent_only=True).build()
    meta = TriageLoader(
        args.output,
        limit=args.limit,
        kev_only=args.kev_only,
        min_epss=args.min_epss,
        min_cvss=args.min_cvss,
    ).load(dataset)
    print(
        f"Wrote {meta['record_count']} of {meta['matched']} matching CVEs "
        f"to {meta['path']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
