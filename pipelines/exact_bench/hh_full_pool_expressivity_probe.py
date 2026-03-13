#!/usr/bin/env python3
"""CLI wrapper for HH full-pool expressivity probe."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.exact_bench.hh_full_pool_expressivity_probe_workflow import (
    format_compact_summary,
    parse_cli_args,
    run_full_pool_expressivity_probe,
)


def main(argv: list[str] | None = None) -> None:
    cfg = parse_cli_args(argv)
    payload = run_full_pool_expressivity_probe(cfg)
    for line in format_compact_summary(payload):
        print(line)


if __name__ == "__main__":
    main()
