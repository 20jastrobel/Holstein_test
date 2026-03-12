#!/usr/bin/env python3
"""Thin CLI wrapper for the HH L=2 logical robustness screen."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.exact_bench.hh_l2_logical_screen_workflow import (
    format_compact_summary,
    parse_cli_args,
    run_hh_l2_logical_screen,
)


def main(argv: list[str] | None = None) -> None:
    workflow_cfg = parse_cli_args(argv)
    payload = run_hh_l2_logical_screen(workflow_cfg)
    for line in format_compact_summary(payload):
        print(line)


if __name__ == "__main__":
    main()
