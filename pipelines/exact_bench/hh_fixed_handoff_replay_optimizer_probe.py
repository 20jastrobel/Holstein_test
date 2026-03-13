#!/usr/bin/env python3
"""CLI wrapper for fixed-handoff HH replay optimizer probe."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.exact_bench.hh_fixed_handoff_replay_optimizer_probe_workflow import (
    format_compact_summary,
    parse_cli_args,
    run_fixed_handoff_replay_optimizer_probe,
)


def main(argv: list[str] | None = None) -> None:
    cfg = parse_cli_args(argv)
    payload = run_fixed_handoff_replay_optimizer_probe(cfg)
    for line in format_compact_summary(payload):
        print(line)


if __name__ == "__main__":
    main()
