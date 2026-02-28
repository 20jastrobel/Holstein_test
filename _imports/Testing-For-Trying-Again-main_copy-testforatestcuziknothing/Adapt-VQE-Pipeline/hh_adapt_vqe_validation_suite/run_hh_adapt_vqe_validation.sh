#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

ARTIFACT_DIR="$SCRIPT_DIR/artifacts"
mkdir -p "$ARTIFACT_DIR"
export ADAPT_HH_TOL_L2="${ADAPT_HH_TOL_L2:-1e-2}"
export ADAPT_HH_TOL_L3="${ADAPT_HH_TOL_L3:-1e-1}"
export ADAPT_HH_MAXITER_L2="${ADAPT_HH_MAXITER_L2:-1200}"
export ADAPT_HH_MAXITER_L3="${ADAPT_HH_MAXITER_L3:-20}"
export ADAPT_HH_ARTIFACT_DIR="$ARTIFACT_DIR"

python -m pytest -q hh_adapt_vqe_validation_suite/tests/test_hh_adapt_vqe_ground_states.py
