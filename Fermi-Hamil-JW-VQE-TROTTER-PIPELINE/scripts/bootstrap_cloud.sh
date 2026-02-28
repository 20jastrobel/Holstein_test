#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBREPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${SUBREPO_ROOT}/.." && pwd)"

VENV_DIR="${1:-${REPO_ROOT}/.venv}"

if [[ -z "${VENV_DIR}" ]]; then
  echo "ERROR: VENV_DIR is empty" >&2
  exit 2
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  else
    echo "ERROR: python not found in PATH" >&2
    exit 2
  fi
fi

if [[ ! -d "${VENV_DIR}" ]]; then
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"

CORE_REQ="${REPO_ROOT}/dependencies/requirements-core.txt"
FULL_REQ="${REPO_ROOT}/dependencies/requirements.txt"
PIP="${VENV_DIR}/bin/pip"

${PIP} install --upgrade pip setuptools wheel
${PIP} install -r "${CORE_REQ}"

if ${PIP} install "matplotlib"; then
  echo "Optional dependency matplotlib installed."
else
  echo "Warning: matplotlib installation failed (proxy/network restriction)."
  echo "Cloud runs will continue with JSON-first outputs."
  echo "Install matplotlib later if available and needed for richer plotting."
fi

echo "Cloud environment ready. Activate with:"
echo "  source ${VENV_DIR}/bin/activate"

if [ -f "${FULL_REQ}" ]; then
  echo "Reference requirements file: ${FULL_REQ}"
fi
