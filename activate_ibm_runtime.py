#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


_PARSE_ENV_FILE_MATH = r"env_map := { key_i -> value_i }"
def _parse_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line_no, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            raise ValueError(f"{path}:{line_no}: expected KEY=VALUE")
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            raise ValueError(f"{path}:{line_no}: empty key is not allowed")
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        values[key] = value
    return values


_REQUIRE_FIELDS_MATH = r"required := {k in keys(env_map) | env_map[k] != ''}"
def _require_nonempty(values: dict[str, str], keys: list[str]) -> dict[str, str]:
    missing = [key for key in keys if not str(values.get(key, "")).strip()]
    if missing:
        names = ", ".join(missing)
        raise ValueError(
            f"Missing required value(s): {names}. Edit .ibm_runtime.env and paste them in."
        )
    return {key: str(values[key]).strip() for key in keys}


_ACTIVATE_ACCOUNT_MATH = r"service := QiskitRuntimeService(channel, token, instance)"
def main() -> int:
    repo_root = Path(__file__).resolve().parent
    env_path = repo_root / ".ibm_runtime.env"
    if not env_path.exists():
        print(f"Missing {env_path}.", file=sys.stderr)
        return 1

    try:
        values = _parse_env_file(env_path)
        required = _require_nonempty(values, ["QISKIT_IBM_TOKEN", "QISKIT_IBM_INSTANCE"])
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    channel = str(values.get("QISKIT_IBM_CHANNEL", "ibm_quantum_platform")).strip() or "ibm_quantum_platform"
    backend_name = str(values.get("IBM_RUNTIME_BACKEND", "")).strip() or None

    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
    except Exception as exc:
        print(
            "qiskit-ibm-runtime is not installed in this environment. "
            "Activate your venv and install it before running this script.",
            file=sys.stderr,
        )
        print(f"Import error: {exc}", file=sys.stderr)
        return 1

    try:
        service = QiskitRuntimeService(
            channel=channel,
            token=required["QISKIT_IBM_TOKEN"],
            instance=required["QISKIT_IBM_INSTANCE"],
        )
        account = service.active_account() or {}
        print("IBM Runtime credentials verified.")
        print(f"channel={account.get('channel', channel)}")
        print(f"instance={required['QISKIT_IBM_INSTANCE']}")
        if backend_name:
            backend = service.backend(backend_name)
            backend_label = getattr(backend, "name", None)
            if callable(backend_label):
                backend_label = backend_label()
            print(f"backend_ok={backend_label or backend_name}")
        else:
            print("backend_ok=skipped")
        return 0
    except Exception as exc:
        print(
            "Failed to verify the IBM Runtime account. "
            "Check the API key, CRN, network access, and backend name if you set one.",
            file=sys.stderr,
        )
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
