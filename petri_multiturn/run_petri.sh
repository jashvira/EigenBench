#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

exec "${INSPECT_BIN:-$ROOT/.venv/bin/inspect}" eval petri_multiturn/run_petri.py@criterion_audit "$@"
