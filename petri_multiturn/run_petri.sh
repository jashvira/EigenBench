#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

exec "${PYTHON_BIN:-$ROOT/.venv/bin/python}" petri_multiturn/run_petri.py "$@"
