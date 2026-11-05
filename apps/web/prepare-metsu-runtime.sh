#!/usr/bin/env bash
set -euo pipefail
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
exec python3 "$here/prepare-metsu-runtime.py" "$@"
