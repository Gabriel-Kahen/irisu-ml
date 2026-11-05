#!/usr/bin/env bash
set -euo pipefail
root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
output=$(realpath -m -- "${1:-$root/apps/web/dist}")
normal_runtime=$(realpath -m -- "${IRISU_EXACT_RUNTIME_DIR:?set IRISU_EXACT_RUNTIME_DIR to the prepared normal runtime}")
metsu_runtime=$(realpath -m -- "${IRISU_METSU_RUNTIME_DIR:?set IRISU_METSU_RUNTIME_DIR to the verified Metsu runtime}")
if [[ "$output" == / || "$output" == "$root" || "$root/" == "$output/"* ]]; then
  echo 'refusing unsafe combined-site path' >&2
  exit 1
fi
for protected in "$root/apps/web/static" "$root/apps/web/metsu-static" "$normal_runtime" "$metsu_runtime"; do
  if [[ "$output" == "$protected" || "$output/" == "$protected/"* || "$protected/" == "$output/"* ]]; then
    echo 'refusing overlapping combined-site path' >&2
    exit 1
  fi
done
if [[ -e "$output" && ! -f "$output/.irisu-static-site" ]]; then
  echo 'refusing to replace unmarked static-site path' >&2
  exit 1
fi
mkdir -p "$(dirname -- "$output")"
stage=$(mktemp -d "${output}.stage.XXXXXX")
trap 'rm -rf -- "$stage"' EXIT
IRISU_EXACT_RUNTIME_DIR="$normal_runtime" "$root/apps/web/build-static.sh" "$stage/site"
IRISU_METSU_RUNTIME_DIR="$metsu_runtime" "$root/apps/web/build-metsu-static.sh" "$stage/site/metsu"
python3 - "$stage/site" <<'PY'
from pathlib import Path
import sys
site = Path(sys.argv[1])
normal = site / 'normal'
normal.mkdir()
# Both normal entry points share the same scripts, worker, runtime and cache.
html = (site / 'index.html').read_text()
(normal / 'index.html').write_text(html.replace('<head>', '<head>\n  <base href="../">', 1))
PY
[[ ! -e "$output" ]] || rm -rf -- "$output"
mv "$stage/site" "$output"
echo "built normal (/ and /normal/) and Metsu (/metsu/) at $output"
