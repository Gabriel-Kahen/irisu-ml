#!/usr/bin/env bash
set -euo pipefail
root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
output=$(realpath -m -- "${1:-$root/build-metsu-web/site/metsu}")
runtime=${IRISU_METSU_RUNTIME_DIR:?set IRISU_METSU_RUNTIME_DIR to a verified Metsu runtime}
runtime=$(realpath -m -- "$runtime")
static="$root/apps/web/metsu-static"
normal="$root/apps/web/static"
for protected in / "$root" "$static" "$normal" "$runtime"; do
  if [[ "$output" == "$protected" || "$protected/" == "$output/"* ||
        ( "$protected" != / && "$protected" != "$root" && "$output/" == "$protected/"* ) ]]; then
    echo 'refusing unsafe or overlapping Metsu static-build path' >&2
    exit 1
  fi
done
[[ -f "$runtime/.irisu-metsu-runtime" ]] || { echo 'Metsu runtime is not verified' >&2; exit 1; }
(cd "$runtime" && sha256sum -c SHA256SUMS >/dev/null)
if [[ -e "$output" && ! -f "$output/.irisu-metsu-static-site" ]]; then
  echo 'refusing to replace an unmarked Metsu static site' >&2
  exit 1
fi
mkdir -p "$(dirname -- "$output")"
stage=$(mktemp -d "${output}.stage.XXXXXX")
trap 'rm -rf -- "$stage"' EXIT
cp -a "$static/." "$stage/"
cp -a "$runtime" "$stage/exact-runtime"
python3 - "$runtime/metsu-runtime.json" "$stage/mode-identity.js" <<'PY'
import json,sys
from pathlib import Path
identity=json.loads(Path(sys.argv[1]).read_text())
if (identity.get('mode') != 1 or not identity.get('prepared') or
        identity.get('backend') != 'exact-msvc9-r58-metsu-forward' or
        identity.get('config_hash') in (None,'0','17009678407634462320')):
    raise SystemExit('runtime does not identify verified Metsu mode 1')
Path(sys.argv[2]).write_text('globalThis.IRISU_METSU_IDENTITY = Object.freeze('+json.dumps(identity,sort_keys=True)+');\n')
PY
printf '%s\n' 'generated isolated Metsu site; safe to replace' > "$stage/.irisu-metsu-static-site"
[[ ! -e "$output" ]] || rm -rf -- "$output"
mv "$stage" "$output"
trap - EXIT
echo "built isolated Metsu client at $output (serve at /metsu/)"
