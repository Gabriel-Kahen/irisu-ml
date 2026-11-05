#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
build=${1:-"${root}/.tmp/metsu-original-state-proxy"}
mkdir -p "$build"
llvm-dlltool -m i386 -k -d "${root}/tools/metsu/state_proxy_kernel32.def" -l "$build/kernel32.lib"
clang --target=i686-pc-windows-msvc -std=c11 -O2 -Wall -Wextra -Werror -ffreestanding -fno-builtin -fno-stack-protector -c "${root}/tools/metsu/original_state_proxy.c" -o "$build/proxy.obj"
lld-link /machine:x86 /dll /entry:DllMainCRTStartup /nodefaultlib /timestamp:0 /out:"$build/Box2D.dll" "$build/proxy.obj" "$build/kernel32.lib"
python3 "${root}/reference/trace-proxy/validate_build.py" "$build/Box2D.dll"
