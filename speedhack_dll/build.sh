#!/bin/sh
# Rebuild both DLLs (needs zig: pip install ziglang)
cd "$(dirname "$0")"
python3 -m ziglang cc -target x86-windows-gnu    -shared -O2 -s -o dqn_speedhack32.dll speedhack.c
python3 -m ziglang cc -target x86_64-windows-gnu -shared -O2 -s -o dqn_speedhack64.dll speedhack.c
