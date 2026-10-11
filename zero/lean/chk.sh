#!/bin/bash
export PATH=$PATH:~/.elan/bin
cd "$(dirname "$0")"
lake env lean "$@" 2>&1 | head -${LINES_MAX:-60}
