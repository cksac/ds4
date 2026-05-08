#!/bin/sh
set -e

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$ROOT"

exec cargo run --quiet --bin download-model-rs -- "$@"
