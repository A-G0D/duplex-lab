#!/bin/bash
# Re-measure both models over the network, rebuild the table, run tests. Needs numpy.
set -eu
cd "$(dirname "$0")"
PY=${PY:-python3}
mkdir -p data
nice -n 10 "$PY" tools/kvrank.py smollm > data/smol.log 2>&1
nice -n 10 "$PY" tools/kvrank.py moshi > data/moshi.log 2>&1
"$PY" tools/table.py > results/kvrank_table.md
"$PY" -m unittest discover -s tests
