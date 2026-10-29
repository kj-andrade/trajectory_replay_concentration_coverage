#!/bin/bash
# Full closed-form run: all three experiments, then figures and tables.
# Set PYTHON to use a specific interpreter, e.g. PYTHON=.venv/bin/python.
cd "$(dirname "$0")"
PY="${PYTHON:-python}"
export PYTHONIOENCODING=utf-8
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
set -x
"$PY" -u -m scripts.run --exp onset      --config configs/final.yaml || exit 1
"$PY" -u -m scripts.run --exp coverage   --config configs/final.yaml || exit 1
"$PY" -u -m scripts.run --exp anisotropy --config configs/final.yaml || exit 1
"$PY" -u -m scripts.plot --all --results results --out figures || exit 1
