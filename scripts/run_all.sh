#!/usr/bin/env bash
set -e
python -m pip install -r requirements.txt
python -u src/run_pipeline.py
echo "Done. See outputs/figures and outputs/tables"
