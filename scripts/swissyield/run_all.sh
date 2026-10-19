#!/usr/bin/env bash
# Run the whole SwissYield analysis; outputs in <work_dir>/swissyield (profile.md, tables/, figures/).
set -euo pipefail
source /scratch2/oscipal/cropyield/bin/activate
cd "$(dirname "$0")"
for s in 00_file_tree 01_inventory 04_raw_profile 02_target_hist 03_feature_hist 05_timeseries_relations 99_assemble_report; do
  echo "=== $s"; python "$s.py"
done
