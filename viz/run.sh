#!/usr/bin/env bash
# Start the visualizer on 127.0.0.1 only; reach it with: ssh -L 8501:localhost:8501 pf-pc37.ethz.ch
set -e
source /scratch2/oscipal/cropyield/bin/activate
cd "$(dirname "$0")"
[ -f /scratch2/oscipal/yieldsat_work/viz/field_shapes_display.parquet ] || python build_index.py
exec streamlit run app.py --server.address 127.0.0.1 --server.port "${PORT:-8501}" --server.headless true \
    --browser.gatherUsageStats false "$@"
