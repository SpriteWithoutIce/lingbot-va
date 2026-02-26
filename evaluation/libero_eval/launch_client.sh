#!/bin/bash
# Run LIBERO evaluation client (requires server from launch_server.sh).
# Usage:
#   ./launch_client.sh                    # default: libero_spatial, port 29536
#   ./launch_client.sh libero_object 29536

cd "$(dirname "$0")/../.."

TASK_SUITE=${1:-libero_spatial}
PORT=${2:-29536}
REPLAN_STEPS=${REPLAN_STEPS:-10}
VIDEO_OUT=${VIDEO_OUT:-data/libero/videos}

# LIBERO env (conda with libero installed) must be active for client
python -m evaluation.libero_eval.run_libero_eval \
    --host 127.0.0.1 \
    --port "$PORT" \
    --task_suite_name "$TASK_SUITE" \
    --video_out_path "$VIDEO_OUT"
