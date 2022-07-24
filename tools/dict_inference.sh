#!/usr/bin/env bash

CONFIG=$1
CHECKPOINT=$2
# GPUS=$3
# PORT=${PORT:-29547}
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python $(dirname "$0")/test.py $CONFIG $CHECKPOINT --show-dir /mnt/vepfs/ML/Users/mazhuang/Monocular-Depth-Estimation-Toolbox/work_dirs/depthformer_swinl_22k_w7_kitti_baseline/depthformer_visualization_baseline
