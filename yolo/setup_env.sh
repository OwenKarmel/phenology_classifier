#!/usr/bin/env bash
# Section 3 of Train_YOLO_Models.ipynb: install Ultralytics (with a CUDA build of
# PyTorch) and pull the YOLO26 weights.
#
#     bash yolo/setup_env.sh
#
# /home is nearly full, so the virtualenv, pip's cache and temp files go to
# $YOLO_HOME (default /data/owen/phenology_classifier/yolo) and yolo/.venv is a
# link to it. Safe to re-run.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
YOLO_HOME="${YOLO_HOME:-/data/owen/phenology_classifier/yolo}"
VENV="$YOLO_HOME/venv"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu126}"

mkdir -p "$YOLO_HOME/tmp" "$YOLO_HOME/pip-cache"
export TMPDIR="$YOLO_HOME/tmp" PIP_CACHE_DIR="$YOLO_HOME/pip-cache"

if [ ! -x "$VENV/bin/python" ]; then
    python3 -m venv "$VENV"
fi
"$VENV/bin/pip" install --upgrade pip
# torch first, from the CUDA 12.6 index, so ultralytics finds it already installed.
"$VENV/bin/pip" install torch==2.14.1 torchvision==0.29.1 --index-url "$TORCH_INDEX"
"$VENV/bin/pip" install -r "$HERE/requirements.txt"
ln -sfn "$VENV" "$HERE/.venv"

"$VENV/bin/python" - <<'EOF'
import torch
print(f"torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
for i in range(torch.cuda.device_count()):
    print(f"  cuda:{i} {torch.cuda.get_device_name(i)}")
EOF

# Download the YOLO26 checkpoint(s) named in the config.
"$VENV/bin/python" "$HERE/pipeline.py" pull
