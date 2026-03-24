#!/bin/bash
# Persistent training launcher for YOLOv12
# Waits for COCO dataset to finish downloading before starting training.

SESSION_NAME="yolo_train"
TRAIN_SCRIPT="train_yolov12_paper.py"
VENV_PATH="./venv/bin/python"
COCO_TRAIN_DIR="/home/raid/vishal_kishore/yolov12/datasets/coco/images/train2017"
EXPECTED_IMAGES=118287

# Kill any leftover broken session before starting fresh
if tmux has-session -t $SESSION_NAME 2>/dev/null; then
  echo "Found existing tmux session '$SESSION_NAME'. Killing it first..."
  tmux kill-session -t $SESSION_NAME
fi

# ── Wait for download to complete ──────────────────────────────────────────
echo "Checking COCO dataset..."
while true; do
  if [ -d "$COCO_TRAIN_DIR" ]; then
    IMAGE_COUNT=$(ls "$COCO_TRAIN_DIR" | wc -l)
  else
    IMAGE_COUNT=0
  fi

  if [ "$IMAGE_COUNT" -ge "$EXPECTED_IMAGES" ]; then
    echo "✅ Dataset ready: $IMAGE_COUNT images found. Starting training..."
    break
  else
    echo "⏳ Download in progress: $IMAGE_COUNT / $EXPECTED_IMAGES images found. Checking again in 60s..."
    sleep 60
  fi
done

# ── Launch training ─────────────────────────────────────────────────────────
echo "Starting training in tmux session: $SESSION_NAME"
tmux new-session -d -s $SESSION_NAME "cd $(pwd) && export PYTHONPATH=\$PYTHONPATH:$(pwd); nohup ${VENV_PATH} ${TRAIN_SCRIPT} > training.log 2>&1 & tail -f training.log"

echo "--------------------------------------------------"
echo "Training is running in the background."
echo "Monitor with:      tail -f training.log"
echo "Attach to session: tmux attach -t $SESSION_NAME"
echo "Detach from tmux:  Ctrl+B then D"
echo "--------------------------------------------------"
