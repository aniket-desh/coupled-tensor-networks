#!/bin/sh
# One run of the animation: train with dense checkpoints, then extract its frames.
#   sh studies/hmm/belief-animation/anim_job.sh <run name> <every> <training command ...>
# Checkpoints go to $CKPT (worker-local, never published: about 0.3 GB per run); logs and frames to runs/belief-animation.
# A run is skipped if runs/belief-animation/STOP exists when its turn comes; a finished training is not repeated.
set -eu
name=$1; every=$2; shift 2
CKPT=${CKPT:-/tmp/belief-animation/ckpt}
O=runs/belief-animation
mkdir -p "$CKPT" "$O/logs" "$O/frames"
if [ -e "$O/STOP" ]; then echo "SKIPPED $name"; exit 0; fi
if [ ! -e "$CKPT/$name/result.json" ]; then
  "$@" --out "$CKPT" > "$O/logs/train_$name.log" 2>&1 || { echo "FAILED train $name"; exit 0; }
fi
PYTHONPATH=. uv run --frozen python studies/hmm/belief-animation/anim_frames.py "$CKPT/$name" "$O/frames" --every "$every" \
  > "$O/logs/frames_$name.log" 2>&1 || { echo "FAILED frames $name"; exit 0; }
echo "DONE $name"
