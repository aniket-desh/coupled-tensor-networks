#!/bin/sh
# The frames of the six HMM chain and tree runs (projective probes), restarted by the amendment of 2026-11-18 19:20 UTC.
#   sh studies/hmm/belief-animation/anim_rest.sh <deadline, unix seconds>
# Their L-BFGS fits are thousands of small GPU operations, and four processes time-slicing one GPU ran them several
# times slower than one process alone. So: all six share the GPU through CUDA MPS if it starts on this worker;
# otherwise they run one after another, each with an equal share of the time left. Either way anim_frames.py keeps the
# frames already saved (--resume), processes checkpoints coarse to fine and starts none after the deadline.
# The two runs that have not trained yet are trained first. At the end the six runs' checkpoints are copied to
# runs/belief-animation/ckpt (published, about 0.5 GB) so that missing frames can be extracted later without retraining.
set -eu
DEADLINE=$1
A=studies/hmm/belief-animation
CKPT=${CKPT:-/tmp/belief-animation/ckpt}
O=runs/belief-animation
PY="env PYTHONPATH=. OMP_NUM_THREADS=1 uv run --frozen python"
S30=$($PY $A/anim_frames.py --steps 30000)
RUNS="rrxor_tree mess3_tree rrxor_chain mess3_chain rrxor_tree8 mess3_tree8"
mkdir -p "$O/logs" "$O/frames" "$O/ckpt" /tmp/mps /tmp/mps-log

export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps CUDA_MPS_LOG_DIRECTORY=/tmp/mps-log
MPS=0
if nvidia-cuda-mps-control -d && timeout 180 $PY -c "import torch; print('cuda under MPS:', torch.zeros(2, device='cuda').sum().item() == 0)"; then
  MPS=1
else
  echo quit | nvidia-cuda-mps-control || true
  unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY
fi
echo "MPS=$MPS"

one() {  # <process>_<model> <deadline>
  name=$1_s0; p=${1%_*}; arm=${1#*_}
  if [ ! -e "$CKPT/$name/result.json" ]; then
    $PY studies/hmm/hmm-belief-geometry/hmm_run.py "$p" "$arm" --ckpts "$S30" --out "$CKPT" > "$O/logs/train_$name.log" 2>&1 || { echo "FAILED train $name"; return 0; }
  fi
  $PY $A/anim_frames.py "$CKPT/$name" "$O/frames" --resume --deadline "$2" > "$O/logs/frames2_$name.log" 2>&1 || { echo "FAILED frames $name"; return 0; }
  echo "DONE $name: $(tail -n 1 "$O/logs/frames2_$name.log")"
}

if [ "$MPS" = 1 ]; then
  for r in $RUNS; do one "$r" "$DEADLINE" & done
  wait
else
  n=6
  for r in $RUNS; do now=$(date +%s); one "$r" $(( now + (DEADLINE - now) / n )); n=$((n - 1)); done
fi
for r in $RUNS; do [ -d "$CKPT/${r}_s0" ] && cp -r "$CKPT/${r}_s0" "$O/ckpt/"; done
if [ "$MPS" = 1 ]; then echo quit | nvidia-cuda-mps-control || true; fi
ls -la "$O/frames"
