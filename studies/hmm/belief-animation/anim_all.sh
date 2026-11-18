#!/bin/sh
# All 13 runs of the animation, $JOBS at a time on one GPU, longest first.
#   EVERY_TREE32=1 JOBS=4 sh studies/hmm/belief-animation/anim_all.sh
# EVERY_TREE32 = k keeps every k-th checkpoint of the two r = 32 HMM trees in the frame step (their product-feature
# projective fits are the slow part); training and every other run are unaffected.
set -eu
A=studies/hmm/belief-animation
PY="env PYTHONPATH=. OMP_NUM_THREADS=2 uv run --frozen python"
S30=$(PYTHONPATH=. uv run --frozen python $A/anim_frames.py --steps 30000)
S50=$(PYTHONPATH=. uv run --frozen python $A/anim_frames.py --steps 50000)
E=${EVERY_TREE32:-1}
HMM=studies/hmm/hmm-belief-geometry/hmm_run.py
ARCH=studies/rhm/rhm-belief-architectures/arch_run.py
G=v8m2s2L4
{
  echo "rrxor_tree_s0 $E $PY $HMM rrxor tree --ckpts $S30"
  echo "mess3_tree_s0 $E $PY $HMM mess3 tree --ckpts $S30"
  echo "${G}_s0 1 $PY studies/rhm/rhm-belief-geometry/run.py $G --ckpts $S50"
  for p in mess3 rrxor; do echo "${p}_transformer_s0 1 $PY $HMM $p transformer --ckpts $S30"; done
  for arm in tree perm chain; do echo "${G}_${arm}_s0 1 $PY $ARCH $G --arm $arm --ckpts $S30"; done
  echo "${G}_r16_s0 1 $PY studies/rhm/rhm-ttn-belief-geometry/ttn_run.py $G --r 16 --ckpts $S30"
  for p in mess3 rrxor; do for arm in chain tree8; do echo "${p}_${arm}_s0 1 $PY $HMM $p $arm --ckpts $S30"; done; done
} | xargs -P "${JOBS:-4}" -I{} sh -c "sh $A/anim_job.sh {}"
