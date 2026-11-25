#!/bin/sh
# Every command behind the results in this folder, in order. Run from the repository root:
#   sh studies/rhm/child_feature_enabling/run_all.sh
# RUNS is the unpacked rhm-dynamics checkpoint archive (runs.tar of job 20261112-214354-rhm-dynamics-blghb2am5xkc6n,
# pub-001, on the results volume). The intervention pairs are independent and can be run as parallel processes.
set -e
PY=${PY:-python}
RUNS=${RUNS:-runs/rhm-dynamics}
M=studies.rhm.child_feature_enabling
R=studies/rhm/child_feature_enabling/results
PAIRS="0:0 0:1 0:2 1:0 1:1 1:2 2:0 2:1 2:2"

# Existing checks (toy_check and grammar_audit rewrite their JSON; values agree with the committed ones to roundoff).
$PY -m studies.rhm.checks
$PY -m $M.population --check
$PY -m $M.losses --check
$PY -m $M.toy_check
$PY -m $M.grammar_audit

# Section 1: planted four-arm sweep.
$PY -m $M.planted
# Section 2: checkpoint readout.
$PY -m $M.checkpoints --runs $RUNS
# Section 3: the intervention. Protocol as written, then amendments A3 and A4 of EXPERIMENT.md.
$PY -m $M.intervene --check
$PY -m $M.intervene --pilot --runs $RUNS
$PY -m $M.intervene --pairs $PAIRS --runs $RUNS
$PY -m $M.intervene --pilot --rates 0.01 0.03 0.1 0.3 1.0 --pilot-steps 4000 --runs $RUNS --out $R/intervention-extended
$PY -m $M.intervene --pairs $PAIRS --steps 40000 --stop-fit 0.9 --runs $RUNS --out $R/intervention-extended
$PY -m $M.intervene --pairs 0:0 1:0 2:0 --steps 40000 --stop-fit 0.9 --lr 0.1 --runs $RUNS --out $R/intervention-extended-lr0.1

$PY -m $M.plot
$PY -m $M.summary

# PNG renders of the figures and the interactive page for the draft post in notes/child-feature-enabling/.
for f in intervention-extended intervention-ceilings planted-time-laws checkpoint-readout intervention-as-written intervention-extended-lr0.1; do
  pdftocairo -png -r 200 -singlefile studies/rhm/child_feature_enabling/figures/$f.pdf notes/child-feature-enabling/zulip/$f
done
# Needs the final intervention weights in runs/child-feature-enabling/ and the step-0 checkpoints in $RUNS.
$PY notes/child-feature-enabling/explorer/export.py
# Optional browser check (needs Playwright): $PY notes/child-feature-enabling/explorer/check_site.py
