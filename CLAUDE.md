# coupled-tensor-networks

Shared code is in `ctn/` (numpy tensor networks and flows; `ctn/ttn.py` is the torch eight-leaf TTN trained by
backprop). Each study lives in `studies/<name>/` with its model, checks, experiments, plots, results and PDF figures.
Run scripts from the repository root (`python -m studies.<study>.<script>`, or `PYTHONPATH=. python <path>`).

## GPU experiments

Experiments run on temporary Runpod GPU workers through the runpod-research toolkit (`orch help`; settings in
`orch.toml`, rules in `EXPERIMENT_RULES.md`). Each one starts from an approved `EXPERIMENT.md` in its folder under
`studies/ttn/<name>/`: use the `runpod-research:experiment` skill to define and launch one, and
`runpod-research:batch` to run several at once in tmux. The experiment log is `EXPERIMENTS.md`.
