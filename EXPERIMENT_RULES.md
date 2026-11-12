# Experiment rules for coupled-tensor-networks

These apply to every experiment run here (definitions, runs, reviews and write-ups). Edit freely.

## Before a run

- Run the relevant correctness checks first (`python -m studies.ttn.checks` for the TTN, `python -m studies.<study>.checks` otherwise). A failing check stops the experiment; it is not a result.
- Every comparison has its control in the same run: the same seeds, data and hyperparameters, differing only in the factor under test (sharing pattern, frozen cores, degree, init scale).

## Conventions

- Time is gradient-flow time, t = eta x steps, so runs with different step sizes are comparable.
- Targets are normalized with s1 = 1; other strengths are given as ratios.
- Coefficients c_k are projections of the student on the teacher's orthonormal components, measured on a held-out probe sample. Half-time is the first time c_k reaches s_k / 2.
- Report the probe loss next to the training loss: a gap means the finite sample, not the architecture, is shaping the dynamics.

## Reporting

- Runs that hit a step or time cap are censored: report how many, and never drop or impute them.
- Give every number with its n (seeds) and spread (median and range or quartiles); one seed is an anecdote.
- Separate implementation correctness (checks passed) from evidence for a hypothesis. Negative and inconclusive results are results.
- Figures are PDFs, made by a `plot.py` in the experiment's folder that reads only saved results.
- Keep the numerical outputs, the parameters and seeds, and the plotting code with the results.
- The write-up is `README.md` in the experiment's folder: the answer first, then evidence with numbers, then what was checked and what wasn't.
