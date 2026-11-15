# Handoff: belief-geometry batch

Kept by the tmux session `belief-batch`, which took over the planning conversation on 2026-11-14 at 21:12 UTC, when
the laptop closed. Times are UTC. This file is updated as things change; the last update is at the bottom of each
section.

## Status at a glance

| Experiment | What it asks | Status | Spend / budget |
|---|---|---|---|
| `rhm-belief-geometry` | Does a transformer hold the RHM belief state linearly in its residual stream? | **finished 01:38. H1 supported: 10 of 12 grammars** | $4.98 / $20 |
| `rhm-ttn-belief-geometry` | The same for the tree tensor network's bond states | **finished 02:15. H2 supported (7 of 8 cells at each width); H1 partly supported (2 of 6)** | $4.39 / $15 |
| `rhm-ttn-messages` | Do the tree's bonds hold the grammar's messages, and do its merges implement the update? | **finished 05:35. Not supported: the tree does not implement the grammar's message passing (H1 and H4 fail in every cell; H2 supported)** | $5.26 / $15 |
| `hmm-belief-geometry` | Simplex's processes (Mess3, RRXOR and others) on a transformer, a tree and a chain | launched 23:44 after two review passes (supervisor `exp-hmm-belief-geometry`) | cap $10 |
| `rhm-belief-architectures` | Aligned tree vs permuted tree vs chain on the RHM | **finished 02:17. Inconclusive: chain hypothesis could not be judged (2 learned grammars, 3 needed); alignment hypothesis not supported by its rule (9 wins, 0 losses, 3 ties)** | $3.04 / $15 |

## What you decided before leaving

- Launch the three new experiments once each passes its CPU pilot and an independent review, within $10 and 8 h each.
- Push commits to `origin/main` as they land.

## Instructions received after the laptop closed

- **01:50, relayed by the main session (now in tmux as `main`):**
  - Every experiment in this batch ends with its plots and a research note as well as the README: `NOTE.md` in the
    experiment folder, written with the research-note skill, with mathematics that renders in GitHub markdown and the
    key figures embedded inline as PNGs under `figures/png/`, next to the unchanged PDF figures; fact-checked and
    committed. The four running supervisors were told directly. `rhm-belief-geometry` had already ended, so its note
    is written by this session. When each of the other four finishes, this session checks that its folder has
    `README.md`, the PDF figures, `figures/png/` and a `NOTE.md` whose images resolve and whose math renders.
  - Each experiment's budget ceiling is now $15, and supervisors may move definition-level time cut-offs so that all
    planned runs are trained and analysed, recorded as amendments authorized by you. The main session extends the
    hard time caps of the `hmm-belief-geometry` and `rhm-ttn-messages` workers itself.

## Deliverables per experiment

| Experiment | README.md | PDF figures | figures/png/ | NOTE.md (images resolve, math renders) |
|---|---|---|---|---|
| `rhm-belief-geometry` | yes | yes (43) | yes (18) | yes: 6 images resolve; math passes a lint for GitHub's known pitfalls (173 inline spans, 12 display blocks) but was not seen rendered, since this machine has no renderer; fact-checked, 16 of about 170 claims reworded |
| `rhm-ttn-belief-geometry` | yes | yes (30) | yes (8) | yes: 7 images resolve; math passes the lint (164 inline spans, 13 display blocks), not seen rendered; fact-checked by its supervisor (17 statements changed) |
| `rhm-ttn-messages` | yes | yes (24) | yes (5) | yes: 5 images resolve; math passes the lint (137 inline spans, 12 display blocks), not seen rendered; three fact-check passes by its supervisor |
| `rhm-belief-architectures` | yes | yes (15) | yes (5) | yes: 5 images resolve; math passes the lint (127 inline spans, 7 display blocks), not seen rendered. Its 7 display equations were written as single-line `$$ … $$`; at 02:30 this session put the delimiters on their own lines, the more robust form, with no change to the content |
| `hmm-belief-geometry` | pending | pending | pending | pending |

## Decisions made while you were away

- 21:12. The two builds that were still in progress when the laptop closed (`hmm-belief-geometry`,
  `rhm-belief-architectures`) were stopped mid-pilot and are being verified from the files on disk, with their first
  reviews running at the same time.
- 21:12. `rhm-ttn-messages` keeps training its own trees (the same script, arguments and seeds as
  `rhm-ttn-belief-geometry`), so it does not wait on that run's checkpoints.

- 21:24. `rhm-belief-architectures` shrinks to fit its 8 h cap. Its first review measured, on the sibling TTN worker,
  17 minutes per T = 16 tree run and 65 minutes per T = 32 tree run with four sharing the GPU, which put training alone
  past 6 hours. So:
  - the builder's extra learning-rate sensitivity runs are dropped, and every comparison is stated as conditional on
    one learning rate (1e-3);
  - the grammar v3m2s2L5 is dropped, leaving v8m4s2L4, v16m4s2L4, v8m2s2L4 and v4m2s2L5 (36 runs);
  - training goes grammar by grammar with the T = 32 grammar last, so a time cut loses a whole grammar;
  - finished runs are analysed on the worker's CPU during training.
- 21:24. Two changes to how `rhm-belief-architectures` reads its results, from the same review: the alignment
  hypothesis is judged only on pairs where both trees learned and against the gap that already exists at
  initialization; and the chain gets a third probe that matches its softmax read-out, so a failure of the affine and
  projective probes is not read as "the belief is not in the state".

- 21:26. The tmux session was restarted once (exit at 21:25:35, back at 21:26:05). It exited cleanly, which points
  to a kill signal: most likely a helper stopped a stale pilot with a pattern-based `pkill -f`, and the pattern matched
  the session's own command line, which then contained the whole handoff prompt. The interrupted helpers and the
  interrupted reviewer were resumed from their saved transcripts, with a rule to kill by PID only and to run one
  python process at a time (the coordinator has 8 GB of RAM). Nothing was lost on disk and no worker was affected.

- 21:32. `hmm-belief-geometry` keeps two things its builder added beyond the brief: a fourth arm (the tree at r = 8,
  the only width at which the second-half product test has enough distinct prefixes per feature on RRXOR and the
  coins; 72 runs in all), and a "halve the control's error" rule in place of a fixed 0.1 margin (on Mess3 the token
  readout already scores 0.96–0.97, so a fixed margin is unreachable).
- 21:32. Four changes to `hmm-belief-geometry` from its first review, all accepted:
  - On five of the six processes the belief is a function of the next-token distribution (an exact linear image on
    Mess3 and the product processes), so a successful probe there shows only that the model predicts well. The
    definition now says which cells can show no more than that. RRXOR is the process that separates the two, and the
    chain and tree hypotheses now have to beat the next-token ceiling there.
  - A cell whose control already scores 0.99 or more is "not testable", not a failure.
  - "Learned" is now below min(0.02, half the optimal 2-gram excess loss) per process, because a model using only the
    last two tokens passed the flat 0.02 gate on the Mess3-type processes.
  - The "which form" verdicts (projective vs affine code; product vs top-only probe) are defined before the run.
  Also: arms that carry verdicts train first, finished runs are analysed on the worker's CPU during training, and
  fewer predictions are saved.

- 21:40. `rhm-ttn-messages` launched. Its first review found that two of its tests asked the wrong question (the
  invariance test expected bonds to drop information the task needs; the "unseen combinations" test was mostly rows
  with no product in them), and that the production-table test was carried by level-1 nodes where it is trivially
  easy. All were fixed and a second reviewer cleared it. After the fixes no hypothesis passes on the short CPU pilots,
  so expect this experiment to come back mixed or negative.

- 21:42. `rhm-belief-architectures` launched with 36 runs. The second review pass cleared it with one note, which was
  taken: a pair now counts as a win for a tree only if the final probe gap is on that tree's side, so two trees that
  end with equal scores never produce a win. The helper's calls that were confirmed: censoring is per hypothesis
  within a grammar; the representation verdict uses t > T/2 only; a measure needs at least 9 usable pairs.

- 23:44. `hmm-belief-geometry` launched with 72 runs, about two hours later than it could have been: this session's
  turn ended on an API error at 21:51, just before the revision went back to the reviewer, and it sat idle until 23:41.
  The four running experiments were not affected. The second review pass cleared the revision with three notes, all
  taken: the RRXOR cells are reported separately from the pooled verdicts (a pooled "supported" could otherwise hide
  a failure in the one process that separates belief from prediction); a sentence that contradicted the summary script
  was fixed; and the r = 32 tree's analyses stay out of the CPU batches that run during training.

## Results

### `rhm-belief-geometry` (transformer), finished 2026-11-15 01:38

Write-up: [studies/rhm/rhm-belief-geometry/README.md](studies/rhm/rhm-belief-geometry/README.md). Supervisor's report:
[SUPERVISOR.md](studies/rhm/rhm-belief-geometry/SUPERVISOR.md). All 51 runs trained and were analysed; 6.92 h and
$4.98 against 8 h and $20.

- **H1 (the last layer holds the belief on prefixes the probe never saw): supported**, 10 of 12 learned and testable
  grammars (the rule needs 75%).
  - Passing grammars: last-layer R² 0.87–0.98 (median over 3 seeds), against 0.19–0.63 for the random-init network
    and 0.19–0.51 for the linear token readout. The margin over the better control is at least 0.22 in every run.
  - Failing: v16m4s2L4 (R² 0.68–0.72 in all three seeds) and the depth-6 grammar v3m2s2L6 (0.52–0.84, passing in 1 of
    3 seeds). Both fail on the 0.8 bar, not on the margin.
  - Not learned: v16m8s2L4 (0.040 nats/token excess loss after 50,000 updates). Four small grammars are figure-only,
    as planned: a random-init network fits them too.
- **H2 (more than the next-token output; descriptive):** in 383 of 414 eligible cases with a next-token ceiling below
  0.9 the probe beats the ceiling by more than 0.05. The exceptions are almost all in the m ≥ 4 grammars (8 of 37
  there). The write-up flags that the ceiling is a lookup estimate, so this count may be inflated.
- **Research note:** [studies/rhm/rhm-belief-geometry/NOTE.md](studies/rhm/rhm-belief-geometry/NOTE.md), written by a
  helper with the research-note skill and fact-checked (committed 02:12). The fact-check also corrected two numbers
  in the README: 148, not 146, of the 414 H2 cases come from v3m2s2L6, and the last-layer range in the synonym-rich
  grammars is 0.68 to 0.98.
- **Follow-up, done 01:45–02:00 (not pre-registered; seed 0 of five grammars that passed H1; 20,000 sequences):**
  [studies/rhm/rhm-belief-geometry/followups/README.md](studies/rhm/rhm-belief-geometry/followups/README.md).
  - The same probe fits a random relabelling of the belief states almost as well as the beliefs: R² 0.70–0.96 against
    0.84–0.97, a pooled gap of 0.015–0.15.
  - It does not extrapolate to beliefs it was not fit on: at most 0.35 where 1.0 was attainable, and no better than
    the random-init network overall (it wins at 9 of 25 positions). This measure is noisy.
  - So H1 stands as stated, but in these runs "linearly determines the belief" is mostly "linearly separates a small
    set of belief states" (a median of 16 per position). Evidence for the belief's geometry, as in the Simplex claim,
    is weak here.
  - Not covered: the two grammars that failed H1 (the v16m4s2L4 follow-up was killed for memory on the coordinator;
    v3m2s2L6 was not run), and seeds 1 and 2.
  - The note was fact-checked: 12 of 46 claims in its first draft did not fully hold, and all were corrected.

### `rhm-ttn-belief-geometry` (tree tensor network), finished 2026-11-15 02:15

Write-up: [studies/rhm/rhm-ttn-belief-geometry/README.md](studies/rhm/rhm-ttn-belief-geometry/README.md); note:
[NOTE.md](studies/rhm/rhm-ttn-belief-geometry/NOTE.md); supervisor's report:
[SUPERVISOR.md](studies/rhm/rhm-ttn-belief-geometry/SUPERVISOR.md). All 60 runs trained and were analysed; 6 h 06 m
and $4.39.

- **H2 (second half: the belief is in the product of the two top bonds): supported at both widths**, 7 of 8 learned,
  testable grammars at each of r = 16 and r = 32. The bilinear probe scores R² 0.85–1.00 on held-out prefixes, and
  0.86–1.00 where the additive ceiling (the best any function additive in the two halves can do) is 0.51–0.78.
- **H1 (first half: affine probe on the top bonds): partly supported**, 2 of 6 at each width. It reaches 0.84–0.99
  in the two small-alphabet grammars (v = 3, 4) and 0.43–0.77 in the four with v = 8 or 16. The write-up's likely
  reading, found after looking and not tested: an r-dimensional bond cannot affinely carry beliefs that span more
  than r dimensions.
- **Geometry: little evidence.** The probes fit a scrambled target about as well as the belief (bilinear difference
  −0.02 to 0.07 per cell) and score below zero on beliefs they were not fit on, in every run. So H1 and H2 show the
  bonds linearly identify the belief state, not that they hold an affine code of it. This matches the transformer
  follow-up above.
- Not learned: v16m4s2L4 at r = 16. The supervisor moved the cut-off for starting training runs so that all 60 ran
  (its decision 4, before your 01:50 message authorized such moves); the r = 16 verdicts depend on that.

- **Follow-up, done 02:32: the transformer and the tree side by side** (seed 0, tree at r = 32, not pre-registered,
  not fact-checked): [studies/rhm/rhm-ttn-belief-geometry/followups/README.md](studies/rhm/rhm-ttn-belief-geometry/followups/README.md)
  with `compare.pdf`. In the second half of the sequence the tree's bilinear probe scores at least as high as the
  transformer's last-layer affine probe in all 9 grammars (median R² 0.87–1.00 against 0.72–0.99) and the tree's
  affine probe is lower in all 9 (0.34–0.86). In the first half the transformer's affine probe is higher in 4 of 5
  grammars. The bilinear probe has 1,088 parameters against the transformer's 129, so this is not evidence that the
  tree represents the belief better.

- **Second follow-up, done 03:05: a probe "up to normalization" on the top bonds** (seed 0 of five grammars, 20,000
  sequences, positions with beliefs of at most 64 entries; not pre-registered, not fact-checked), in the same
  follow-up note. The projective probe scores within 0.07 of the affine probe in every run and half, and within 0.011
  in the first half of v8m4s2L4, where H1 fails. So the first-half shortfall is not explained by the bonds holding
  unnormalized messages. The experiment's bond-dimension explanation is untouched by this.

### `rhm-belief-architectures` (aligned tree, permuted tree, chain), finished 2026-11-15 02:17

Write-up: [studies/rhm/rhm-belief-architectures/README.md](studies/rhm/rhm-belief-architectures/README.md); note:
[NOTE.md](studies/rhm/rhm-belief-architectures/NOTE.md); supervisor's report:
[SUPERVISOR.md](studies/rhm/rhm-belief-architectures/SUPERVISOR.md). All 36 runs trained and were analysed; 4 h 13 m
and $3.04. Recorded as **inconclusive**.

- **Learning.** The aligned tree learned all four grammars (0.0007–0.0059 nats/token above Bayes). The permuted tree
  and the chain learned only the two grammars with m = 2, and stayed above the 0.02 threshold on the two with m = 4
  (chain 0.024–0.068, permuted tree 0.039–0.157). All at one learning rate, 1e-3.
- **Chain hypothesis: inconclusive**, because the rule needs three learned grammars and the chain learned two. In
  both, the affine probe on the chain state scores R² 0.96–0.98 on unseen prefixes, against 0.45–0.55 for the token
  readout and 0.19–0.31 for the untrained chain.
- **Alignment hypothesis: not supported by its pre-registered rule.** On learning the aligned tree won 9 of 12 pairs
  with 0 losses and 3 ties (median gap 0.020 nats), one win short of the 80% bar. On representation it won 6 of 6
  usable pairs (median adjusted gap 0.153), below the 9 pairs needed to count.
- A choice for you: the log label is "inconclusive"; the supervisor notes it is one command away from
  "not-supported" if you prefer that reading.

### `rhm-ttn-messages` (does the tree implement the grammar's message passing?), finished 2026-11-15 05:35

Write-up: [studies/rhm/rhm-ttn-messages/README.md](studies/rhm/rhm-ttn-messages/README.md); note:
[NOTE.md](studies/rhm/rhm-ttn-messages/NOTE.md); supervisor's report:
[SUPERVISOR.md](studies/rhm/rhm-ttn-messages/SUPERVISOR.md). All 30 runs trained and were analysed; 7 h 18 m and
$5.26. Recorded as **not supported**.

- **H1 (bonds hold the partially observed likelihood messages): not supported**, 0 of 5 cells at r = 32 and 0 of 4 at
  r = 16. Per-cell median R² 0.08–0.75 against the 0.8 bar.
- **H2 (a bond keeps what the task needs from its subtree and little else): supported**, 4 of 5 cells at r = 32 and 3
  of 4 at r = 16. The fraction of bond variance explained is 0.81–0.96 in the 25 passing runs, against 0.08–0.41 at
  identity init. Any code that identifies the extended message can pass this, so it does not show message passing.
- **H3 (the learned cores reproduce the production rules): partly supported** at r = 32 (2 of 5 cells), not supported
  at r = 16 (1 of 4).
- **H4 (the merge is the grammar's message update): not supported**, 0 of 9 cells. Per-cell median R² 0.17–0.58 on
  the pairs where both children are informative, against 0.8.
- **Exploratory, added after the run:** zeroing the invalid child pairs in the saved read-out tables recovers 37–79%
  of H4's shortfall at r = 32. So a large part of the gap is that the tree does not suppress child pairs the grammar
  forbids.
- Not learned: v16m4s2L4 at r = 16, as in the tree belief experiment.

## Things to look at first

- [studies/rhm/rhm-belief-geometry/figures/geometry_v8m4s2L4.pdf](studies/rhm/rhm-belief-geometry/figures/geometry_v8m4s2L4.pdf):
  the Simplex-style panel (true beliefs, probes, shuffled and random-init controls, raw PCA) at four positions.
- `triangles_*.pdf` in the same folder for the six v = 3 grammars; `summary.pdf`, `positions.pdf`, `training.pdf`.
- Dashboards: ports 8010 and 8011 on the coordinator.

## Open items

- From the transformer run's supervisor, for you to decide:
  - an unmerged results branch remains (`results/20261114-183036-rhm-belief-geometry-dl1dad9eotnowo`, about 590 MB of
    projection files; everything useful is already in the folder), and a duplicate 1.5 GB copy of the large
    projection files sits in `.orch/incoming/`. Neither was deleted;
  - the projection files (`results/*/proj_ckpt*.npz`, 2.1 GB) are on disk but kept out of git by a `.gitignore` in the
    experiment folder.

- Follow-ups: all three planned ones are done (see Results): the transformer's scrambled-target and belief-split
  controls, the transformer-vs-TTN overlay, and the projective probe on the TTN checkpoints. Each is one seed on a
  reduced sample; the grammars with 4,096-entry beliefs could not be run on the coordinator's 8 GB.
