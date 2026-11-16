# Handoff: follow-up batch on the theory notes

Written by the tmux session `wrapup`, which took over the planning conversation on 2026-11-15 at 19:55 UTC. Times are
UTC. **Final version, 2026-11-16 01:40: both GPU experiments are finished, logged and pushed, no worker is running, and
the session has ended.** GPU spend was about $4.67 ($3.21 + $1.46, the log's estimates) against two budgets of $10. The earlier batch's handoff
follows this section, unchanged.

You asked for experiments planned from the theory agent's notes, then for the coordinator's storage to be cleaned up
and the experiments run. Neither GPU experiment gave a clean yes or no: both are recorded as inconclusive, for
different reasons given below. The three reanalyses on the coordinator's CPU did give definite answers.

## Status at a glance

| Experiment | What it asks | Result against its criteria | Spend / budget |
|---|---|---|---|
| `rhm-ttn-masks` | Does training the tree on masks where both children are informative make its merges compose uncertain messages? | **Inconclusive, leaning negative.** The mixed-mask tree passes the merge test in 0 of 12 runs, as does the prefix-mask tree. The definition calls that "not supported" only if the mixed tree learned the varied task in 3 of 4 grammars, and it did in 2. The secondary hypothesis (invalid child pairs suppressed) is not supported. | $3.21 / $10 |
| `belief-probe-controls` | Three reanalyses of saved checkpoints: is the joint belief carried as its factor marginals, is the belief the dominant geometry of the state, and the two open RRXOR tree cells | **Inconclusive, standing for "partly supported", the outcome its definition expected.** The chain and the r = 32 tree hold exact marginals plus part of the joint. The belief is the dominant geometry in 5 of 10 judged Mess3 and RRXOR cells, 7 of 12 RHM transformer cells and 2 of 15 RHM tree cells. | $1.46 / $10 |

Each folder has `README.md`, `NOTE.md` with embedded PNG figures, the PDF figures and a `SUPERVISOR.md`. Both were
reviewed twice before launch and fact-checked once after.

## The two GPU experiments

**`rhm-ttn-masks`** ([write-up](studies/rhm/rhm-ttn-masks/README.md)). 36 runs: 4 grammars, 3 mask distributions
(`prefix`, `varied`, and `mixed`, half of each), 3 seeds, r = 32, with the same sequences, initial tree and optimizer in
the three arms.

- On pairs of child messages that are both informative, the decoded merge matches the exact update with R² 0.22–0.78
  in the 12 mixed-mask runs and 0.18–0.55 in the 12 prefix-mask runs, against a bar of 0.8. Mixed is higher in 8 of
  12 runs (median difference +0.12, range −0.26 to +0.41).
- Invalid child pairs are not suppressed. Their read-out mass relative to valid pairs is lower under mixed masks in 6
  of 12 pairs (10 needed), and the median paired ratio is 1.10 as the script takes it, 1.04 as the mean of the two
  middle values (0.5 at most needed).
- The mixed tree learned the varied task to within 0.02 nats/token of Bayes only in the two grammars with 2
  productions per symbol. In `v8m4s2L4` it ends at 0.025–0.030 and in `v16m4s2L4` at 0.090–0.098. The loss fell
  through the last 10,000 updates in both; over the last 2,000 it was still falling in all seeds of `v16m4s2L4` and
  flat in two of three seeds of `v8m4s2L4`.
- A prefix-trained tree does not transfer to varied masks: its varied excess loss is 1.4–9.2 nats/token.
- The new training script reproduces the earlier experiment: the prefix arm's prefix excess loss agrees with
  `rhm-ttn-messages` to 1.1 × 10⁻⁸ nats/token in the 12 shared runs.
- Why a failed merge test is weak evidence here, as the review established before launch: the test never sees a
  bond from a varied-mask input, and a tree with a softmax read-out may be unable to hold exact messages and reach
  the Bayes loss on varied rows.

**`belief-probe-controls`** ([write-up](studies/hmm/belief-probe-controls/README.md)). Nothing was trained: 24
factor-code runs and 105 low-dimensional probe runs on the saved checkpoints of three earlier experiments.

- **Factor code.** On `factored`, the product of the model's own two marginal probes gives the 9-state joint belief
  at R² 0.999 (chain) and 0.996 (r = 32 tree, first half, the half the rule judges), against 0.967 for the same
  product from token-readout marginals. Their direct joint probes (0.834–0.836 and 0.804–0.819) are 0.08 and
  0.05–0.06 above the 0.755–0.756 that a state holding exactly the two marginals scores. So both pass the rule for
  exact marginals (product at least 0.99) and hold a little of the joint. The transformer's direct probe is
  0.976–0.982, against 0.921–0.923 for the initial transformer and 0.919 for a token readout. The r = 8 tree sits at
  the floor, with marginals that are not exact (product 0.980–0.981). Most of this followed from saved numbers,
  which the definition says.
- **`coupled` cannot test correlations.** Its residual beyond the product of marginals carries 0.10–0.11% of the
  joint's variance, so it was not read.
- **Is the belief the dominant geometry of the state?** "Dominant" means most of the state's variance is a linear
  function of the belief; "compact" means the belief is readable from as few of the state's top principal
  directions as the belief itself needs. Among learned cells with three judged seeds:
  - Mess3 and RRXOR: dominant in 5 of 10 cells, and the same 5 are compact: Mess3 in the chain, the transformer and
    the first half of both trees, and RRXOR in the chain.
  - RRXOR in the transformer is neither: the belief is readable from the full last layer, but the reverse score is
    0.10, 0.10 and 0.43 in the three seeds.
  - RHM transformer: dominant in 7 of 12 grammars, compact in 0 of 12. Both counts are sensitive to choices in the
    rule: with positions weighted equally the dominant count is 5 of 12, and at twice the number of directions 2 of
    12 are compact.
  - RHM tree at r = 32: dominant in 2 of 15 cells, compact in 4 of 15, none both.
  - Two cautions. On Mess3 the belief is a function of the exact next-token distribution, so "dominant" there says
    the main directions are the model's own prediction. And some cells rest on few positions: 2 or 3 for the first
    half of the Mess3 trees and 2 for the RRXOR transformer.
- **RRXOR trees.** The r = 8 seed that scored −4.55 has its error in 146 of 31,129 test rows at t = 13; without them
  it scores 0.839 against 0.673 and 0.715 for the other seeds, so the rule written for it (within 0.05) is not met,
  and the affine probe, which drops no rows, gives 0.751. The r = 32 tree with reduced product features scores 0.884,
  0.883 and 0.895 against a required 0.886: it misses in two seeds by 0.002, with no interval computed.

## The reanalyses on the coordinator's CPU

All three were run from saved results with no network trained, and are written up next to their scripts.

- **Capacity ceiling** ([write-up](studies/rhm/rhm-ttn-belief-geometry/followups/README.md)). An affine probe on a
  bond with r coordinates cannot exceed the rank-r Eckart–Young value of the beliefs. Of the 8 judged cells that
  failed the first-half criterion of `rhm-ttn-belief-geometry`, 3 have a ceiling below the 0.8 bar (0.609, 0.688,
  and 0.796, which is at the threshold) and 5 do not (0.840–0.969). In all 8 the probe is 0.15–0.29 below its ceiling, so the low first-half scores
  are mostly an inefficient affine code. No probe score exceeds its ceiling (largest difference −1.6 × 10⁻⁴ over 39
  runs).
- **What the scrambled-target control could detect** (same write-up). With the exact belief as the activations, the
  experiment's probe fits the scrambled target at R² 0.72–0.96 in the first half of all 13 cells that have eligible
  first-half positions, because the distinct beliefs are nearly affinely independent. The control could therefore
  not have tested geometry there. In the four r = 32 second-half cells where that oracle is low (0.26–0.48), the
  bilinear probe fits the scrambled target at 0.81–1.00 (median of three seeds), which shows the product features
  carry more than the belief, not that the belief's geometry is absent. This check was not in the theory notes; it replaced the matched
  permutation control, which behaves the same way.
- **Partial parities in the stalled runs** ([write-up](studies/lang/lang-parity/followups/README.md)). In the 12
  runs that never learn, the share of a level-2 bond's variance along its 4-bit partial parity grows by a factor of
  3 × 10³ to 8 × 10⁹ during training and ends at a median of 1.5 × 10⁻⁴ (range 2.9 × 10⁻⁶ to 3.0 × 10⁻²). At each
  init scale where both kinds of run exist, that is at least as large in the median as the share the learned runs
  have at their last checkpoint before they learn, with overlapping ranges; over all 24 learned runs that share has
  a median of 6.7 × 10⁻³. Patching those directions changes the
  output's KL by at most 2.3 × 10⁻⁵ nats, against 3.35–3.58 in the learned runs. So the stalled runs built the 4-bit
  parities as small, growing directions and never combined them.

## What the batch says, in one paragraph

The picture from the earlier batch gets sharper and less negative. The tensor networks do hold beliefs in
interpretable forms: the chain and the r = 32 tree carry a product process as its factor marginals, to R² above
0.99, plus a little of the joint, and on Mess3 the belief is the dominant, low-dimensional geometry of the chain's
state, of the transformer's last layer and of the first half of the trees (where it is also the model's own
prediction). On RRXOR that holds for the chain and not for the
transformer, and on the RHM the belief dominates the transformer's last layer in 7 of 12 grammars without being
compact in any. The earlier reading that probes fit a scrambled target "about as well as" the belief was a weaker
statement than it looked, since the control could not detect geometry in the first half. What stays negative is the
grammar's message algebra: adding the observation patterns that the prefix task lacks did not make the tree's merge
pass the update test in any run, though the test and the training both have limits that keep this from being a
clean negative.

## Decisions that are yours

- **`results.keep_days` in `orch.toml`.** I changed it from 14 to 0 and left the edit uncommitted. When I asked you
  about deleting the old `runs/` folders I said they could be restored from the published archives, without knowing
  that `orch` deletes those archives 14 days after collection. With 0 the archives are kept; revert the edit if you
  would rather let them expire, which makes those deletions permanent.
- **The varied-pattern follow-up for `rhm-ttn-masks`:** probe a bond for its normalized message on bonds collected
  from varied-mask inputs. Its reading was fixed before the run for one outcome: if the probe reaches R² ≥ 0.8 in the
  mixed arm where the prefix arm does not, the mixed tree carries uncertain messages even though the merge test
  fails. It needs a new analysis script and the saved checkpoints only (`runs/rhm-ttn-masks/`, 274 MB). It does not
  change the recorded verdict, which depends on the varied gate in the next item.
- **A longer schedule or a wider tree for the two grammars with 4 productions per symbol.** The primary hypothesis
  cannot return "not supported" until the mixed tree meets the varied gate in a third grammar, and `v8m4s2L4` missed
  it by 0.005–0.010 nats/token.
- **The label for `belief-probe-controls`.** The tool has no "partly supported", so its supervisor recorded
  "inconclusive", as `hmm-belief-geometry` did. Change it with `orch experiment done` if you prefer another label.
- **Follow-ups its write-up lists as not yet checked** (six in all; these are the three I would weigh first): a
  bootstrap over prefixes for the r = 32 RRXOR tree, which two seeds miss by 0.002; repeating the low-dimensional
  probes for the chain and the trees after fixing the gauge of the bonds; and what the RRXOR transformer's one or two
  top-variance directions encode.

## Things to know

- **Storage.** The coordinator's volume read 40G before the cleanup and 22G after it on 2026-11-15 (`du -sh
  /workspace`, a 50 GB volume, no expansion). Removed: eight `runs/` folders belonging to seven older experiments
  (each compared byte for byte with its published archive first), a duplicate copy of projection files, the package
  cache and unreachable git objects. It reads 35G now, because another session has since added
  `/workspace/participation-ratios` (9.8 GB) and refilled part of the package cache.
- **An unexplained difference.** The prefix arm of `rhm-ttn-masks` repeats 12 runs of `rhm-ttn-messages` with prefix
  losses equal to 1.1 × 10⁻⁸, yet the analysis of those runs differs by up to 0.027 in both-informative R² and 0.11
  in the invalid read-out ratio. The supervisor reported it and did not investigate the cause. It is far smaller
  than the prefix arm's distance to the 0.8 bar (0.25 or more), but larger than the 0.017 by which the best mixed
  run misses that bar on the both-informative pairs (`v3m2s2L4` seed 1 at 0.78, which also fails the test's other
  clauses).
- **One statement in a definition was too strong.** `belief-probe-controls` says the tree's second half cannot be
  dominant by construction; no second-half cell is, but one run is (`v4m2s2L5` seed 0). The write-ups say "not
  expected".
- **Not seen rendered.** Neither `NOTE.md` has been viewed rendered, since the coordinator has no Markdown renderer.
  Their image links resolve.
- **Fact-checks ran once.** Each supervisor applied its fact-checker's corrections and did not run a second pass on
  the corrected text.
- **Changes from the plan as I first presented it.** In `belief-probe-controls` the matched permutation control was
  dropped and the low-dimensional and reverse probes became the geometry test, and the capacity ceiling and the
  parity measurement were done on the coordinator's CPU outside the experiment. In `rhm-ttn-masks` two choices were
  made without asking, which its definition counts as within the plan: the query is the leftmost mask instead of a
  new query symbol, and the T = 32 grammar `v4m2s2L5` was left out for cost.
- **This section was fact-checked once.** An independent pass checked about 110 claims against the result files and
  reworded 13. The two that mattered: the storage figure was out of date, and the list of changes from the plan
  misdescribed two of them. It also asked for the cautions on the dominant and compact counts. The corrections were
  applied and not re-checked. What it could not verify from files: the storage readings before and after the
  cleanup, and what was said in conversation.
- **Nothing failed.** Three RHM transformer runs in `belief-probe-controls` ran out of GPU memory and were rerun
  one at a time, as its definition prescribes.

---

# Handoff: belief-geometry batch

Kept by the tmux session `belief-batch`, which took over the planning conversation on 2026-11-14 at 21:12 UTC, when
the laptop closed. Times are UTC. **Final version, 2026-11-15 09:25: all five experiments are finished, logged and
pushed, no worker is running, and the session has ended.** Total GPU spend was $23.85 (4.98 + 4.39 + 3.04 + 5.26 +
6.18) against ceilings of $20 and four times $15.

## Status at a glance

| Experiment | What it asks | Status | Spend / budget |
|---|---|---|---|
| `rhm-belief-geometry` | Does a transformer hold the RHM belief state linearly in its residual stream? | **finished 01:38. H1 supported: 10 of 12 grammars** | $4.98 / $20 |
| `rhm-ttn-belief-geometry` | The same for the tree tensor network's bond states | **finished 02:15. H2 supported (7 of 8 cells at each width); H1 partly supported (2 of 6)** | $4.39 / $15 |
| `rhm-ttn-messages` | Do the tree's bonds hold the grammar's messages, and do its merges implement the update? | **finished 05:35. Not supported: the tree does not implement the grammar's message passing (H1 and H4 fail in every cell; H2 supported)** | $5.26 / $15 |
| `hmm-belief-geometry` | Simplex's processes (Mess3, RRXOR and others) on a transformer, a tree and a chain | **finished 09:00. Mixed (logged as inconclusive): transformer reproduction supported (2 of 2); chain and tree each partly supported** | $6.18 / $15 |
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
| `hmm-belief-geometry` | yes | yes (9) | yes (9) | yes: 5 images resolve; math passes the lint (168 inline spans, 15 display blocks), not seen rendered; two fact-check passes by its supervisor |

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

### `hmm-belief-geometry` (Simplex's processes on a transformer, a chain and trees), finished 2026-11-15 09:00

Write-up: [studies/hmm/hmm-belief-geometry/README.md](studies/hmm/hmm-belief-geometry/README.md); note:
[NOTE.md](studies/hmm/hmm-belief-geometry/NOTE.md); supervisor's report:
[SUPERVISOR.md](studies/hmm/hmm-belief-geometry/SUPERVISOR.md). All 72 runs trained and were analysed, and every
(process, model) cell learned; 8 h 57 m and $6.18. Logged as **inconclusive**, which here means a mixed outcome.

- **H1 (transformer reproduction): supported, 2 of 2.** The last-layer affine probe scores R² 0.9997 on Mess3
  (initial network 0.976–0.978, token readout 0.967) and 0.865–0.872 on RRXOR (initial network 0.25–0.29, tokens
  0.063, next-token ceiling 0.474), in all three seeds.
- **H2 (chain, projective probe): partly supported, 2 of 4.** It passes on Mess3 (0.9999) and RRXOR (0.994–0.997
  against a next-token ceiling of 0.469). It fails on the two product processes against the token readout
  (`factored` 0.844–0.845 against 0.919; `coupled` 0.956–0.960 against 0.928, short of the halving rule). The coins
  cell is not testable, because its control is saturated.
- **H3a (tree, first half): partly supported, 2 of 6.** Mess3 passes at both bond dimensions; `factored` and `coupled`
  fail at both (0.75–0.89 for the joint belief, below the token readout at 0.92–0.93).
- **H3b (tree, second half, product of the top bonds): partly supported, 1 of 2.** The coins pass at r = 8
  (0.991–0.994). RRXOR at r = 8 fails: 0.733 and 0.767 in two seeds and −4.55 in the third, against an additive
  ceiling of 0.772.
- **RRXOR, the process that separates belief from prediction, splits 2 to 1:** the transformer and the chain pass,
  the r = 8 tree fails. The write-up adds the same caveat as the RHM experiments: RRXOR has 17 distinct beliefs per
  position, and an affine probe fits a random vector per belief state with R² 0.81 for the transformer (0.865–0.872
  for the belief) and 0.88–0.92 for the chain, so the passes show the state identifies the belief state, not that it
  lays out the belief geometry.
- **Product processes (descriptive):** every model reads the two factor marginals at 0.97 or more; only the
  transformer carries the joint (0.98–0.995, against 0.75–0.96 in the tensor networks).

## What the batch says, in one paragraph

Across all five experiments the pre-registered probe results mostly hold where the model learned the task: the
transformer's last layer, the chain's state and the product of the tree's top bonds each linearly determine the exact
belief on prefixes the probe never saw, well above an untrained network and a token readout. What does not hold up is
the stronger, Simplex-style reading that these representations lay out the belief *geometry*. In all four belief
experiments, a probe fits a scrambled relabelling of the belief states most of the way to how well it fits the beliefs
themselves (a pooled gap of 0.015–0.15 for the transformer on the RHM, within 0.07 for the tree's bilinear probe,
within 0.06 for every arm of the architecture comparison on learned grammars, and 0.81 against 0.865–0.872 for the
transformer on RRXOR). In the three RHM belief experiments a probe also fails to extrapolate to beliefs it was not fit
on; the HMM experiment did not run that test, and the test is a noisy one. The tree does not implement the grammar's
message passing
(`rhm-ttn-messages`: its merges fail the update test in every cell), though its bonds keep what the task needs and
little else. The aligned tree learned all four grammars where the permuted tree and the chain learned two, but the
alignment comparison fell one win short of its pre-registered bar.

## Things to look at first

- [studies/rhm/rhm-belief-geometry/figures/geometry_v8m4s2L4.pdf](studies/rhm/rhm-belief-geometry/figures/geometry_v8m4s2L4.pdf):
  the Simplex-style panel (true beliefs, probes, shuffled and random-init controls, raw PCA) at four positions.
- `triangles_*.pdf` in the same folder for the six v = 3 grammars; `summary.pdf`, `positions.pdf`, `training.pdf`.
- The five research notes, one per experiment folder (`NOTE.md`), each with embedded figures:
  [transformer](studies/rhm/rhm-belief-geometry/NOTE.md), [tree](studies/rhm/rhm-ttn-belief-geometry/NOTE.md),
  [architectures](studies/rhm/rhm-belief-architectures/NOTE.md), [messages](studies/rhm/rhm-ttn-messages/NOTE.md),
  [HMM processes](studies/hmm/hmm-belief-geometry/NOTE.md). Their math passed a lint but was not seen rendered: open
  one on GitHub first and tell me if an equation shows as raw text.
- [studies/rhm/rhm-ttn-belief-geometry/followups/compare.pdf](studies/rhm/rhm-ttn-belief-geometry/followups/compare.pdf):
  the transformer and the tree on the same held-out prefixes.
- The experiment log, `EXPERIMENTS.md`, has all five runs.

## Open items

- **Decisions that are yours:**
  - `rhm-belief-architectures` is logged as "inconclusive"; its supervisor notes that "not-supported" is one
    `orch experiment done` command away if you prefer that label for the alignment hypothesis.
  - `rhm-ttn-belief-geometry`'s supervisor moved the cut-off for starting training runs so that all 60 ran, before
    your 01:50 message authorized such moves. The r = 16 verdicts depend on that; the r = 32 ones do not.
  - Each supervisor kept its large projection files (1–3 GB per experiment) out of git and left notes on what
    remains on disk; see each `SUPERVISOR.md` under "What you need to do".
- **The notes' math was never seen rendered** (no renderer on the coordinator). All five pass a lint for GitHub's known
  pitfalls.
- **Follow-ups not done:** seeds 1 and 2 for the three follow-ups; the two RHM grammars with 4,096-entry beliefs,
  which need more than the coordinator's 8 GB; a low-dimensional probe that would test belief geometry directly.

- From the transformer run's supervisor, for you to decide:
  - an unmerged results branch remains (`results/20261114-183036-rhm-belief-geometry-dl1dad9eotnowo`, about 590 MB of
    projection files; everything useful is already in the folder), and a duplicate 1.5 GB copy of the large
    projection files sits in `.orch/incoming/`. Neither was deleted;
  - the projection files (`results/*/proj_ckpt*.npz`, 2.1 GB) are on disk but kept out of git by a `.gitignore` in the
    experiment folder.

- Follow-ups: all three planned ones are done (see Results): the transformer's scrambled-target and belief-split
  controls, the transformer-vs-TTN overlay, and the projective probe on the TTN checkpoints. Each is one seed on a
  reduced sample; the grammars with 4,096-entry beliefs could not be run on the coordinator's 8 GB.
