# Handoff: belief-geometry batch

Kept by the tmux session `belief-batch`, which took over the planning conversation on 2026-11-14 at 21:12 UTC, when
the laptop closed. Times are UTC. This file is updated as things change; the last update is at the bottom of each
section.

## Status at a glance

| Experiment | What it asks | Status | Spend / budget |
|---|---|---|---|
| `rhm-belief-geometry` | Does a transformer hold the RHM belief state linearly in its residual stream? | running (supervisor `exp-rhm-belief-geometry`) | $3.73 / $20 at 23:41 |
| `rhm-ttn-belief-geometry` | The same for the tree tensor network's bond states | running (supervisor `exp-rhm-ttn-belief-geometry`) | $2.81 / $10 at 23:41 |
| `rhm-ttn-messages` | Do the tree's bonds hold the grammar's messages, and do its merges implement the update? | launched 21:40 after two review passes (supervisor `exp-rhm-ttn-messages`) | $1.45 / $10 at 23:41 |
| `hmm-belief-geometry` | Simplex's processes (Mess3, RRXOR and others) on a transformer, a tree and a chain | launched 23:44 after two review passes (supervisor `exp-hmm-belief-geometry`) | cap $10 |
| `rhm-belief-architectures` | Aligned tree vs permuted tree vs chain on the RHM | launched 21:42 after two review passes (supervisor `exp-rhm-belief-architectures`) | $1.42 / $10 at 23:41 |

## What you decided before leaving

- Launch the three new experiments once each passes its CPU pilot and an independent review, within $10 and 8 h each.
- Push commits to `origin/main` as they land.

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

## Things to look at first

Nothing finished yet. Dashboards: ports 8010 and 8011 on the coordinator.

## Open items

- Follow-ups planned once results exist: the transformer-vs-TTN overlay figure; the scrambled-target control and the
  belief split on the transformer's checkpoints; the projective probe on the TTN checkpoints.
