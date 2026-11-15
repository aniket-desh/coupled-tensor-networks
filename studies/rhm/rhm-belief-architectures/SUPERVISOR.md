# Supervisor report: rhm-belief-architectures

The experiment ran to completion with nothing skipped or censored, and is recorded as **inconclusive**: H_A is
inconclusive and H_B is "not supported" by the rules fixed in the definition. It used 4 h 13 min of the 8 h cap and
about $3.04 of the $10 budget. The write-up is [README.md](README.md), the research note is [NOTE.md](NOTE.md), and
the definition with its Amendments and Outcome is [EXPERIMENT.md](EXPERIMENT.md).

## What ran

One worker, `i91nwntyctojrb` (RTX PRO 4500 Blackwell, EU-RO-1, $0.72/h), created 2026-11-14 21:43:09 UTC and deleted
2026-11-15 01:56:17 UTC. The first GPU type on the definition's list, the RTX A6000, had no stock.

| step | when (UTC) | outcome |
|---|---|---|
| checks | 21:44–21:45 | 28 RHM checks, the probe check, the tree check and the 17 `arch_check.py` checks all passed; CUDA present; 112 cores, 32 GB GPU |
| training | 21:46–01:13 | 36 of 36 runs reached 30,000 updates, status `finished`; none skipped, failed or diverged |
| analysis | 21:46–01:46 | 36 of 36 analyzed, no `FAILED` file; 34 on the CPU beside training and the last 2 on the GPU, by the loop's rule |
| verdict and figures | 01:46–01:47 | `summary.json`, `summary.txt` and 15 PDF figures |
| move of the projection files | 01:49 | 72 files moved (see decision 3); the job's exit code 1 came from a trailing `git status`, not from the move |
| finish | 01:50–01:56 | published, collected, merged into `main` (commit 7b21bf0), worker deleted |

Training and analysis were started as the definition's commands, unchanged (the files I sent were diffed against the
definition first).

## Result against each criterion

All at learning rate 1e-3, the only rate tried.

| criterion | observed | verdict |
|---|---|---|
| Run failed (checks, CUDA, at least 31 of 36 runs, shuffled-label R² ≤ 0.05) | checks passed, CUDA present, 36 of 36, shuffled-label R² at most 0.0003 | run valid |
| Learned (all 3 seeds under 0.02 nats excess) | aligned tree: 4 of 4 grammars. Permuted tree and chain: 2 of 4, not v8m4s2L4 and v16m4s2L4 | a finding |
| H_A, chain (at least 3 learned grammars, 75% supporting) | 2 learned grammars, both support it: affine and projective probes hold in all 3 seeds of both, softmax in v4m2s2L5 only | **inconclusive** |
| H_B learning (aligned wins at least 80% of usable pairs, consistent across grammars) | 9 wins, 0 losses, 3 ties of 12; median gap 0.020 nats | to neither tree |
| H_B representation (at least 9 usable pairs) | 6 wins, 0 losses of 6 usable pairs; median adjusted gap 0.153 | untestable |
| H_B overall | no measure goes to either tree, one is testable | **not supported** |
| Descriptive: aligned tree against chain | learning 9 wins, 0 losses, 3 ties of 12; representation 6 ties of 6 | no verdict |

## Cost

- Worker: 4 h 13 min at $0.72/h, about $3.04. The definition expected $4 to $4.7; the chain turned out to cost half
  what the tree does per update (15 to 16 ms against 32 ms at T = 16), so training ended 3.5 h after launch.
- No other worker was started. `orch ls` shows none of this experiment's left.

## Decisions I made

1. **`STOP_AT` kept as written.** Training started 3 minutes after launch, earlier than the 20 minutes the definition
   assumed, so 18600 s put the cutoff before the worker's 5.5 h mark and needed no lowering. It was never reached.
2. **Chain timing rule: carry on.** The first chain run reached update 1,000 in 15 s, a projected 7.5 minutes against
   the 40-minute limit, so the v4m2s2L5 chain runs were not skipped.
3. **Projection files kept out of git (the one amendment).** The 72 `proj_ckpt*.npz` files came to 1.2 GB where the
   definition estimated under 200 MB, more than the repository's whole `.git`. Before finishing I moved them on the
   worker to `runs/rhm-belief-architectures/proj/`, which comes back as data. On the coordinator I copied them back
   into `results/<run>/`, where the plot scripts expect them, and added a `.gitignore` in this folder. No number or
   figure depends on this. It is recorded under Amendments, written before the move.
4. **Result label "inconclusive".** `orch experiment done` takes one label for two hypotheses with different
   verdicts. H_B's formal verdict is "not supported", reached one pair short of its bar with no losses and with its
   second measure untestable; H_A could not be judged. I chose "inconclusive" and stated both formal verdicts in the
   note of `orch experiment done`, in the Outcome and at the top of the README. If you would rather the log say
   "not-supported", it is one `orch experiment done ... --result not-supported` away.
5. **Two small scripts added to the folder.** `arch_table.py` prints the write-up's tables from the saved results
   (`results/tables.md`), and `arch_figures_png.py` writes the PNG copies the note embeds, with the plotting functions
   of `arch_plot.py`. The PDFs were not regenerated or touched.
6. **The relaxed limits were not used.** The relayed message of 01:50 UTC (ceiling $15, movable cutoffs) arrived
   after the last run had been analyzed, so there was nothing to extend and nothing to record under Amendments.
7. **The style reference of the research-note skill was not read.** It lies outside the working directories; one
   read was refused before the instruction not to try arrived, and nothing hung. The note follows the skill's loaded
   text, `write-technical-prose` and the sibling README.

## Fact-checks

Two independent fact-checker agents read the README and the note against the result files, logs and code. Neither
changed a verdict or a table cell, and I applied every correction.

- **README:** about 115 prose claims and the four tables checked; 8 statements corrected. The ones that matter: the
  softmax probe on the unlearned grammars is now described as a weak hint of a log-probability code, because that
  probe is also the strongest on raw tokens; "still falling at the end" became "over the last 10,000 updates",
  because one run was flat over its last 1,000; a wrong argument about an unstable token control was replaced; and
  the size-matched token control of the bilinear probe, missing from the draft, was added.
- **Note:** about 165 claims checked, including the mathematics against the grammar code (the pending-node
  definition, the linearity of the next-token distribution and the midpoint derivation were verified numerically by
  the checker). 21 statements corrected. The two that changed meaning: the draft's opening said the belief is
  readable at 0.90 or above "where a network learned", which is false for the aligned tree on the two m = 4 grammars
  (0.85 and 0.66), and it said both hypotheses "fell short of a verdict" where H_B's verdict is "not supported".
  Both now match the README.
- After applying the corrections I rechecked the changed numbers against `summary.json` and the `analysis.json`
  files myself. The corrected documents were not sent back for a second pass.

## For you to do

- **Decide whether to run the check that would make both hypotheses testable.** The chain and the permuted tree were
  still improving on the two m = 4 grammars over the last 10,000 updates. A higher learning rate or a doubled run for
  those 12 runs is about 1.5 to 2.5 hours on one worker, under $2 (README, "Not yet checked"). It needs a new or
  amended definition, because the learning rate is fixed by this one.
- **The transformer overlay** on `positions.pdf` is left to the planning session, as the definition says:
  `arch_plot.py RESULTS FIGURES <rhm-belief-geometry results folder>`. It rewrites all the PDFs in `figures/`.
- **The experiment log and shared notes** are untouched by me. `orch experiment done` modified `EXPERIMENTS.md` in
  the working tree; I did not commit it.
- **Check how the note renders on GitHub.** The LaTeX avoids the constructs that GitHub's markdown is known to break,
  but nothing on the coordinator renders GitHub markdown, so it has not been seen rendered.
