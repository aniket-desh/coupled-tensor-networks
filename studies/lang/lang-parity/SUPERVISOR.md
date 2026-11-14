# lang-parity: supervisor report

The experiment ran to completion from the approved definition without amendments: all 111 runs trained and were interpreted, and H1–H4 are supported (result recorded as `supported`). It cost about $1.75 against a $12 budget, and the worker is deleted. The write-up is [`README.md`](README.md), and the verdict lines are in the definition's "Outcome".

## What ran

One worker (`tjv7csucg46gfg`, NVIDIA A40 in CA-MTL-1, $0.59/h) was created at 23:52 UTC on 2026-11-13 and deleted at about 02:51 UTC on 2026-11-14, so it ran for 2.97 h. The RTX A6000 and L40S weren't available, and setup took about 10 minutes. Every command from the definition ran in order through `orch exec --wait`, and none failed:

| Step | Start (UTC) | Duration | Result |
|---|---|---|---|
| setup (nvidia-smi, CUDA check) | 00:02 | < 1 min | CUDA available |
| checks | 00:02 | 2.5 min | 47 of 47 passed |
| rank | 00:05 | < 1 min | rank 1 at every cut for parity, k = 2, 4, 8, 16 |
| train (`--procs 10`, 6 h deadline) | 00:06 | 2 h 21 min | 111 of 111 runs finished, none skipped |
| interpret, bias-0 runs (1.5 h deadline) | 02:27 | 11 min | 63 of 63 |
| interpret, the rest (0.75 h deadline) | 02:38 | 10 min | 111 of 111 |
| analyze | 02:48 | < 1 min | verdicts below |
| plot | 02:48 | < 1 min | 3 figures |

`orch finish` published 2,583 run files (0.54 GB, under `runs/lang-parity/`, which git ignores) and merged the experiment folder, committing it as `2486230` with nothing held back and no conflicts.

## Result against each criterion

- **H1 (degree):** supported at all 4 ε.
- **H2 (ε = 0.3 vs 0.01):** supported, 6 of 6 pairs.
- **H3 (bias 0.2 vs 0, unbiased timing):** supported, 17 wins and 1 tie in 18 pairs.
- **H4 (causal partial parity):** supported, 203 of 203 nodes moved, against a random median of 0.17.
- **H5 (descriptive):** 21 simultaneous, 3 other, 0 bottom-up. The checkpoint spacing is 1.1–1.8×, so "simultaneous" is at that resolution.
- **Health:** 0 of 111 runs unfinished, so the inconclusive override doesn't apply. 24 runs are censored (18 main, 3 mod-4, 3 r = 2), and they are counted, never dropped.

## Decisions I made

- **The step grouping.** I ran the three setup lines (`mkdir`, `nvidia-smi`, CUDA assert) as one `setup` step, and every later command as its own named step. Two background chains ran with `set -e`, so a failing step would have stopped everything after it.
- **No `--procs 6` fallback.** GPU use was 76% with 10 processes, and the first 10 runs finished in about 16 minutes, so there was no contention to act on.
- **The SSH timeouts.** The `orch exec --wait` poller logged several `ssh: connect … timed out` lines during training. I checked that the waiting process was alive and that direct `orch ssh` reads of the worker worked, and left it running. It completed normally.
- **The plot fix.** In `plot.py` I changed the loss axis of `curves.pdf` to a log scale, because on the linear axis the ε = 1.0 runs (700–2,400 nats at the start) flattened every other curve. I also corrected the docstring, which called the bias-0.2 points squares. This is presentation only and changes no number. The figure was regenerated on the coordinator from the saved results.
- **The exploratory section.** The write-up has one section marked "found after looking". In the censored runs, the 4-bit partial parities become linearly decodable while the 8-bit ones don't. It rests on the descriptive probe only.
- **The fact-check.** The fact-checker found 16 claims that didn't fully hold, and I applied all 16. The substantive ones: the label noise is a 0.05 replacement by a uniform bit (a 0.025 flip), the root core keeps its variance init, ε = 1.0 breaks the ε trend, the bias claim is limited to the pre-registered cells, and the censored count is 12, not 11.

## Commits (this folder only)

- `2486230`: results merged by `orch finish` (figures, `results/*.json`).
- `a4acb1e`: write-up, Outcome and the plot fix.
- This report.

## What the user should do

- **Add the experiment-log entry.** `EXPERIMENTS.md` has uncommitted changes in the working tree that I didn't write, and per the instructions I left it alone, so the lang-parity entry is for the planning session to add or commit.
- **Optional follow-ups,** both listed under "Not yet checked" in the write-up. Denser checkpoints around the loss drop would settle whether H5's "simultaneous" holds at finer resolution. A longer horizon for k = 8 at ε ≤ 0.1 would turn the censored cells into numbers. Each costs well under $1.
- **Run data** is under `runs/lang-parity/` (0.54 GB, not in git), and the published tar on the global volume is deleted after 14 days by `results.keep_days`.
