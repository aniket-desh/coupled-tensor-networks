# rhm-controls: supervisor report

The experiment ran to completion as defined, with no failures and no amendments, and the result is **supported**: H1 and H3 hold on 12 of 12 pairs for both measures, and H2 shows no consistent difference. It cost about $0.36 of the $6.6 budget. The one thing worth your attention is that the piloted LR for the losing controls sat at the top of the grid, so a cheap follow-up pilot at 1e-2 would firm up H1 and H3.

## What ran

- **Worker.** One NVIDIA GeForce RTX 4090 in EU-RO-1 at $0.89/h (pod `da8btzb43irhzu`, job `20261112-214348-rhm-controls-da8btzb43irhzu`), from 21:43 to 22:08 UTC on 2026-11-12. The L40S, RTX A6000 and A40 were out of stock at launch, so the launcher took the 4090, which is third in the definition's GPU list.
- **Steps.** These were run as separate `orch exec` steps, in the definition's order, and every one exited 0.
  1. `checks`: nvidia-smi, the CUDA assertion and `studies.rhm.checks`, with 26 of 26 checks passing.
  2. `train`: `run.py --hours 4.5`. The 32 LR pilots and 48 grid runs took 19 minutes, at about 2.1–2.3 ms/update for the trees and 3.7–4.8 ms/update for the chain, far inside the 4.5 h deadline. Nothing was skipped.
  3. `analyze` and `plot`, which wrote `results/analysis.json`, `results/curves.json` and `figures/{curves,paired}.pdf`.
- **Collection.** `orch finish` published and merged the results (489 data files into `runs/rhm-controls`, 5 files into the folder) and deleted the worker. `orch ls` afterwards showed no rhm-controls pod. The pods still listed belong to other experiments.

## Result against each criterion

| Criterion | Steps (wins A–B–tie) | Quality (wins A–B–tie) | Verdict |
|---|---|---|---|
| H1: tree vs perm | 12–0–0 | 12–0–0 | supported |
| H2: tree vs tied (two-sided) | 6–6–0 | 0–0–12 | no consistent difference |
| H3: tied vs chain | 12–0–0 | 12–0–0 | supported |
| Descriptive: tree vs chain | 12–0–0 | 12–0–0 | tree better on both |
| Inconclusive or failed rules | | | none triggered: 48/48 finished, no pair dropped, every arm learned |

Detail and caveats are in [README.md](README.md). In brief:

- **H2 splits by task.** The tied tree is faster on 'root' and the untied tree on 'last', 6 of 6 each. The 'root' direction disappears when time is measured as summed LR, because the pilot gave the two arms different LRs.
- **H3's speed win on 'last' disappears in summed LR as well**, since the chain ran at 1e-3 and the tied tree at 3e-3, so on 'last' H3 rests on final quality.
- **The H1 and H3 controls may be under-tuned.** The losing controls' LR was at the grid top (3e-3) for the permuted tree on both tasks and for the chain on 'root'.

## Cost

25 minutes of an RTX 4090 at $0.89/h comes to about $0.36 (the experiment log's figure), against a $6.6 budget and a 6 h cap. The fact-check agent's API usage isn't included.

## Decisions I made

- **Grouped the first three commands into one `checks` step.** nvidia-smi, the CUDA assertion and the correctness checks ran as one `orch exec` step under `set -euo pipefail`, and training, analysis and plotting each got their own step. Grouping changed nothing about what ran or the order.
- **Added `summary.py` and `results/summary.json` after the run.** They compute each arm's excess test error over the Bayes predictor and the summed LR to threshold, read only saved results, and change no verdict. I added them so every number in the README traces to a committed file, since `runs/` is gitignored.
- **Reported summed LR (gradient-flow time) next to the step counts.** The definition's measure is update count, and the verdicts follow it. But the project's rules define time as η × steps, and two of the step-count wins (H2 on 'root', H3 on 'last') vanish under that measure, so the README says so and marks the reading as made after looking at the data.
- **Applied every fact-check correction.** The checker found 13 claims in the first draft that didn't fully hold, and all were corrected. The largest error was that the first draft said scrambling the leaves cost as much as using the chain, when the permuted tree is in fact clearly worse.
- **Recorded the overall result as "supported".** Two hypotheses are supported, and H2 is two-sided with "no consistent difference" as one of its pre-set outcomes, not a failure.

## What you need to do

- **Decide whether to run the LR follow-up.** A pilot at 1e-2 for the permuted tree (both tasks) and the chain ('root'), with a rerun of those 18 runs if 1e-2 wins, would take roughly 15–20 minutes and cost under $0.30. It's the open qualification on H1 and on the 'root' half of H3.
- **Add this experiment to the shared records yourself.** I didn't touch `EXPERIMENTS.md` beyond what `orch finish` wrote automatically, and I didn't touch any shared notes. The experiment-log entry still says "result not recorded yet" and will pick up the result the next time the log is rebuilt.
- **Push when you're ready.** Nothing was pushed. The commits are on `main`: the results merge (`3e9eec2`), the write-up (`b23a44f`) and this report.
