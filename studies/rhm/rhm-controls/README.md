# RHM architecture controls: correct tree vs permuted leaves, tied cores and a parameter-matched chain

On the Random Hierarchy Model, a TreeTN whose leaf order matches the data tree learns faster and ends better than the same tree on permuted leaves (H1 supported, 12 of 12 pairs on both measures). A level-tied tree, which has a quarter of the parameters, also beats a parameter-heavier shared-core chain (H3 supported, 12 of 12 on both). Weight sharing makes no consistent difference (H2): tying and not tying end at the same quality in all 12 pairs, and their speed difference flips sign between the two tasks. Two qualifications matter.

- **The control arms' tuning.** The per-arm LR pilot chose the top of its grid (3e-3) for the losing arm in H1 on both tasks and in H3 on 'root', so a higher LR for the permuted tree and the chain hasn't been ruled out.
- **What the step counts measure.** On 'last' the tied tree runs at a 3× larger LR than the chain, and measured in gradient-flow time (summed LR) the two reach threshold at about the same point (1.46–2.10 against 1.40–2.05), so only the final-quality win is clear of the LR difference.

The grammar is v = n = 16 symbols, m = 4 productions per symbol, binary branching, depth 4 (16 tokens), with three seeded grammars. There are two tasks: 'root' predicts the root symbol from all 16 tokens (Bayes error 0), and 'last' predicts token 15 from tokens 0–14 (Bayes error 0.097–0.128 depending on the grammar). The tree arms use r = 16 and the chain width 40; every arm uses the identity init and its own piloted LR. Each comparison has 12 pairs, one per (task, grammar, init seed), and the pairs within a comparison share the same online data stream. All 48 planned runs finished. The run took 25 minutes on one RTX 4090 and cost about $0.36.

## Evidence

[`figures/curves.pdf`](figures/curves.pdf) shows diagnostic loss (top) and accuracy (bottom) against updates, with one line per run (6 per arm and task) and the Bayes values per grammar as dashed lines on 'last'. On 'root' the correct and tied trees pass accuracy 0.9 within about 180–540 updates and 0.99 by about 250 (tied) and 700 (tree). At 40,000 the permuted tree has nearly levelled off at accuracy 0.60–0.72, and the chain is still slowly rising at 0.82–0.95. On 'last' the four arms overlap for about the first 50 updates and have separated by 100. The permuted tree and the chain then level off above the Bayes loss (final 0.19–0.30 and 0.18–0.25 against 0.14–0.18), while the tree and the tied tree end within 0.006 of it.

[`figures/paired.pdf`](figures/paired.pdf) has one column per comparison: updates to threshold on top and test error rate below. Colour marks the grammar and shape marks the task. A point above the diagonal means the first-named arm is better, and runs that never reached threshold or had zero error are drawn at the axis edge.

| Arm | Params ('root'/'last') | LR ('root'/'last') | Updates to threshold, 'root' | Updates to threshold, 'last' | Excess test error over Bayes, 'root' | Excess test error over Bayes, 'last' | ms/update ('root') |
|---|---|---|---|---|---|---|---|
| `tree`, correct leaf order | 65,536 / 65,792 | 1e-3 / 3e-3\* | 500 [500, 537] | 400 [313, 449] | 0 [0, 0] | 0.0001 [−0.0013, 0.0014] | 2.1 |
| `perm`, permuted leaves | 65,536 / 65,792 | 3e-3\* / 3e-3\* | never (6 of 6) | 4,500 [643, 7,000] | 0.308 [0.278, 0.398] | 0.038 [0.027, 0.050] | 2.1 |
| `tied`, one core per level | 16,640 / 16,656 | 3e-3\* / 3e-3\* | 200 [183, 219] | 647 [537, 750] | 0 [0, 0.0001] | 0.0001 [−0.0022, 0.0016] | 2.2 |
| `chain`, shared core, width 40 | 65,320 / 65,360 | 3e-3\* / 1e-3 | never in 4 of 6; 25,000 and 27,000 | 1,819 [1,450, 2,100] | 0.152 [0.052, 0.179] | 0.017 [0.015, 0.025] | 4.8 |

Each cell is the median [min, max] over 6 runs (3 grammars × 2 init seeds). "Threshold" is the first logged update where diagnostic accuracy on 20,000 fixed trees reaches 0.9 × the Bayes accuracy. Test error is measured on 50,000 separate trees, and the Bayes error on the same trees, so the excess is a paired difference whose noise comes from the trees where the model and the Bayes predictor disagree; that is why the tree and tied arms land on both sides of zero. \* marks an LR at the top edge of the pilot grid {1e-4, 3e-4, 1e-3, 3e-3}. The numbers are in `results/analysis.json` (table, pairs) and `results/summary.json` (excess error, summed LR).

### Verdicts against the pre-set criteria

A measure goes to an arm if that arm wins at least 10 of the 12 pairs and the per-grammar consistency rule holds. For steps to threshold, a pair is a win only if one arm is ahead by more than a margin, the larger of the logging interval and the reference arm's seed range. Take H2 on 'root', grammar 0, seed 0: the tree reaches threshold at 500 updates and the tied tree at 183, the margin is 37, and 183 < 500 − 37, so the pair goes to the tied tree.

| Comparison | Steps to threshold (wins A–B–tie) | Final quality (wins A–B–tie) | Verdict |
|---|---|---|---|
| H1: tree vs perm | 12–0–0 (sign test p = 0.0002) | 12–0–0 | tree better on both: **supported** |
| H2: tree vs tied (two-sided) | 6–6–0 | 0–0–12 | **no consistent difference** |
| H3: tied vs chain | 12–0–0 | 12–0–0 | tied better on both: **supported** |
| Descriptive: tree vs chain | 12–0–0 | 12–0–0 | tree better on both |

Every arm passed the "did it learn" rule on both tasks (median test accuracy ≥ 0.125 and final diagnostic loss more than 0.05 nats below log 16), so no verdict carries the "control did not learn" tag. No pair was dropped. In H1, H3 and tree vs chain, every grammar split 4–0 for the reference arm on both measures; in H2, every grammar split 2–2 on steps and 4 ties on quality.

### H2 splits by task

The 6–6 tie in H2's step counts isn't noise, because it splits cleanly by task. On 'root' the tied tree is faster in 6 of 6 pairs (183–219 updates against 500–537), and on 'last' the tree is faster in 6 of 6 (313–449 against 537–750). Neither direction carries a pre-set verdict, since each task has only 6 pairs. The 'root' direction also disappears in gradient-flow time: the pilot gave the tied tree 3e-3 and the tree 1e-3, and the summed LR to threshold is 0.40–0.51 for the tied tree against 0.45–0.49 for the tree. On 'root' the tree's pilot losses at 1e-3 and 3e-3 were 3.9e-4 and 6.6e-4 nats, both saturated, so that LR choice was close to arbitrary. On 'last' both arms run at 3e-3, so the tree's lead holds in summed LR too (0.79–1.20 against 1.46–2.10). Both arms end at the Bayes error within sampling noise on 'last', and at 0 error on 'root' apart from one tied run at 4 errors in 50,000.

### H3 on 'last' is a quality result

On 'root' the gap between the tied tree and the chain is large under any measure. On 'last' the tied tree reaches threshold in about a third of the chain's updates (537–750 against 1,450–2,100), but the chain's pilot chose 1e-3 there (its 4,000-update pilot loss was 0.269 at 1e-3 against 0.284 at 3e-3), while the tied tree runs at 3e-3. The summed LR to threshold is about the same for both (tied 1.46–2.10, chain 1.40–2.05; the chain's is lower in 4 of 6 pairs). So in the project's gradient-flow convention the chain isn't slower on 'last'. It ends worse, though: 0.015–0.025 excess error against −0.002 to 0.002 for the tied tree, in 6 of 6 pairs. The pre-set measure is update count with summed LR reported alongside it, so the verdict stands as computed. This reading of it was made after looking at the numbers.

## How it was done

The full method is in [`EXPERIMENT.md`](EXPERIMENT.md). Training is online cross-entropy with Adam, batch 256 and 40,000 updates, a 100-update linear warmup, then a cosine decay to 0.1 × the peak LR.

- **Leaf permutations.** Each (grammar, init seed) gets its own seeded permutation, redrawn until no two data-tree siblings are model siblings. In all six permutations, no model node at levels 1–3 covers exactly the leaves of a data node (`controls_meta.json` in the runs folder, copied into `results/analysis.json` under `meta`).
- **Tied tree and chain.** The tied tree has one leaf embedding and one fusion core per level. The chain is h_t = W(h_{t−1}, A onehot(x_t)) with one W and one A for all positions, at width 40, the width whose parameter count is closest to the untied tree's (within 0.7%).
- **Init and LR.** Every arm uses the identity init (eps = 0.1). The LR was piloted per (task, arm) on grammar 0 with init seed 9, which no evaluation run uses, over 4,000 updates of the 40,000-update schedule.

## How it was checked

- **Correctness checks.** The 26 project checks (`studies/rhm/checks.py`: Bayes posteriors against brute force, task construction, checkpoint round trips for every architecture under the variance init and for the untied tree and the chain under the identity init, the LR schedule) passed on the worker before training (`results/checks.json`).
- **Run health.** 48 of 48 planned runs finished their 40,000 updates, with no truncation, divergence or non-finite metric, so no pair was dropped and the "more than a quarter unfinished" override didn't apply.
- **Whether each arm learned.** Every arm on both tasks passed the rule, so the H1 and H3 wins aren't against a control stuck at the uniform-output saddle. The weakest was the permuted tree on 'root', at median test accuracy 0.69.
- **Fact-check.** A fact-checking pass compared about 90 claims here with `results/analysis.json`, `results/summary.json`, the per-run logs, `checks.json` and the definition. It found 13 that didn't fully hold, all corrected here. The largest corrections: the permuted tree is worse than the chain, where the first draft said they were equal; the grid-edge LR caveat also covers the permuted tree on 'last'; and the summed-LR comparisons are now stated as consistent with an LR explanation rather than as proof of one.

## Not yet checked

- **A higher LR for the losing controls.** Six of the eight (task, arm) pilots chose 3e-3, the top of the grid. The cases that matter are the losing controls at the grid edge: the permuted tree on both tasks and the chain on 'root', whose pilot losses were still falling at 3e-3 (from 1e-3 to 3e-3: perm 'root' 0.966 → 0.918, perm 'last' 0.389 → 0.384, chain 'root' 0.835 → 0.751). A pilot point at 1e-2 would settle it. It would take about 20 s per pilot run on the GPU, plus a rerun of the 18 runs of those three (task, arm) cells, roughly 15–20 minutes and under $0.30. H1 on both tasks and the 'root' half of H3 depend on it. The size of the gaps (0 error against 0.28–0.40 for perm, and 0.05–0.18 for the chain) makes a reversal unlikely but not impossible.
- **Pilots on one seed and grammar, over a tenth of the schedule.** An LR that wins at 4,000 updates can lose at 40,000. Every comparison depends on this, the H2 'root' speed difference most of all, since it disappears in summed LR and so may come entirely from the LR choice.
- **Other chain orderings or a chain with position-specific cores.** H3 compares the tied tree with one chain, scanning left to right. A chain reading the tokens in another order, or one with per-position cores, wasn't run.

## Judgement calls

- **Steps against gradient-flow time.** The definition fixed update count as the speed measure, with summed LR reported next to it. The project's rules say time is t = η × steps. The verdicts above follow the definition. Under summed LR, the H2 'root' direction and the H3 'last' speed win both disappear, and every other speed win holds. A reader who prefers flow time should read H3 as "supported on 'root', quality-only on 'last'".
- **`summary.py` was added after the run.** It computes excess error over Bayes and summed LR per arm from the saved results, because `analysis.json` tables the error but not Bayes, and gives summed LR only per pair. It doesn't change any verdict.

## What this means

On this grammar, the TreeTN's advantage comes from its structure matching the data's hierarchy. Scrambling the leaves costs more than using the chain, which has four times the tied tree's parameters: the permuted tree ends at 0.28–0.40 excess error on 'root' and 0.027–0.050 on 'last', against 0.05–0.18 and 0.015–0.025 for the chain (a descriptive comparison, not one of the pre-set ones). Position-specific parameters buy nothing in final quality, and a quarter-size tied tree fits the Bayes predictor just as well. The useful next step is a 1e-2 LR pilot for the grid-edge controls (the permuted tree on both tasks and the chain on 'root'), which would close the main open qualification on H1 and H3.
