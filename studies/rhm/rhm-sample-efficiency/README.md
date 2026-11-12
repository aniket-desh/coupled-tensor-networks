# RHM sample efficiency: r = 16 TreeTN on fixed training sets of 2^10 to 2^20 trees

An r = 16 TreeTN with the identity init learns the Random Hierarchy Model's root task between N = 2^12 and N = 2^14 training trees, that is, between P* and 4 P*, where P* = n m^L = 4,096 is the sample complexity of deep CNNs on this grammar (Cagnetta et al.). The median root test accuracy over 3 grammars is 0.084 at 2^10 and 0.152 at 2^12, against a chance level of 1/16 = 0.0625. It is 0.99996 at 2^14 and at least 0.9999 at every larger size. Both pre-registered hypotheses are supported. Every run that failed to learn fit its own training set exactly, so in this grid the failures come from too little data, not from failed optimization. The step from failure to success falls inside one factor-of-4 gap in N, so this run places the TreeTN's sample complexity within that factor of P*. It does not resolve the threshold more finely than that.

![test loss and accuracy vs N](figures/vs_N.pdf)

*[figures/vs_N.pdf](figures/vs_N.pdf): one point per run, colored by grammar (n = 3 per size and task). Columns are the two tasks. The bottom row shows test accuracy at the accuracy-selected checkpoint, and the dotted horizontal lines are the online references (fresh trees every update, 20,000 updates). The top row shows test loss at the loss-selected checkpoint (filled), the final test loss (ring) and the final KL to Bayes on the training trees (x). Training-set KLs below 10^-5, which covers most root runs, are drawn at 10^-5. The dashed line is grammar 0's Bayes predictor (the other grammars' Bayes loss and accuracy differ from it by up to 0.033 nats and 0.020), and the dotted vertical line is P* = 4,096 = 2^12. For 'root', the Bayes loss is 0 and its accuracy is 1, so no dashed line appears.*

## Results

**Root task.** The table gives the median and range over the 3 grammars, at the accuracy-selected checkpoint. Bayes accuracy is 1.0 and the Bayes loss is 0 at every size.

| N | N / P* | test accuracy | final training loss (nats) | regimes (of 3) |
|---|---|---|---|---|
| 2^10 | 0.25 | 0.084 (0.080–0.101) | < 10^-4 | 3 data-limited |
| 2^12 | 1 | 0.152 (0.146–0.288) | < 10^-4 | 3 data-limited |
| 2^14 | 4 | 0.99996 (all three equal) | < 10^-4 | 3 learned |
| 2^16 | 16 | 0.99994 (all three equal) | < 10^-4 | 3 learned |
| 2^18 | 64 | 0.99994 (0.99974–1.0) | ≤ 4.3 × 10^-4 | 3 learned |
| 2^20 | 256 | 0.99992 (0.99982–0.99994) | ≤ 2.4 × 10^-4 | 3 learned |
| online | | 0.99948 (0.99936–1.0) | | 3 learned |

At N ≤ 2^12, the root runs memorize. Their training loss goes to 0 within about 3,000 updates, while their diagnostic accuracy stays flat after that, their diagnostic loss climbs to 24–37 nats, and their final test loss is 19–27 nats, because they make confident wrong predictions on new trees. The loss-selected checkpoint of these runs is the near-initial model (update 28–50, test loss 2.77 ≈ log 16), since the diagnostic loss only rises once training starts to move the predictions. So the test loss at small N in the top panel measures the init, not anything the model learned.

**Last-token task.** Bayes accuracy is 0.872–0.903 across the 3 grammars and the Bayes loss is 0.137–0.182 nats. Accuracy is taken at the accuracy-selected checkpoint, and the KL to Bayes at the loss-selected checkpoint. "Excess training loss" is the final training loss minus the Bayes loss on the same 20,000 (or N) training trees. It is negative when a model memorizes and about 0 when it matches Bayes.

| N | test accuracy | test KL to Bayes (nats) | excess training loss (nats) | regimes (of 3) |
|---|---|---|---|---|
| 2^10 | 0.324 (0.320–0.355) | 2.45 (2.37–2.56) | −0.14 to −0.18 | 3 data-limited |
| 2^12 | 0.671 (0.614–0.675) | 1.19 (1.13–1.26) | −0.13 to −0.18 | 3 data-limited |
| 2^14 | 0.842 (0.798–0.852) | 0.44 (0.39–0.52) | −0.14 to −0.18 | 3 learned |
| 2^16 | 0.857 (0.825–0.867) | 0.146 (0.129–0.176) | −0.14 to −0.18 | 3 learned |
| 2^18 | 0.889 (0.870–0.899) | 0.047 (0.044–0.054) | −0.09 to −0.10 | 3 learned |
| 2^20 | 0.893 (0.871–0.903) | 0.012 (0.011–0.013) | −0.004 to −0.006 | 3 learned |
| online | 0.890 (0.873–0.900) | 0.027 (0.025–0.039) | | 3 learned |

For 'last', "learned" (test accuracy ≥ 0.9 × Bayes accuracy) is a loose bar. The 2^14 runs pass it while their KL to Bayes is still 0.39–0.52 nats and their final test loss is 2.6–4.1 nats, against a Bayes loss of 0.14–0.18. The KL column describes generalization better. It falls by a factor of 2 to 4 for each factor of 4 in N, from 2.45 nats at 2^10 to 0.012 nats at 2^20. The training sets stop being memorized only at 2^20, where the excess training loss is within 0.006 nats of 0. At 2^18 the excess is still −0.09 nats, so those runs partly memorize their training sets while generalizing well. The 2^20 runs reach a lower KL than the online references (0.012 at their loss-selected checkpoints, updates 76,000–85,000, against 0.027 at the online runs' last update), but the comparison is not like for like. The online references stopped at update 20,000 of the same 100,000-update schedule, where the LR had decayed only to 0.91 × peak, while the 2^20 runs ran the full schedule down to 0.1 × peak.

![loss curves](figures/curves.pdf)

*[figures/curves.pdf](figures/curves.pdf): diagnostic loss (solid) and training-set loss (dashed) against updates for grammar 0, one curve per N. When a dashed curve keeps falling while its solid curve rises, the run is memorizing.*

## Criteria

Both criteria are computed by `analyze.py` and stored in [results/analysis.json](results/analysis.json). All 42 planned runs finished, so every median is valid.

- **H1, root-task sample complexity near P*: supported.** The median root accuracy at 2^10 is 0.084, below 0.5, and 2^14 reaches 0.99996, above 0.9.
- **H2, failures at large N are data-limited, not optimization-limited: supported.** All 18 runs with N ≥ 2^16 finished, and none is optimization-limited, against a threshold of more than 4.5. This passes because every one of those runs learned, so H2 holds without any large-N failure to classify. The 12 runs that did fail (N ≤ 2^12, both tasks) give the more direct evidence. All 12 fit their training sets, with root training loss below 10^-4 and last-token excess training loss between −0.13 and −0.18 nats, so they are data-limited.
- **Run health:** 42 of 42 finished. 22 hit the early-stop rule after fitting their training data, 14 ran the full 100,000 updates, the 6 online references ran to their 20,000-update cap, and none diverged or was truncated.

## How it was done

The setup follows [EXPERIMENT.md](EXPERIMENT.md). The grammar has v = n = 16, m = 4, s = 2 and L = 4, with 3 grammars (seeds 0–2), on two tasks: 'root' (predict the root symbol from all 16 leaves) and 'last' (predict the 16th leaf from the first 15). The model is an untied r = 16 TreeTN with init seed 0 and the identity init, eps = 0.1. Each grammar has one seeded sample of 2^20 trees, and the training set of size N is its first N trees, so the sets are nested and the comparison across N is paired within a grammar. Training uses Adam with batches of 256 drawn with replacement, for up to 100,000 updates, with a 100-update warmup and a cosine decay to 0.1 × peak. A run stops early only when the diagnostic loss hasn't improved for 25,000 updates and the training loss is below 10^-3 nats. The fixed 20,000-tree diagnostic set is the validation set for both checkpoint selections. The test set is separate.

The peak LR came from an online pilot on grammar 0 with init seed 9, which no evaluation run uses. It ran 4,000 updates of a 40,000-update schedule over {1e-4, 3e-4, 1e-3, 3e-3}. Both tasks chose 3e-3, the top of the grid. The pilot's diagnostic loss at 3e-3 was 1.9 × 10^-4 for 'root' (next best 1.4 × 10^-3 at 3e-4) and 0.223 for 'last' (next best 0.233 at 1e-3).

The run used one NVIDIA A40 in EU-SE-1 at $0.59/h. The worker existed for 1 h 04 min and cost about $0.63, against a cap of 7 h and $8. Training took 59 minutes with 6 runs sharing the GPU, at 7.3–9.8 ms per update per run.

## How it was checked

- The 26 correctness checks in `studies/rhm/checks.py` passed before training ([results/checks.json](results/checks.json)). They compare the Bayes posteriors against brute force on a tiny grammar, check the task batches, round-trip checkpoints with the identity init, and check that training reduces the loss on a fixed batch. These test the implementation and say nothing about the hypotheses.
- Truncated small-N runs: the early-stop rule watches the diagnostic loss, not accuracy, and in several small-N runs the accuracy-selected checkpoint is the stopping step. So I checked whether those runs were cut off while still improving. In the root 2^12 runs, diagnostic accuracy is flat from about update 3,000 to 24,000 (for example, 0.130 to 0.132 for grammar 1), and then rises by at most 0.015 over the last 1,500 updates, next to a spike in the training loss. This doesn't look like a run cut off while it was learning, but the LR at the stopping step was still 0.86 × peak, so 25,500 updates is not proof that longer training wouldn't move it.
- A fact-checker agent compared this write-up against analysis.json, curves.json, the per-run result.json files, the training log and the experiment log. It found 11 claims that didn't fully hold, all corrected here: five numbers rounded the wrong way, the figure caption's description of the clipped training KLs and of the Bayes line, the size of the small-N test losses, the run-health split, the 'about as many examples as CNNs' wording, and the online-reference comparison, which had wrongly said the online LR never decayed.

## Not yet checked

- **Threshold resolution.** The root task goes from 0.15 to 1.0 between 2^12 and 2^14, so the threshold could sit anywhere in that factor of 4. A grid at 2^12.5, 2^13 and 2^13.5 would locate it, at about 6 runs per size and well under an hour on one GPU. Any statement more precise than "between P* and 4 P*" depends on it.
- **LR at the grid edge.** Both tasks picked the largest LR in the grid. A larger LR might train faster, or behave differently at small N. A pilot extended to 1e-2 would take a few minutes. The large-N results are unlikely to change, since they reach Bayes, but the 2^12 failures were measured at only this one LR.
- **Regularization and other recipes at small N.** The small-N failures are data-limited under this recipe, which has no weight decay, dropout or augmentation. Whether a regularized TreeTN learns the root task at 2^12 is a separate question.
- **One init seed.** Every run uses init seed 0, so the spread shown is over grammars only.

## Judgement calls

There were no amendments, and the supervisor made no decisions beyond those in the definition: all 42 runs finished on the first attempt, and nothing needed a rerun. The launcher placed the job on an A40, the second GPU type in the list, because L40S was not in stock outside the coordinator's data center and RTX A6000 had no instances available. This changes only cost and speed.

## What this means

On this grammar, the r = 16 TreeTN with the identity init needs more than P* and at most 4 P* training trees to learn the root task, so its sample complexity is within a factor of 4 of the deep-CNN value. Its failures at small N come from overfitting a training set it can represent, not from an optimizer that can't find a solution. For 'last', accuracy saturates near Bayes by 2^18, while the KL to Bayes keeps falling through 2^20, so a question about calibration rather than accuracy needs the largest sets. The natural next step is the finer grid between 2^12 and 2^14 for 'root'.
