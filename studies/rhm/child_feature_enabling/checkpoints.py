"""Read the two root contrasts out of the saved depth-four CE/Adam checkpoints (EXPERIMENT.md, section 2).

Runs: rhm-dynamics root task, width 16, aligned untied identity-init TreeTN, grammars 0-2 x init seeds 0-2, at the 15
saved updates. For each checkpoint file that exists:

  A_c   = 1 - min_w E[(c_Y - w.Z)^2]   exact, from the population moments of the root's product features
          Z = z_left (x) z_right (population.root_moments / regression_oracle; float64). It depends only on the lower
          tree: the best a refitted linear root could do on contrast c with these children frozen.
  CE, accuracy                          of the unmodified classifier on the grammar's fixed 20,000 diagnostic trees.
  D_c^CE = 1 - mean (c_y - sum_j p_j(x) c_j)^2
          the fit of the classifier's own posterior-mean prediction of the contrast, on the same diagnostic sample.
          This is a sample estimate with a different predictor class from A_c (softmax of a trilinear root, not a
          linear readout of Z), so D_c^CE is not bounded by A_c.
  CE interval at reveal=1               deterministic bounds from losses.ce_bracket (64 cells); both endpoints kept.

The contrasts c_e, c_h come from each grammar's top rule table alone (grammar_audit.root_contrasts). A missing
checkpoint is recorded as a row with status 'missing'; nothing is imputed.

  python -m studies.rhm.child_feature_enabling.checkpoints --runs runs/rhm-dynamics
"""
import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from studies.rhm import common
from studies.rhm.child_feature_enabling import grammar_audit, losses, population, record

HERE = Path(__file__).resolve().parent
STEPS = (0, 50, 100, 150, 200, 300, 400, 500, 700, 1000, 1500, 3000, 10000, 20000, 40000)
GRAMMARS, SEEDS, R = (0, 1, 2), (0, 1, 2), 16


def contrasts_for(gcfg):
    """(spec dict for the JSON, float64 tensor (2, n) with rows c_e, c_h) from the top rule table."""
    top = common.rules(gcfg)[0]
    spec = grammar_audit.root_contrasts(top.numpy(), gcfg['v'])
    spec['top_rules_sha256'] = record.tensor_sha256(top)
    c = torch.tensor([spec['easy']['vector'], spec['hard']['vector']], dtype=torch.float64)
    return spec, c


def oracle_rows(moments, c):
    """A_c at the default pseudoinverse cutoff and at ten times it, for the easy and hard contrasts."""
    out = {}
    default = len(moments.gram) * torch.finfo(torch.float64).eps
    for name, target in zip(('e', 'h'), c):
        fit, loose = (population.regression_oracle(moments, target, rtol=k * default) for k in (1, 10))
        out[f'A_{name}'] = fit.explained
        out[f'A_{name}_rank'] = fit.rank
        out[f'A_{name}_cutoff10'] = loose.explained
        out[f'A_{name}_rank_cutoff10'] = loose.rank
    out['gram_rtol'] = default
    return out


@torch.no_grad()
def classifier_metrics(model, x, y, c, batch=512):
    """Sample CE (nats), accuracy and posterior-mean contrast fits of the unmodified classifier (float64 forward)."""
    ce, hits, sq = 0.0, 0, torch.zeros(2, dtype=torch.float64)
    for i in range(0, len(x), batch):
        logits, yb = model(x[i:i + batch]), y[i:i + batch]
        ce += F.cross_entropy(logits, yb, reduction='sum').item()
        hits += (logits.argmax(1) == yb).sum().item()
        sq += (c[:, yb] - c @ logits.softmax(1).T).square().sum(1)
    n = len(x)
    return dict(ce=ce / n, acc=hits / n, D_e_ce=1 - sq[0].item() / n, D_h_ce=1 - sq[1].item() / n)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--runs', default='runs/rhm-dynamics', help='the original checkpoint archive, unpacked')
    parser.add_argument('--out', default=str(HERE / 'results' / 'checkpoints' / 'checkpoints.json'))
    a = parser.parse_args()
    t0 = time.time()
    runs = Path(a.runs)
    meta = record.provenance(__file__, population.__file__, losses.__file__, grammar_audit.__file__)
    meta.update(runs_dir=str(runs), steps=STEPS, width=R, n_diag=common.N_DIAG, n_test=common.N_TEST,
                precision='float64 CPU for moments, oracle and classifier forward passes; checkpoints are float32')
    grammars, rows, final_test = {}, [], []
    for g in GRAMMARS:
        gcfg = dict(common.GRAMMAR, seed=g)
        rules = common.rules(gcfg)
        spec, c = contrasts_for(gcfg)
        grammars[g] = spec
        x, y = common.task_batch(common.diag_trees(gcfg), dict(task='root', grammar=gcfg))
        for s in SEEDS:
            name = f'root_m{gcfg["m"]}_g{g}_r{R}_s{s}'
            logged = {}
            if (runs / name / 'result.json').exists():
                res = json.loads((runs / name / 'result.json').read_text())
                log = res['log']  # columns: step, lr, train_loss, diag_loss, diag_acc, diag_kl
                logged = {step: dict(diag_loss=loss, diag_acc=acc)
                          for step, loss, acc in zip(log['step'], log['diag_loss'], log['diag_acc'])}
            for step in STEPS:
                path = runs / name / f'ckpt_{step}.pt'
                row = dict(run=name, grammar=g, seed=s, step=step, file=str(path))
                if not path.exists():
                    rows.append(dict(row, status='missing'))
                    continue
                model, cfg, saved_step = common.load(path, device='cpu')
                expected = dict(model='tree', r=R, tie='none', perm=None, seed=s, task='root', grammar=gcfg,
                                init='identity')
                assert cfg == expected and saved_step == step, (path, cfg, saved_step)
                model = model.double().eval()
                moments = population.root_moments(model, rules)
                row.update(status='ok', sha256=record.sha256(path), **oracle_rows(moments, c),
                           **classifier_metrics(model, x, y, c))
                low, high, cells = losses.ce_bracket(model, rules, reveal=1)
                row.update(ce_reveal1_lower=low.item(), ce_reveal1_upper=high.item(), ce_reveal1_cells=cells)
                if step in logged:  # the training run's own float32 GPU evaluation on the same trees
                    row.update(logged_diag_loss=logged[step]['diag_loss'], logged_diag_acc=logged[step]['diag_acc'])
                rows.append(row)
                print(f"{name} {step:>6}  A_h {row['A_h']:.4f}  A_e {row['A_e']:.4f}  rank {row['A_h_rank']:>3}  "
                      f"CE {row['ce']:.4f}  acc {row['acc']:.3f}  D_h^CE {row['D_h_ce']:.3f}", flush=True)
                if step == STEPS[-1]:  # the held-out test set is used once, at the last checkpoint
                    xt, yt = common.task_batch(common.test_trees(gcfg), dict(task='root', grammar=gcfg))
                    final_test.append(dict(run=name, grammar=g, seed=s, step=step,
                                           **classifier_metrics(model, xt, yt, c)))
    meta.update(seconds=time.time() - t0,
                missing=[r['file'] for r in rows if r['status'] == 'missing'], n_ok=sum(r['status'] == 'ok' for r in rows))
    record.dump(a.out, dict(meta=meta, grammars=grammars, rows=rows, final_test=final_test))


if __name__ == '__main__':
    main()
