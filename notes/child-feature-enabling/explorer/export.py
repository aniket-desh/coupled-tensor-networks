"""Build the public study page: notes/child-feature-enabling/explorer/site/ (index.html and data/*.json).

  python notes/child-feature-enabling/explorer/export.py [--runs runs]

Everything the page shows comes from files written here, so a visitor needs nothing else:

  data/study.json        rule tables of grammar seeds 0-9 (0-2 are the study's), the two contrasts of each study
                         grammar, every saved evaluation of every intervention pair in the three protocols, both
                         pilots, the 135-row checkpoint readout and the 20 planted event times.
  data/weights-g{g}.json for grammar g and init seed 0: the stored step-0 tree (leaf embeddings, fusion cores and the
                         16-way root of the original cross-entropy run), and the lower cores and two linear heads of
                         the intact and detached arms of the rate-0.3 pair at the update where that pair stopped.
                         Cores are float16 (little-endian, base64); heads are float32. Rounding to float16 moves a
                         hard-head prediction by at most 0.006 on 4,000 trees and leaves its sample fit unchanged to
                         four digits (checked when this file was written).
  check_vectors.json     (not published) Python reference outputs on fixed trees, read by check_site.py.

It also writes notes/child-feature-enabling/explorer.html, the same page as one file with the data inlined. site/ is
what gets published as the Artifact (index.html as the page, data/ as its files) and is not tracked; explorer.html is.

The final weights live in the git-ignored runs/child-feature-enabling/ and the step-0 checkpoints in runs/rhm-dynamics/
(see the study README for the archive). Needs the project environment (torch).
"""
import argparse
import base64
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from studies.rhm import common, grammar  # noqa: E402

STUDY = ROOT / 'studies/rhm/child_feature_enabling'
RESULTS = STUDY / 'results'
SITE = HERE / 'site'
PROTOCOLS = (  # key, label, folder
    ('written', 'As written', 'intervention'),
    ('amended', 'Amended, rate 0.3', 'intervention-extended'),
    ('rate01', 'Amended, rate 0.1', 'intervention-extended-lr0.1'),
)
KEEP = ('D_h', 'A_h', 'D_e', 'A_e')


def sig(x, digits=4):
    return None if x is None else float(f'{x:.{digits}g}')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def b64(t, dtype):
    """Row-major little-endian bytes of t in dtype ('<f2' or '<f4'), base64."""
    return base64.b64encode(t.detach().double().contiguous().numpy().astype(dtype).tobytes()).decode()


def protocol(key, label, folder):
    pairs = []
    for f in sorted((RESULTS / folder / 'pairs').glob('g*_s*.json')):
        p = json.loads(f.read_text())
        row = dict(g=p['grammar'], s=p['seed'], steps_done=p['steps_done'], init=p['init']['kind'])
        for arm, rec in p['arms'].items():
            cols, cr = rec['columns'], rec['crossing']
            row[arm] = dict(T=cr['T'], censored=cr['censored'], bound=cr.get('bound'), bracket=cr.get('bracket'),
                            bracket_fits=[sig(v, 5) for v in cr.get('bracket_fits', [])],
                            max_fit=sig(max(cols['D_h']), 4), last=cols['step'][-1],
                            step=cols['step'], **{k: [sig(v) for v in cols[k]] for k in KEEP})
        pairs.append(row)
    return dict(key=key, label=label, folder=folder, lr=p['lr'], horizon=p['steps'], stop_fit=p.get('stop_fit'),
                eval_every=p['eval_every'], batch=p['batch'], commit=p['meta']['commit'], pairs=pairs)


def pilot(folder):
    p = json.loads((RESULTS / folder / 'pilot.json').read_text())
    return dict(rates=p['rates'], steps=p['steps'], seed=p['seed'], grammar=p['grammar'], choice=p['choice'],
                rows=[dict(lr=r['lr'], eligible=r['eligible'], max_pop_loss=sig(r['max_pop_loss']),
                           mean_final_hard_mse=sig(r['mean_final_hard_mse'], 5)) for r in p['results']])


def weights(g, runs):
    """Step-0 tree of run root_m4_g{g}_r16_s0 and the final intact/detached arms of the rate-0.3 pair (g, 0)."""
    path0 = Path(runs) / 'rhm-dynamics' / f'root_m4_g{g}_r16_s0' / 'ckpt_0.pt'
    model, cfg, step = common.load(path0)
    assert step == 0 and cfg['tie'] == 'none' and cfg['task'] == 'root' and cfg['grammar']['seed'] == g
    pair = json.loads((RESULTS / 'intervention-extended/pairs' / f'g{g}_s0.json').read_text())
    out = dict(grammar=g, seed=0, r=model.r, init=dict(
        label='step 0 (identity init, before any training)', file=f'rhm-dynamics/root_m4_g{g}_r16_s0/ckpt_0.pt',
        sha256=sha256(path0), A=b64(model.A, '<f2'), W=[b64(w, '<f2') for w in model.W],
        C=b64(model.C, '<f2')))
    assert pair['init']['sha256'] == out['init']['sha256'], 'pair did not start from this checkpoint'
    for arm in ('intact', 'detached'):
        f = Path(runs) / 'child-feature-enabling/intervention-extended' / f'g{g}_s0_{arm}_final.pt'
        ck = torch.load(f, map_location='cpu')
        assert ck['step'] == pair['steps_done'] and ck['lr'] == pair['lr']
        assert sha256(f) == pair['arms'][arm]['final_state']['sha256']
        cols = pair['arms'][arm]['columns']
        out[arm] = dict(label=f'{arm} arm, rate {ck["lr"]:g}, update {ck["step"]:,}', step=ck['step'], lr=ck['lr'],
                        file=f'child-feature-enabling/intervention-extended/{f.name}', sha256=sha256(f),
                        saved_D_h=sig(cols['D_h'][-1], 5), saved_D_e=sig(cols['D_e'][-1], 5),
                        saved_A_h=sig(cols['A_h'][-1], 5), saved_A_e=sig(cols['A_e'][-1], 5),
                        A=b64(ck['lower'][0], '<f2'), W=[b64(w, '<f2') for w in ck['lower'][1:]],
                        w_e=b64(ck['heads'][0], '<f4'), w_h=b64(ck['heads'][1], '<f4'))
    return out


def reference(g, runs, n=24):
    """Python outputs on n fixed trees for check_site.py: parse, step-0 logits and both arms' head predictions."""
    from studies.rhm.child_feature_enabling.intervene import features, lower, run_cfg
    cfg = run_cfg(g, 0)
    rules = common.rules(cfg['grammar'])
    trees = grammar.sample_trees(n, rules, seed=4242 + g)
    x, y = common.task_batch(trees, cfg)
    model, _, _ = common.load(Path(runs) / 'rhm-dynamics' / f'root_m4_g{g}_r16_s0' / 'ckpt_0.pt')
    model = model.double()
    out = dict(grammar=g, leaves=x.tolist(), root=y.tolist(),
               rules=[trees[('rule', l)].tolist() for l in range(4)],
               parse_ok=bool((grammar.parse(x, rules, 16)[0][:, 0] == y).all()))
    with torch.no_grad():
        out['init_logits'] = model(x).tolist()
        for arm in ('intact', 'detached'):
            ck = torch.load(Path(runs) / 'child-feature-enabling/intervention-extended' / f'g{g}_s0_{arm}_final.pt')
            for p, q in zip(lower(model), ck['lower']):
                p.copy_(q)
            Z = features(model, x)
            out[arm] = dict(easy=(Z @ ck['heads'][0].double()).tolist(), hard=(Z @ ck['heads'][1].double()).tolist())
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--runs', default=str(ROOT / 'runs'))
    a = ap.parse_args()
    (SITE / 'data').mkdir(parents=True, exist_ok=True)

    gcfg = common.GRAMMAR
    grammars = {str(s): [common.rules(dict(gcfg, seed=s))[l].tolist() for l in range(gcfg['L'])] for s in range(10)}
    contrasts = {}
    for g in range(3):
        c = json.loads((RESULTS / 'intervention-extended/pairs' / f'g{g}_s0.json').read_text())['contrasts']
        contrasts[str(g)] = dict(
            easy=[sig(v, 6) for v in c['easy']['vector']], hard=[sig(v, 6) for v in c['hard']['vector']],
            additive_easy=sig(c['easy']['additive_fraction'], 4), additive_hard=sig(c['hard']['additive_fraction'], 4),
            gap_easy=sig(c['easy']['neighbor_gap'], 3), gap_hard=sig(c['hard']['neighbor_gap'], 3),
            spectrum=[sig(v, 4) for v in c['additive_fractions']], sign_rule=c['sign_rule'],
            top_rules_sha256=c['top_rules_sha256'])
    ck = json.loads((RESULTS / 'checkpoints/checkpoints.json').read_text())
    readout = [dict(g=r['grammar'], s=r['seed'], step=r['step'], A_e=sig(r['A_e']), A_h=sig(r['A_h']),
                    ce=sig(r['ce']), acc=sig(r['acc']), D_e_ce=sig(r['D_e_ce']), D_h_ce=sig(r['D_h_ce']))
               for r in ck['rows']]
    planted = json.loads((RESULTS / 'planted/planted.json').read_text())
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    study = dict(
        built_from=commit, grammar=gcfg, grammars=grammars, study_grammars=[0, 1, 2], contrasts=contrasts,
        protocols=[protocol(*p) for p in PROTOCOLS],
        pilots=dict(written=pilot('intervention'), amended=pilot('intervention-extended')),
        readout=readout, readout_final_test=ck['final_test'],
        planted=[dict(tied=arm['tied'], s=arm['s'], delta=arm['delta'], left=arm['events']['left_half'],
                      right=arm['events']['right_half'], hard=arm['events']['hard_half']) for arm in planted['arms']])
    (SITE / 'data/study.json').write_text(json.dumps(study, separators=(',', ':')))
    for g in range(3):
        (SITE / f'data/weights-g{g}.json').write_text(json.dumps(weights(g, a.runs), separators=(',', ':')))
    (HERE / 'check_vectors.json').write_text(json.dumps([reference(g, a.runs) for g in range(3)]))

    core = (HERE / 'planted.js').read_text().split('if (typeof module')[0]
    page = (HERE / 'page.html').read_text()
    assert page.count('/*__PLANTED_JS__*/') == 1
    page = page.replace('/*__PLANTED_JS__*/', core)
    (SITE / 'index.html').write_text(page)
    # Single-file copy for readers without the Artifact link: the same page with its data inlined, so it also works
    # from disk (file://). Written over the older explorer.html, which this page replaces.
    bundle = {f'data/{f.name}': json.loads(f.read_text()) for f in sorted((SITE / 'data').glob('*.json'))}
    wrapper = ('<!doctype html>\n<html lang="en">\n<meta charset="utf-8">\n'
               '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n')
    inline = '<script>window.BUNDLE = ' + json.dumps(bundle, separators=(',', ':')) + ';</script>\n'
    (HERE.parent / 'explorer.html').write_text(wrapper + inline + page)
    for f in [*sorted(SITE.rglob('*.*')), HERE.parent / 'explorer.html']:
        print(f'{f.relative_to(HERE.parent)}  {f.stat().st_size / 1e3:.0f} kB')


if __name__ == '__main__':
    main()
