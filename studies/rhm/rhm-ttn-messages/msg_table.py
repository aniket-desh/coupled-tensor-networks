"""Cell-level markdown tables for README.md and NOTE.md, from results/summary.json and results/*/analysis.json.

    python studies/rhm/rhm-ttn-messages/msg_table.py RESULTS_DIR

A cell is one grammar at one width (three seeds). Every entry is the median over the three seeds with the range in
brackets, at the final checkpoint; `init` is checkpoint 0 (the identity-init tree). The columns are msg_summary.py's:
M1 and M2 pool their eligible nodes, M3 and M4 pool parents of tree level >= 2. The last table is per tree level
(msg_analyze.pooled, scopes l1, l2, ...), again the median over seeds."""
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median

res = Path(sys.argv[1])
S = json.load(open(res / 'summary.json'))
by = defaultdict(list)
for x in S['runs']:
    by[x['run'].rsplit('_s', 1)[0]].append(x)
order = sorted(by, key=lambda c: (-by[c][0]['r'], c))


def mr(rows, k, fmt='.2f'):
    v = [x.get(k) for x in rows]
    if any(a is None for a in v):
        return '-'
    return f'{median(v):{fmt}} [{min(v):{fmt}}, {max(v):{fmt}}]'


def verdict(c, h):
    v = S['cells'][c]
    if not v['learned']:
        return 'not learned'
    n = sum(bool(x.get(h)) for x in by[c])
    return {True: 'yes', False: 'no', None: 'not testable'}[v[h]] + f' ({n}/3 seeds)'


def name(c):
    g, r = c.rsplit('_r', 1)
    return f'`{g}` | {r}'


def table(title, head, row):
    print(f'\n{title}\n')
    print('| grammar | r | ' + ' | '.join(head) + ' |')
    print('| --- | --- | ' + ' | '.join('---' for _ in head) + ' |')
    for c in order:
        print(f'| {name(c)} | ' + ' | '.join(row(c, by[c])) + ' |')


sh = [x['shuffle'] for x in S['runs'] if x['shuffle'] is not None]
ex = [x['excess'] for x in S['runs']]
print('runs analyzed: %d; run_failed: %s' % (S['n_runs'], S['run_failed'] or 'no'))
print('shuffled-label R2, worst eligible node per run: min %.4f max %.4f (n = %d runs)' % (min(sh), max(sh), len(sh)))
print('excess loss: min %.4f max %.4f; runs at or above 0.02: %s' % (
    min(ex), max(ex), ', '.join(f"{x['run']} ({x['excess']:.3f})" for x in S['runs'] if x['excess'] >= 0.02)))
for k, v in S['verdicts'].items():
    print(k, v['verdict'], f"{v['supported']} of {len(v['judged'])}", 'pass:', [c for c in v['judged'] if S['cells'][c][k[:2]]])

table('M1 / H1: projective probe to the normalized message of partially observed nodes, held-out local prefixes',
      ['eligible nodes', 'distinct partial messages per node', 'trained', 'identity init', 'token readout',
       'scrambled target', 'fully observed rows', 'H1'],
      lambda c, rr: ['-'.join(sorted({str(x['n_elig']) for x in rr})),
                     f"{min(min(x['n_partial_messages']) for x in rr)}-{max(max(x['n_partial_messages']) for x in rr)}"
                     + (' (identity test)' if S['cells'][c]['h1_identity_test'] else ''),
                     mr(rr, 'm1_proj'), mr(rr, 'm1_init'), mr(rr, 'm1_tok'), mr(rr, 'm1_scr'), mr(rr, 'm1_full'), verdict(c, 'h1')])

table('M1 on the nodes where the pairwise token readout is eligible, and M1x (the extended message, descriptive)',
      ['nodes with pairwise readout', 'bond probe there', 'pairwise token readout', 'M1x trained', 'M1x identity init', 'M1x token readout'],
      lambda c, rr: ['-'.join(sorted({str(x['m1_tok2_nodes']) for x in rr})), mr(rr, 'm1_proj_at_tok2'), mr(rr, 'm1_tok2'),
                     mr(rr, 'm1x_proj'), mr(rr, 'm1x_init'), mr(rr, 'm1x_tok')])

table('M2 / H2: share of whitened bond variance explained by the grouping, rows with a synonymous partner',
      ['extended message: trained', 'identity init', 'share of rows with a partner', 'message alone: trained', 'identity init', 'H2'],
      lambda c, rr: [mr(rr, 'm2_ve'), mr(rr, 'm2_init'), mr(rr, 'm2_cover'), mr(rr, 'm2_msg'), mr(rr, 'm2_msg_init'), verdict(c, 'h2')])

table('M3 / H3: production table from the cores, parents of tree level >= 2 (level 1 in the last two columns)',
      ['valid pairs', 'accuracy: trained', 'identity init', 'invalid/valid bond norm: trained', 'identity init',
       'invalid/valid read-out mass: trained', 'identity init', 'level 1 accuracy: trained', 'identity init', 'H3'],
      lambda c, rr: ['-'.join(sorted({str(x['m3_pairs']) for x in rr})), mr(rr, 'm3_acc'), mr(rr, 'm3_init'),
                     mr(rr, 'm3_invalid'), mr(rr, 'm3_invalid_init'), mr(rr, 'm3_invalid_q'), mr(rr, 'm3_invalid_q_init'),
                     mr(rr, 'm3_l1_acc'), mr(rr, 'm3_l1_init'), verdict(c, 'h3')])

table('M4 / H4: decoded merge of mixture bonds against the exact update, parents of tree level >= 2',
      ['all empirical pairs', 'both-informative: trained', 'identity init', 'additive ceiling', 'left-masked: trained',
       'additive ceiling', 'seen in training: trained', 'additive ceiling', 'H4'],
      lambda c, rr: [mr(rr, 'm4_emp'), mr(rr, 'm4_both'), mr(rr, 'm4_both_init'), mr(rr, 'm4_both_ceil'), mr(rr, 'm4_left'),
                     mr(rr, 'm4_left_ceil'), mr(rr, 'm4_seen'), mr(rr, 'm4_seen_ceil'), verdict(c, 'h4')])

# M4a at nodes of level >= 2 is not in summary.json (which pools M4a over all internal nodes, level 1 included)
for rr in by.values():
    for x in rr:
        a = json.load(open(res / x['run'] / 'analysis.json'))
        P = a['metrics'][str(max(int(k) for k in a['metrics']))]['pooled']
        x['m4a_l2'] = (P['l2+'].get('m4a_emp_proj') or {}).get('R2')
table('M4, other subsets (descriptive): Dirichlet pairs, code linearity (M4a), level-1 parents',
      ['Dirichlet pairs: trained', 'identity init', 'additive ceiling', 'M4a, all internal nodes: trained', 'identity init',
       'M4a, nodes of level >= 2: trained', 'level 1, all empirical pairs: trained', 'identity init'],
      lambda c, rr: [mr(rr, 'm4_dir'), mr(rr, 'm4_dir_init'), mr(rr, 'm4_dir_ceil'), mr(rr, 'm4a_emp'), mr(rr, 'm4a_init'),
                     mr(rr, 'm4a_l2'), mr(rr, 'm4_l1_emp'), mr(rr, 'm4_l1_init')])

# per tree level, from the analysis files: final checkpoint, median over seeds
print('\nPer tree level of the node (M1) or parent (M3, M4), final checkpoint, median over the three seeds\n')
print('| grammar | r | level | M1 trained (nodes) | M3 accuracy | M3 invalid/valid norm | M3 invalid/valid read-out mass | '
      'M4 all empirical | M4 both-informative | its additive ceiling |')
print('| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |')
g = lambda P, key, f='R2': (P.get(key) or {}).get(f)
med = lambda v: '-' if any(a is None for a in v) else f'{median(v):.2f}'
for c in order:
    A = []
    for x in by[c]:
        a = json.load(open(res / x['run'] / 'analysis.json'))
        A.append(a['metrics'][str(max(int(k) for k in a['metrics']))]['pooled'])
    for l in sorted(int(k[1:]) for k in A[0] if k[0] == 'l' and k[1:].isdigit()):
        P = [a[f'l{l}'] for a in A]
        nodes = '-'.join(sorted({str(g(p, 'm1_proj_partial', 'nodes') or 0) for p in P}))
        print(f'| {name(c)} | {l} | {med([g(p, "m1_proj_partial") for p in P])} ({nodes}) | {med([g(p, "m3", "acc_proj") for p in P])} | '
              f'{med([g(p, "m3", "invalid_norm_ratio") for p in P])} | {med([g(p, "m3", "invalid_q_ratio") for p in P])} | '
              f'{med([g(p, "m4_emp_proj") for p in P])} | {med([g(p, "m4_emp_both_proj") for p in P])} | {med([g(p, "m4_emp_both_addceil") for p in P])} |')
